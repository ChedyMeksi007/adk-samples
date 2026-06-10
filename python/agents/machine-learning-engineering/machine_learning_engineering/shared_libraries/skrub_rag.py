"""RAG tool for skrub documentation retrieval using a LiteLLM-compatible endpoint.

Two retrieval stacks are supported and selected at runtime via env vars, so the
original baseline is never lost:

    RAG_CHUNKING   = fixed | structural   (default: fixed)
    RAG_RETRIEVAL  = dense | hybrid       (default: dense)
    RAG_TOKEN_BUDGET = <int tokens>       (default: 2000)  -- caps context sent to LLM
    RAG_MIN_SCORE    = <float cosine>     (default: 0.2)   -- dense candidate filter

Baseline (default): fixed-size chunks (1000/200) -> qwen3-embedding:8b ->
FAISS IndexFlatIP dense top-5.

Second stack (RAG_CHUNKING=structural, RAG_RETRIEVAL=hybrid): structural
variable-size chunks (RST sections + Python AST + numpydoc docstrings) ->
FAISS IndexFlatIP + BM25Okapi -> Hybrid Reciprocal Rank Fusion (20 dense + 20
BM25 candidates) -> filter + dedup + token budget -> top-5 chunks to the LLM.

Each chunking mode has its own on-disk artifacts so both can coexist:
    skrub_rag_index{suffix}.faiss   skrub_rag_chunks{suffix}.pkl   skrub_rag_bm25{suffix}.pkl
where suffix = "" for fixed and "_structural" for structural.
"""

import ast
import os
import pickle
import re
import time

import faiss
import numpy as np
import requests

_DIR = os.path.dirname(os.path.abspath(__file__))
_SKRUB_DOCS_DIR = os.path.join(_DIR, "..", "..", "tmp", "skrub-docs")

_EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:8b")
_EMBED_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:11434/v1")
# Strip /chat/completions to get base URL, then use /embeddings
_BASE_URL = _EMBED_ENDPOINT.replace("/chat/completions", "")
_API_KEY = os.getenv("API_KEY", "")

# ---- runtime toggles (defaults preserve the original fixed+dense behaviour) ----
_CHUNKING = os.getenv("RAG_CHUNKING", "fixed").lower()
_RETRIEVAL = os.getenv("RAG_RETRIEVAL", "dense").lower()
_TOKEN_BUDGET = int(os.getenv("RAG_TOKEN_BUDGET", "2000"))   # max tokens of context returned
_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.2"))        # dense cosine floor for candidates

# External web source (SearXNG) — off by default so the second-type behaviour is unchanged.
# Only used in hybrid retrieval. If disabled/unset/unreachable it is skipped gracefully.
_WEB_ENABLED = os.getenv("RAG_WEB", "0").lower() in ("1", "true", "yes")
_SEARXNG_URL = os.getenv("SEARXNG_URL", "").rstrip("/")
_WEB_TOP = int(os.getenv("RAG_WEB_TOP", "10"))  # SearXNG results requested before preprocessing

# Source priority used when structuring the final context (token budget split):
#   internal = dense skrub-docs, keyword = BM25 API matches, web = SearXNG.
def _parse_weights():
    try:
        i, k, w = (float(x) for x in os.getenv("RAG_WEIGHTS", "0.60,0.25,0.15").split(","))
        return {"internal": i, "keyword": k, "web": w}
    except Exception:
        return {"internal": 0.60, "keyword": 0.25, "web": 0.15}

_WEIGHTS = _parse_weights()

# retrieval sizing
_DENSE_CAND = 20     # dense candidates before fusion
_BM25_CAND = 20      # BM25 candidates before fusion
_FINAL_K = 5         # chunks handed to the LLM
_RRF_K = 60          # Reciprocal Rank Fusion constant

# chunking sizing
_FIXED_SIZE = 1000
_FIXED_OVERLAP = 200
_MAX_STRUCT_CHARS = 2000   # structural chunks larger than this fall back to fixed splitting
_MIN_CHUNK_CHARS = 50

# cache: chunking mode -> {"index", "chunks", "bm25"(lazy), "paths"}
_cache = {}


def _paths(chunking):
    suffix = "_structural" if chunking == "structural" else ""
    return (
        os.path.join(_DIR, f"skrub_rag_index{suffix}.faiss"),
        os.path.join(_DIR, f"skrub_rag_chunks{suffix}.pkl"),
        os.path.join(_DIR, f"skrub_rag_bm25{suffix}.pkl"),
    )


def _embed_texts(texts, batch_size=32):
    """Embed texts using the configured Ollama endpoint with retry/backoff."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                resp = requests.post(
                    f"{_BASE_URL}/embeddings",
                    headers={"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"},
                    json={"model": _EMBED_MODEL, "input": batch},
                    timeout=120,
                )
                resp.raise_for_status()
                break
            except Exception as e:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"  Embedding attempt {attempt + 1}/5 failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding failed after 5 attempts for batch {i}–{i+batch_size}")
        data = resp.json()["data"]
        batch_embeddings = [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        time.sleep(0.2)
    return np.array(all_embeddings, dtype="float32")


def _read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _tokenize(text):
    """Tokenizer for BM25 — keeps API identifiers (GapEncoder, n_components) intact."""
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


# --------------------------------------------------------------------------- #
# Chunking                                                                     #
# --------------------------------------------------------------------------- #
def _chunk_text(text, source, chunk_size=_FIXED_SIZE, overlap=_FIXED_OVERLAP):
    """Fixed-size overlapping chunks (baseline)."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        if len(chunk.strip()) > _MIN_CHUNK_CHARS:
            chunks.append({"text": chunk, "source": source})
    return chunks


def _add_sized(chunks, text, source, max_chars=_MAX_STRUCT_CHARS):
    """Append a structural unit; if oversized, fall back to fixed splitting."""
    text = text.strip()
    if len(text) <= _MIN_CHUNK_CHARS:
        return
    if len(text) <= max_chars:
        chunks.append({"text": text, "source": source})
    else:
        chunks.extend(_chunk_text(text, source))


_ADORN = set('=-~^"#*+.:_`')


def _is_adornment(line):
    s = line.rstrip("\n")
    return len(s) >= 3 and len(set(s)) == 1 and s[0] in _ADORN


def _chunk_rst(text, source):
    """Structural RST/numpydoc chunking: split at section headings (title + adornment)."""
    lines = text.splitlines(keepends=True)
    # heading = non-empty, non-adornment line immediately followed by an adornment line
    starts = []
    for i in range(len(lines) - 1):
        if lines[i].strip() and not _is_adornment(lines[i]) and _is_adornment(lines[i + 1]):
            starts.append(i)
    if not starts:
        chunks = []
        _add_sized(chunks, text, source)
        return chunks
    boundaries = [0] + starts + [len(lines)]
    # dedupe/sort boundaries and build sections
    boundaries = sorted(set(boundaries))
    chunks = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        _add_sized(chunks, "".join(lines[a:b]), source)
    return chunks


def _chunk_python(text, source):
    """Structural Python chunking via AST: functions/classes as whole units."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _chunk_text(text, source)  # unparseable -> fixed fallback

    chunks = []
    doc = ast.get_docstring(tree, clean=False)
    if doc:
        _add_sized(chunks, doc, source)

    body = tree.body
    start = 0
    if (
        doc and body and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        start = 1  # skip the module docstring node (already added)

    buf = []  # run of top-level non-def statements grouped together

    def flush():
        if buf:
            _add_sized(chunks, "".join(buf), source)
            buf.clear()

    for node in body[start:]:
        seg = ast.get_source_segment(text, node)
        if seg is None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush()
            _add_sized(chunks, seg, source)
        else:
            buf.append(seg + "\n")
    flush()
    return chunks or _chunk_text(text, source)


def _collect_chunks(chunking):
    """Build the chunk list for the given chunking mode."""
    structural = chunking == "structural"
    all_chunks = []

    def handle(path, kind):
        text = _read_file(path)
        if not text:
            return
        source = os.path.relpath(path, _SKRUB_DOCS_DIR)
        if not structural:
            all_chunks.extend(_chunk_text(text, source))
        elif kind == "rst":
            all_chunks.extend(_chunk_rst(text, source))
        else:  # python
            all_chunks.extend(_chunk_python(text, source))

    # RST docs (+ any .py under doc/)
    for root, _, files in os.walk(os.path.join(_SKRUB_DOCS_DIR, "doc")):
        for fname in files:
            if fname.endswith(".rst"):
                handle(os.path.join(root, fname), "rst")
            elif fname.endswith(".py"):
                handle(os.path.join(root, fname), "py")

    # Examples
    examples_dir = os.path.join(_SKRUB_DOCS_DIR, "examples")
    if os.path.exists(examples_dir):
        for root, _, files in os.walk(examples_dir):
            for fname in files:
                if fname.endswith(".py"):
                    handle(os.path.join(root, fname), "py")

    # Installed skrub docstrings (numpydoc)
    import inspect
    import skrub

    for name, obj in inspect.getmembers(skrub):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            doc = inspect.getdoc(obj)
            if doc and len(doc) > _MIN_CHUNK_CHARS:
                src = f"skrub.{name} docstring"
                if structural:
                    all_chunks.extend(_chunk_rst(doc, src))
                else:
                    all_chunks.extend(_chunk_text(doc, src))

    return all_chunks


def build_index(chunking=None):
    """Build FAISS (dense) + BM25 (sparse) indexes for a chunking mode."""
    chunking = (chunking or _CHUNKING).lower()
    faiss_path, chunks_path, bm25_path = _paths(chunking)

    all_chunks = _collect_chunks(chunking)
    print(f"[{chunking}] Total chunks: {len(all_chunks)}")
    print(f"Embedding with {_EMBED_MODEL} via {_BASE_URL}/embeddings ...")

    texts = [c["text"] for c in all_chunks]
    embeddings = _embed_texts(texts)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, faiss_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Dense index -> {faiss_path} ({index.ntotal} vectors, dim={embeddings.shape[1]})")

    _build_bm25(all_chunks, bm25_path)
    print(f"BM25 index  -> {bm25_path}")
    print(f"Chunks      -> {chunks_path}")


def _build_bm25(chunks, bm25_path):
    """Build and persist a BM25Okapi index over the chunk texts (no embeddings needed)."""
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)


def build_bm25_from_chunks(chunking="fixed"):
    """Build only the BM25 index from already-saved chunks (cheap, no endpoint calls)."""
    _, chunks_path, bm25_path = _paths(chunking)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    _build_bm25(chunks, bm25_path)
    print(f"BM25 index -> {bm25_path} ({len(chunks)} chunks)")


# --------------------------------------------------------------------------- #
# Loading / retrieval                                                          #
# --------------------------------------------------------------------------- #
def _load(chunking, need_bm25):
    entry = _cache.get(chunking)
    if entry is None:
        faiss_path, chunks_path, bm25_path = _paths(chunking)
        index = faiss.read_index(faiss_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        entry = {"index": index, "chunks": chunks, "bm25": None, "bm25_path": bm25_path}
        _cache[chunking] = entry
    if need_bm25 and entry["bm25"] is None:
        with open(entry["bm25_path"], "rb") as f:
            entry["bm25"] = pickle.load(f)
    return entry


def _resolve_mode():
    """Pick chunking/retrieval, degrading gracefully if artifacts are missing."""
    chunking, retrieval = _CHUNKING, _RETRIEVAL
    faiss_path, _, bm25_path = _paths(chunking)
    if not os.path.exists(faiss_path):
        if chunking != "fixed":
            print(f"  [rag] {chunking} index missing, falling back to fixed")
        chunking = "fixed"
        _, _, bm25_path = _paths(chunking)
    if retrieval == "hybrid" and not os.path.exists(bm25_path):
        print("  [rag] BM25 index missing, falling back to dense retrieval")
        retrieval = "dense"
    return chunking, retrieval


def _dedup_ok(norm, seen):
    """True if `norm` is not an exact/containment duplicate of anything in `seen`."""
    return not any(norm == s or norm in s or s in norm for s in seen)


def _pack_context(candidates):
    """Dense-only packing: dedup + token budget over ranked (chunk, score) items."""
    selected, seen, used = [], [], 0
    for chunk, score in candidates:
        text = chunk["text"].strip()
        norm = " ".join(text.lower().split())
        if not _dedup_ok(norm, seen):
            continue
        tokens = max(1, len(text) // 4)  # ~4 chars/token
        if not selected and tokens > _TOKEN_BUDGET:
            text = text[: _TOKEN_BUDGET * 4]  # single oversized chunk: truncate to budget
            tokens = _TOKEN_BUDGET
        elif selected and used + tokens > _TOKEN_BUDGET:
            continue
        selected.append((chunk, score, text, "internal"))
        seen.append(norm)
        used += tokens
        if len(selected) >= _FINAL_K:
            break
    return selected


# --------------------------------------------------------------------------- #
# External web source (SearXNG)                                                #
# --------------------------------------------------------------------------- #
def _clean_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _domain(url):
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc or url
    except Exception:
        return url


def _web_search(query):
    """Query SearXNG and preprocess results into short web chunks; [] on any failure.

    Preprocessing: strip HTML, collapse whitespace, drop empties, dedup by URL and domain.
    Chunking: SearXNG returns short, self-contained snippets, so each result becomes one
    small chunk (size-cap fallback only for unusually long content — no page fetch needed).
    """
    if not _WEB_ENABLED or not _SEARXNG_URL:
        return []
    try:
        resp = requests.get(
            f"{_SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:
        print(f"  [rag] SearXNG unavailable ({e}); skipping web source")
        return []

    chunks, seen_urls, seen_domains = [], set(), set()
    for r in results[: _WEB_TOP]:
        url = r.get("url", "")
        content = _clean_html(r.get("content", ""))
        if len(content) < 20:  # drop empty / too-thin snippets
            continue
        dom = _domain(url)
        if url in seen_urls or dom in seen_domains:  # dedup by URL and domain
            continue
        seen_urls.add(url)
        seen_domains.add(dom)
        title = _clean_html(r.get("title", ""))
        text = f"{title}\n{content}" if title else content
        if len(text) > _MAX_STRUCT_CHARS:
            text = text[:_MAX_STRUCT_CHARS]
        chunks.append({"text": text, "source": f"web:{dom}", "url": url})
    return chunks


# --------------------------------------------------------------------------- #
# Weighted RRF over three sources + source-priority context structuring        #
# --------------------------------------------------------------------------- #
def _hybrid_ranked(query, query_vec, entry):
    """Fuse dense (internal) + BM25 (keyword) + SearXNG (web) with source weights (RRF)."""
    index, chunks = entry["index"], entry["chunks"]

    d_scores, d_idx = index.search(query_vec, _DENSE_CAND)
    dense_ranked = [int(i) for i, sc in zip(d_idx[0], d_scores[0]) if i >= 0 and sc >= _MIN_SCORE]

    bm_scores = entry["bm25"].get_scores(_tokenize(query))
    bm25_ranked = [int(i) for i in np.argsort(bm_scores)[::-1][:_BM25_CAND] if bm_scores[i] > 0]

    web_chunks = _web_search(query)

    # renormalize weights over the sources that actually returned candidates
    active = {"internal": bool(dense_ranked), "keyword": bool(bm25_ranked), "web": bool(web_chunks)}
    wsum = sum(_WEIGHTS[o] for o, on in active.items() if on) or 1.0
    weights = {o: (_WEIGHTS[o] / wsum if active[o] else 0.0) for o in _WEIGHTS}

    fused = {}  # key -> {"score", "chunk", "origin"}

    def add(key, rank, origin, chunk):
        contrib = weights[origin] * (1.0 / (_RRF_K + rank + 1))
        e = fused.get(key)
        if e is None:  # first-seen origin sticks (dense added before BM25 -> "internal" wins ties)
            fused[key] = {"score": contrib, "chunk": chunk, "origin": origin}
        else:
            e["score"] += contrib

    for rank, idx in enumerate(dense_ranked):
        add(("i", idx), rank, "internal", chunks[idx])
    for rank, idx in enumerate(bm25_ranked):
        add(("i", idx), rank, "keyword", chunks[idx])
    for rank, wc in enumerate(web_chunks):
        add(("w", wc["url"]), rank, "web", wc)

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True), weights


def _pack_by_source(items, weights):
    """Structure context with per-source token budget (e.g. 60/25/15), dedup, then spill."""
    budgets = {o: int(_TOKEN_BUDGET * w) for o, w in weights.items()}
    used = {o: 0 for o in weights}
    selected, seen, chosen = [], [], set()

    def prep(it):
        text = it["chunk"]["text"].strip()
        norm = " ".join(text.lower().split())
        return (text, norm) if _dedup_ok(norm, seen) else None

    # pass 1: fill each source up to its share of the budget
    for it in items:
        o = it["origin"]
        p = prep(it)
        if p is None:
            continue
        text, norm = p
        tokens = max(1, len(text) // 4)
        if used[o] == 0 and tokens > budgets[o] and budgets[o] > 0:
            text = text[: budgets[o] * 4]
            tokens = budgets[o]
        if used[o] + tokens <= budgets[o]:
            selected.append((it["chunk"], it["score"], text, o))
            seen.append(norm)
            used[o] += tokens
            chosen.add(id(it))

    # pass 2: spill any unused total budget with the best remaining items (any source)
    total_used = sum(used.values())
    for it in items:
        if id(it) in chosen:
            continue
        p = prep(it)
        if p is None:
            continue
        text, norm = p
        tokens = max(1, len(text) // 4)
        if total_used + tokens > _TOKEN_BUDGET:
            continue
        selected.append((it["chunk"], it["score"], text, it["origin"]))
        seen.append(norm)
        total_used += tokens
    return selected


def _format(selected):
    return "\n\n---\n\n".join(
        f"## Source [{origin}]: {c['source']} (score: {score:.3f})\n{text}"
        for c, score, text, origin in selected
    )


def search_skrub_docs(query: str) -> str:
    """Search the skrub documentation for relevant information.

    Use this tool to find skrub API usage, examples, and best practices
    for building DataOps pipelines with TableVectorizer, encoders,
    fuzzy_join, and other skrub components.

    Args:
        query: The search query about skrub functionality.

    Returns:
        Relevant documentation excerpts from skrub.
    """
    chunking, retrieval = _resolve_mode()
    entry = _load(chunking, need_bm25=(retrieval == "hybrid"))
    index, chunks = entry["index"], entry["chunks"]

    query_vec = _embed_texts([query])
    faiss.normalize_L2(query_vec)

    if retrieval == "hybrid":
        items, weights = _hybrid_ranked(query, query_vec, entry)
        selected = _pack_by_source(items, weights)
    else:
        # dense-only: retrieve a candidate pool, filter by cosine floor
        scores, idxs = index.search(query_vec, _DENSE_CAND)
        candidates = [
            (chunks[int(i)], float(sc))
            for i, sc in zip(idxs[0], scores[0])
            if 0 <= i < len(chunks) and sc >= _MIN_SCORE
        ]
        selected = _pack_context(candidates)
    return _format(selected)


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else _CHUNKING
    if arg == "bm25-fixed":
        build_bm25_from_chunks("fixed")
    else:
        build_index(arg)  # "fixed" or "structural"
