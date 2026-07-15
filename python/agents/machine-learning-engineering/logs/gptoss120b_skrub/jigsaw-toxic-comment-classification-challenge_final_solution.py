import sys
import subprocess
import importlib
import warnings
from pathlib import Path

def install_and_import(pkg_name):
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
    finally:
        globals()[pkg_name] = importlib.import_module(pkg_name)

# Ensure required packages are available
for pkg in ["sklearn", "skrub", "scipy"]:
    install_and_import(pkg)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from skrub import var, TableVectorizer

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
DATA_DIR = Path("./input")
train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

TARGETS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COL = "comment_text"

# --------------------------------------------------
# 2. Train/validation split (stratified on the most frequent label per row)
# --------------------------------------------------
train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df[TARGETS].idxmax(axis=1)
)

X_train_raw = train_split[[TEXT_COL]]
X_val_raw = val_split[[TEXT_COL]]
X_test_raw = test_df[[TEXT_COL]]

y_train = train_split[TARGETS].values
y_val = val_split[TARGETS].values

# --------------------------------------------------
# 3. Build DataOps graph (skrub style)
# --------------------------------------------------
# Define skrub variables
raw_X = var("raw_X").skb.mark_as_X()
raw_y = var("raw_y").skb.mark_as_y()

# Define preprocessing and model
vectorizer = TableVectorizer()
base_clf = LogisticRegression(max_iter=1000, n_jobs=5, solver="liblinear")
multi_clf = MultiOutputClassifier(base_clf, n_jobs=5)

# Assemble learner via DataOps graph, explicitly passing y to the model node
learner = (
    raw_X
    .skb.apply(vectorizer)                     # turn text into a numeric matrix
    .skb.apply(multi_clf, y=raw_y)             # multi-output classifier receives y
    .skb.make_learner()
)

# --------------------------------------------------
# 4. Fit on training split
# --------------------------------------------------
learner.fit({"raw_X": X_train_raw, "raw_y": y_train})

# --------------------------------------------------
# 5. Validation evaluation
# --------------------------------------------------
val_proba_list = learner.predict_proba({"raw_X": X_val_raw})
val_pred_proba = np.column_stack([p[:, 1] for p in val_proba_list])

auc_per_label = [
    roc_auc_score(y_val[:, i], val_pred_proba[:, i]) for i in range(len(TARGETS))
]
final_validation_score = np.mean(auc_per_label)
print(f"Final Validation Performance: {final_validation_score:.6f}")

# --------------------------------------------------
# 6. Retrain on full training data
# --------------------------------------------------
full_X_raw = train_df[[TEXT_COL]]
full_y = train_df[TARGETS].values

full_X = var("full_X").skb.mark_as_X()
full_y_var = var("full_y").skb.mark_as_y()

learner_full = (
    full_X
    .skb.apply(vectorizer)                     # refit vectorizer on full data
    .skb.apply(multi_clf, y=full_y_var)        # pass full target
    .skb.make_learner()
)

learner_full.fit({"full_X": full_X_raw, "full_y": full_y})

# --------------------------------------------------
# 7. Predict on test set and write submission
# --------------------------------------------------
test_proba_list = learner_full.predict_proba({"full_X": X_test_raw})
test_pred_proba = np.column_stack([p[:, 1] for p in test_proba_list])

submission = pd.DataFrame(
    {"id": test_df["id"], **{t: test_pred_proba[:, idx] for idx, t in enumerate(TARGETS)}}
)

output_dir = Path("./final")
output_dir.mkdir(parents=True, exist_ok=True)
submission_path = output_dir / "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path.resolve()}")