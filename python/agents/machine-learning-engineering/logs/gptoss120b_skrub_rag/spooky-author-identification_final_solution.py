# -*- coding: utf-8 -*-
"""
Ensemble solution using skrub DataOps:
- Shared TableVectorizer for full training & test
- Two base learners: LogisticRegression and skrub.tabular_pipeline("classification")
- Meta-learner (LogisticRegression) trained on out-of-fold predictions
"""

import os
import sys
import subprocess

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

# Install required packages if missing
for pkg in ["pandas", "numpy", "scikit-learn", "skrub"]:
    try:
        __import__(pkg.split("-")[0])
    except ImportError:
        pip_install(pkg)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import skrub
from skrub import TableVectorizer, var

# ----------------------------------
# 1. Load data
# ----------------------------------
INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
OUTPUT_DIR = "./final"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ----------------------------------
# 2. Encode target
# ----------------------------------
le = LabelEncoder()
train_df["author_enc"] = le.fit_transform(train_df["author"])
class_names = le.classes_.tolist()          # order for submission & metrics

X = train_df[["text"]].copy()
y = train_df["author_enc"].copy()

# ----------------------------------
# 3. Shared TableVectorizer for full data (used later for final fit & test)
# ----------------------------------
X_full_var = var("X_all", X).skb.mark_as_X()
vec_full = X_full_var.skb.apply(TableVectorizer())   # not fitted yet

# ----------------------------------
# 4. Base-learner factories
# ----------------------------------
logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)

auto_pipe = skrub.tabular_pipeline("classification")

# ----------------------------------
# 5. Out-of-fold predictions for meta-learner training
# ----------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = []          # list of np.ndarray (proba_A concat proba_B)
oof_targets = []        # list of int labels

for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ---- create X variable that matches the current training split ----
    X_tr_var = var("X_all", X_tr).skb.mark_as_X()
    vec_tr = X_tr_var.skb.apply(TableVectorizer())

    # ---- base learner A (logistic regression) ----
    y_tr_var = var("y", y_tr).skb.mark_as_y()
    base_A = vec_tr.skb.apply(logreg, y=y_tr_var)
    learner_A = base_A.skb.make_learner()
    learner_A.fit({"X_all": X_tr, "y": y_tr})

    # ---- base learner B (skrub auto pipeline) ----
    y_tr_var_b = var("y", y_tr).skb.mark_as_y()
    base_B = vec_tr.skb.apply(auto_pipe, y=y_tr_var_b)
    learner_B = base_B.skb.make_learner()
    learner_B.fit({"X_all": X_tr, "y": y_tr})

    # ---- predictions on validation fold ----
    prob_A = learner_A.predict_proba({"X_all": X_val})
    prob_B = learner_B.predict_proba({"X_all": X_val})

    prob_AB = np.concatenate([prob_A, prob_B], axis=1)

    oof_preds.append(prob_AB)
    oof_targets.append(y_val.values)

# Aggregate OOF tensors
P_oof = np.vstack(oof_preds)                 # (n_samples, 2 * n_classes)
y_oof = np.concatenate(oof_targets)         # (n_samples,)

# ----------------------------------
# 6. Meta-learner (logistic regression) on OOF predictions
# ----------------------------------
meta_X_var = var("X_meta", P_oof).skb.mark_as_X()
meta_y_var = var("y_oof", y_oof).skb.mark_as_y()
meta_pred = meta_X_var.skb.apply(
    LogisticRegression(multi_class="multinomial", max_iter=2000, solver="lbfgs"),
    y=meta_y_var,
)
meta_learner = meta_pred.skb.make_learner()
meta_learner.fit({"X_meta": P_oof, "y_oof": y_oof})

# ----------------------------------
# 7. Validation metric (log-loss) on the OOF data using meta-learner
# ----------------------------------
meta_oof_proba = meta_learner.predict_proba({"X_meta": P_oof})
val_logloss = log_loss(y_oof, meta_oof_proba, labels=np.arange(len(class_names)))
print(f"Final Validation Performance: {val_logloss}")

# ----------------------------------
# 8. Fit base learners on the full training data
# ----------------------------------
full_y_var = var("y", y).skb.mark_as_y()

# Base A full
base_A_full = vec_full.skb.apply(logreg, y=full_y_var)
learner_A_full = base_A_full.skb.make_learner()
learner_A_full.fit({"X_all": X, "y": y})

# Base B full
base_B_full = vec_full.skb.apply(auto_pipe, y=full_y_var)
learner_B_full = base_B_full.skb.make_learner()
learner_B_full.fit({"X_all": X, "y": y})

# ----------------------------------
# 9. Test-time predictions and meta-ensemble
# ----------------------------------
test_X = test_df[["text"]].copy()

test_prob_A = learner_A_full.predict_proba({"X_all": test_X})
test_prob_B = learner_B_full.predict_proba({"X_all": test_X})
test_prob_AB = np.concatenate([test_prob_A, test_prob_B], axis=1)

# Direct prediction with the already-trained meta-learner
final_proba = meta_learner.predict_proba({"X_meta": test_prob_AB})

# ----------------------------------
# 10. Build submission file
# ----------------------------------
submission = pd.DataFrame(final_proba, columns=class_names)
submission.insert(0, "id", test_df["id"])
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Submission written to {OUTPUT_PATH}")