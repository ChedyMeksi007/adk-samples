#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate submission for the Reddit pizza request prediction competition.
The workflow uses skrub's DataOps style (skrub.var → .skb.apply → .skb.make_learner)
for the TableVectorizer step, then a scikit-learn classifier for training.
"""

import subprocess
import sys
import os
import json

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install required packages if missing
try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
except ImportError:
    install("scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

try:
    import skrub
    from skrub import TableVectorizer, var
except ImportError:
    install("skrub")
    from skrub import TableVectorizer, var

# ------------------------------
# Paths
# ------------------------------
INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_PATH = os.path.join(INPUT_DIR, "test.json")
OUTPUT_DIR = "./final"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Load data
# ------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

train_df = load_json(TRAIN_PATH)
test_df = load_json(TEST_PATH)

# ------------------------------
# Feature selection
# ------------------------------
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]
feature_cols = [c for c in feature_cols if c in test_df.columns]

X = train_df[feature_cols]
y = train_df[TARGET_COL].astype(int)
X_test = test_df[feature_cols]

# ------------------------------
# Train/validation split (optional, just to show score)
# ------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Build DataOps graph for vectorization
# ------------------------------
X_var = var("X")
y_var = var("y")

vectorizer = TableVectorizer()
X_vec_var = X_var.skb.apply(vectorizer)               # creates a transformed variable
# Attach a classifier to the graph
clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",
    solver="lbfgs"
)
pipeline_var = X_vec_var.skb.apply(clf, y=y_var)     # adds the estimator
learner = pipeline_var.skb.make_learner()            # final learner

# ------------------------------
# Fit on validation split to report score
# ------------------------------
learner_val = learner.fit({"X": X_tr, "y": y_tr})
val_probs = learner_val.predict_proba({"X": X_val})[:, 1]   # probability of class 1
val_score = roc_auc_score(y_val, val_probs)
print(f"Final Validation Performance: {val_score:.6f}")

# ------------------------------
# Retrain on full training data
# ------------------------------
learner_full = learner.fit({"X": X, "y": y})

# ------------------------------
# Predict on test set (probabilities)
# ------------------------------
test_probs = learner_full.predict_proba({"X": X_test})[:, 1]

# ------------------------------
# Write submission
# ------------------------------
submission = pd.DataFrame({
    ID_COL: test_df[ID_COL],
    TARGET_COL: test_probs
})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Submission written to {OUTPUT_PATH}")