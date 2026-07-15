import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import skrub

# Paths
INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Target columns
TARGETS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ------ Build a skrub DataOps graph for the text column ------
# X variable (features)
X_var = skrub.var("X", train_df[["comment_text"]]).skb.mark_as_X()
# y variable (targets)
y_var = skrub.var("y", train_df[TARGETS]).skb.mark_as_y()

# Apply TableVectorizer inside the graph
vectorized = X_var.skb.apply(skrub.TableVectorizer())
# -------------------------------------------------------------

# ------ Fit the vectorizer and transform train / test -------
vectorizer = vectorized.skb.preview()  # obtains the TableVectorizer instance
vectorizer.fit(train_df["comment_text"])

X_train_vect = vectorizer.transform(train_df["comment_text"])
X_test_vect = vectorizer.transform(test_df["comment_text"])
# -------------------------------------------------------------

# ------ Validation split for performance estimation ----------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_vect,
    train_df[TARGETS],
    test_size=0.2,
    random_state=42,
    stratify=train_df["toxic"],
)

# One-vs-rest Logistic Regression
base_clf = LogisticRegression(max_iter=1000, n_jobs=5, class_weight="balanced")
model = OneVsRestClassifier(base_clf)

model.fit(X_tr, y_tr)

# Predict probabilities on validation set
val_pred_proba = model.predict_proba(X_val)

# Compute ROC-AUC for each label, then average
auc_scores = []
for i, col in enumerate(TARGETS):
    auc = roc_auc_score(y_val[col], val_pred_proba[:, i])
    auc_scores.append(auc)

final_validation_score = np.mean(auc_scores)
print(f"Final Validation Performance: {final_validation_score:.6f}")
# -------------------------------------------------------------

# ------ Retrain on full data and predict test set ----------
model.fit(X_train_vect, train_df[TARGETS])
test_pred_proba = model.predict_proba(X_test_vect)

# Build submission file
submission = pd.DataFrame(test_pred_proba, columns=TARGETS)
submission.insert(0, "id", test_df["id"])
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
# -------------------------------------------------------------