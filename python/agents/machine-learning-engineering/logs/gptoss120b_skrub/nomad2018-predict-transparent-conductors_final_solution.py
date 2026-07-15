# -*- coding: utf-8 -*-
# This script builds a regression model for two targets using skrub's TableVectorizer
# and predicts on the test set. It validates using RMSLE (averaged over both targets).

import os
import sys
import subprocess

# Ensure required packages are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_log_error
except ImportError:
    install("scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_log_error

try:
    import skrub
    from skrub import TableVectorizer
except ImportError:
    install("skrub")
    import skrub
    from skrub import TableVectorizer

# Paths
TRAIN_PATH = os.path.join(".", "input", "train.csv")
TEST_PATH = os.path.join(".", "input", "test.csv")
SUBMISSION_PATH = os.path.join(".", "submission.csv")

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Target columns
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# Feature columns (everything except id and targets)
FEATURES = [c for c in train_df.columns if c not in ["id"] + TARGETS]

X = train_df[FEATURES]
y = train_df[TARGETS]

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------------
# Build DataOps graph with skrub's TableVectorizer
# ----------------------------------------------------------------------
# Vectorizer that can handle mixed types; all our features are numeric but
# using TableVectorizer keeps the required skrub DataOps structure.
vectorizer = TableVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(test_df[FEATURES])

# ----------------------------------------------------------------------
# Train separate learners for each target
# ----------------------------------------------------------------------
learners = {}
val_preds = pd.DataFrame(index=y_val.index)

for target in TARGETS:
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train_vec, y_train[target])
    learners[target] = model
    val_preds[target] = model.predict(X_val_vec)

# ----------------------------------------------------------------------
# Validation metric: RMSLE averaged over both targets
# ----------------------------------------------------------------------
def rmsle(y_true, y_pred):
    # Clip to avoid negative values in log
    y_true_clipped = np.clip(y_true, a_min=0, a_max=None)
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    return np.sqrt(mean_squared_log_error(y_true_clipped, y_pred_clipped))

rmsle_scores = []
for target in TARGETS:
    score = rmsle(y_val[target].values, val_preds[target].values)
    rmsle_scores.append(score)

final_validation_score = np.mean(rmsle_scores)
print(f"Final Validation Performance: {final_validation_score}")

# ----------------------------------------------------------------------
# Retrain on the full training data and predict test set
# ----------------------------------------------------------------------
full_X_vec = vectorizer.fit_transform(X)  # refit on all data
test_predictions = pd.DataFrame({"id": test_df["id"]})

for target in TARGETS:
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(full_X_vec, y[target])
    test_predictions[target] = model.predict(X_test_vec)

# Save submission file
test_predictions.to_csv(SUBMISSION_PATH, index=False)