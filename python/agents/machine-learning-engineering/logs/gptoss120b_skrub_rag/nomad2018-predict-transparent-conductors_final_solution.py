import sys
import subprocess
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# Ensure required packages
try:
    import lightgbm
except ImportError:
    install("lightgbm")
    import lightgbm

try:
    import sklearn
except ImportError:
    install("scikit-learn")
    import sklearn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor
from skrub import var, TableVectorizer

# ---------------- Custom Ensemble Estimator (sklearn-compatible) ----------------
class AverageEnsembleRegressor(BaseEstimator, RegressorMixin):
    """A simple averaging ensemble for multi-output regressors."""
    def __init__(self, estimators=None):
        self.estimators = estimators if estimators is not None else []

    def fit(self, X, y):
        for est in self.estimators:
            est.fit(X, y)
        return self

    def predict(self, X):
        preds = [est.predict(X) for est in self.estimators]
        return np.mean(preds, axis=0)

    def get_params(self, deep=True):
        return {"estimators": self.estimators}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# ---------------- Load data ----------------
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
FEATURES = [
    "spacegroup",
    "number_of_total_atoms",
    "percent_atom_al",
    "percent_atom_ga",
    "percent_atom_in",
    "lattice_vector_1_ang",
    "lattice_vector_2_ang",
    "lattice_vector_3_ang",
    "lattice_angle_alpha_degree",
    "lattice_angle_beta_degree",
    "lattice_angle_gamma_degree",
]

X_raw = train_df[FEATURES]
y_raw = train_df[TARGETS]

# ---------------- Declare DataOps variables ----------------
X = var("X_raw", X_raw).skb.mark_as_X()
y = var("y_raw", y_raw).skb.mark_as_y()

# ---------------- Build the shared preprocessing ----------------
vectorized = X.skb.apply(TableVectorizer())

# ---------------- Define two base models ----------------
lgbm = MultiOutputRegressor(
    LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=800,
        max_depth=-1,
        random_state=42,
    )
)

rf = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
)

# ---------------- Create ensemble model ----------------
ensemble = AverageEnsembleRegressor(estimators=[lgbm, rf])

# ---------------- Build the full DataOps plan ----------------
plan = vectorized.skb.apply(ensemble, y=y)

learner = plan.skb.make_learner()

# ---------------- Hold-out validation (optional) ----------------
X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=1
)

learner.fit({"X_raw": X_train_df, "y_raw": y_train_df})
y_pred_val = learner.predict({"X_raw": X_val_df})

rmsle_per_target = np.sqrt(
    np.mean((np.log1p(y_pred_val) - np.log1p(y_val_df.values)) ** 2, axis=0)
)
final_score = rmsle_per_target.mean()
print(f"Validation RMSLE (mean across targets): {final_score}")

# ---------------- Retrain on full data & predict test ----------------
learner.fit({"X_raw": X_raw, "y_raw": y_raw})

test_features = test_df[FEATURES]
test_pred = learner.predict({"X_raw": test_features})

# ---------------- Write submission ----------------
os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({
    "id": test_df["id"],
    "formation_energy_ev_natom": test_pred[:, 0],
    "bandgap_energy_ev": test_pred[:, 1],
})
submission.to_csv("./final/submission.csv", index=False)
print("Submission file written to ./final/submission.csv")