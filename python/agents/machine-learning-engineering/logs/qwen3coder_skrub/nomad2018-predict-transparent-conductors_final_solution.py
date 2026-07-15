import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from skrub import TableVectorizer

# Create skrub variables
X_train = pd.read_csv('./input/train.csv')
X_test = pd.read_csv('./input/test.csv')

# Create target variables
y_train = X_train[['formation_energy_ev_natom', 'bandgap_energy_ev']].copy()

# Mark features and targets
feature_columns = [col for col in X_train.columns if col not in ['formation_energy_ev_natom', 'bandgap_energy_ev', 'id']]
X_train_skrub = X_train[feature_columns].copy()
X_test_skrub = X_test[feature_columns].copy()

# Apply TableVectorizer for feature preprocessing
tv = TableVectorizer()
X_train_transformed = tv.fit_transform(X_train_skrub)
X_test_transformed = tv.transform(X_test_skrub)

# Train first model
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(X_train_transformed, y_train)

# Train second model (additional model for ensembling)
model2 = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model2.fit(X_train_transformed, y_train)

# Predict on test set with both models
test_predictions_1 = model1.predict(X_test_transformed)
test_predictions_2 = model2.predict(X_test_transformed)

# Ensemble test predictions (average)
test_predictions_ensemble = (test_predictions_1 + test_predictions_2) / 2

# Create submission
submission = pd.DataFrame({
    'id': X_test['id'],
    'formation_energy_ev_natom': test_predictions_ensemble[:, 0],
    'bandgap_energy_ev': test_predictions_ensemble[:, 1]
})

# Save submission to final directory
os.makedirs('./final', exist_ok=True)
submission.to_csv('./final/submission.csv', index=False)