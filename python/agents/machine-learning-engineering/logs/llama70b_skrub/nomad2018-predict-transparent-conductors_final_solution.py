import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from skrub import var, TableVectorizer, as_data_op
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Load training data
train_df = pd.read_csv('./input/train.csv')

# Split the data into features (X) and target variables (y)
X = train_df.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
y = train_df[['formation_energy_ev_natom', 'bandgap_energy_ev']]

# Define the first model (Random Forest Regressor)
model1 = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
chain_model = RegressorChain(RandomForestRegressor(n_estimators=100, random_state=42), order=[0, 1])

# Define the second model (Gradient Boosting Regressor)
model2 = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100))

# Fit models on the full training set
model1.fit(X_scaled, y.values)  
chain_model.fit(X_scaled, y.values)

X_var_train = var('X').mark_as_X()
y_var_train = X_var_train.mark_as_y(as_data_op(y)) # Wrap y with as_data_op
vectorized_data_train = y_var_train.apply(TableVectorizer())

# Define the weighting mechanism using another skrub operation
weights = np.array([0.4, 0.6]) # assuming weights for model1 and model2 respectively

# Submission part
test_df = pd.read_csv('./input/test.csv')

X_test_scaled = scaler.transform(test_df.drop(['id'], axis=1))

test_predictions1 = model1.predict(X_test_scaled)
chain_test_predictions1 = chain_model.predict(X_test_scaled)

ensemble_test_predictions = (test_predictions1 * weights[0] + chain_test_predictions1 * weights[1])/np.sum(weights[:])

submission_df = pd.DataFrame({'id': test_df['id'], 'formation_energy_ev_natom': ensemble_test_predictions[:, 0], 'bandgap_energy_ev': ensemble_test_predictions[:, 1]})
submission_df.to_csv('./final/submission.csv', index=False)