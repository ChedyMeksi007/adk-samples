import pandas as pd
import numpy as np
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import os

# Load data
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

# Identify target columns
target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Create a validation split
train_df_shuffled = train_df.sample(frac=1, random_state=42)
split_idx = int(0.8 * len(train_df_shuffled))
train_split = train_df_shuffled[:split_idx]
val_split = train_df_shuffled[split_idx:]

# Mark features and targets for training data
X_train = train_split.drop(columns=target_columns + ['id'])
y_train_dict = {col: train_split[col] for col in target_columns}

# Mark features for test data
X_test = test_df.drop(columns=['id'])

# Apply TableVectorizer
vectorizer = TableVectorizer()

# Fit the vectorizer on training features
vectorizer.fit(X_train)

# Transform features
X_train_vec = vectorizer.transform(X_train)
X_val_vec = vectorizer.transform(val_split.drop(columns=target_columns + ['id']))
X_test_vec = vectorizer.transform(X_test)

# Train models for each target
models = []
val_preds = []

for target_col in target_columns:
    y_train = y_train_dict[target_col]
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train_vec, y_train)
    models.append(model)
    
    # Get validation predictions for scoring
    val_pred = model.predict(X_val_vec)
    val_preds.append(val_pred)

# Calculate validation score
val_true = val_split[target_columns].values
val_preds_array = np.column_stack(val_preds)
final_validation_score = roc_auc_score(val_true, val_preds_array, average='macro')
print(f"Final Validation Performance: {final_validation_score}")

# Make predictions for test set
test_preds = []
for model in models:
    pred = model.predict(X_test_vec)
    test_preds.append(pred)

# Create submission
submission = test_df[['id']].copy()
for i, target_col in enumerate(target_columns):
    submission[target_col] = test_preds[i]

# Save submission
os.makedirs('./final', exist_ok=True)
submission.to_csv('./final/submission.csv', index=False)