import pandas as pd
import skrub
import xgboost
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# Load the data
with open('./input/train.json', 'r') as f:
    train_data = json.load(f)

df = pd.DataFrame(train_data)
data = skrub.var("data", df)

# Define features and target
target_col = 'requester_received_pizza'

# Solution 1 features - using columns available in both train and test
feature_columns_1 = [
    "request_text_edit_aware" if "request_text_edit_aware" in df.columns else "request_text", 
    "request_title", 
    "requester_account_age_in_days_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_posts_at_request",
    "requester_upvotes_minus_downvotes_at_request"
]

# Handle case where request_text_edit_aware might not be in train but is in test
if "request_text_edit_aware" in df.columns and "request_text" in df.columns:
    # Use request_text_edit_aware if available (more appropriate for test set)
    feature_columns_1[0] = "request_text_edit_aware"
elif "request_text" in df.columns:
    feature_columns_1[0] = "request_text"

X1 = data[feature_columns_1].skb.mark_as_X()
y1 = data[target_col].skb.mark_as_y()

# Apply TableVectorizer for preprocessing (Solution 1)
X1_vec = X1.skb.apply(skrub.TableVectorizer())

# Apply XGBoost model (Solution 1)
model1 = xgboost.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
pred1 = X1_vec.skb.apply(model1, y=y1)

# Solution 2 features - using columns available in both train and test
feature_columns_2 = [
    "request_text_edit_aware" if "request_text_edit_aware" in df.columns else "request_text", 
    "request_title", 
    "requester_account_age_in_days_at_request",
    "requester_number_of_comments_at_request"
]

# Handle the text column consistently
if "request_text_edit_aware" in df.columns and "request_text" in df.columns:
    feature_columns_2[0] = "request_text_edit_aware"
elif "request_text" in df.columns:
    feature_columns_2[0] = "request_text"

X2 = data[feature_columns_2].skb.mark_as_X()
y2 = data[target_col].skb.mark_as_y()

# Apply TableVectorizer for automatic preprocessing (Solution 2)
X2_transformed = X2.skb.apply(skrub.TableVectorizer())

# Define XGBoost classifier (Solution 2)
model2 = xgboost.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)

# Apply the model inside the graph (Solution 2)
pred2 = X2_transformed.skb.apply(model2, y=y2)

# Create learners for base models
learner1 = pred1.skb.make_learner()
learner2 = pred2.skb.make_learner()

# Split data for validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])

# Fit the base learners on training data
train_data_dict = {"data": train_df}
learner1.fit(train_data_dict)
learner2.fit(train_data_dict)

# Validate on validation set
val_data_dict = {"data": val_df}
pred1_val = learner1.predict(val_data_dict)
pred2_val = learner2.predict(val_data_dict)

# Combine predictions for meta-learner validation
meta_X_val = pd.DataFrame({
    "pred1": pred1_val,
    "pred2": pred2_val
})

# Train meta-learner on training data
train_data_for_meta = {"data": train_df}
pred1_train_meta = learner1.predict(train_data_for_meta)
pred2_train_meta = learner2.predict(train_data_for_meta)

meta_X_train = pd.DataFrame({
    "pred1": pred1_train_meta,
    "pred2": pred2_train_meta
})

meta_y_train = train_df[target_col]

# Train the meta-learner
meta_learner = LogisticRegression(random_state=42)
meta_learner.fit(meta_X_train, meta_y_train)

# Validate the ensemble
final_val_predictions = meta_learner.predict_proba(meta_X_val)[:, 1]
final_validation_score = roc_auc_score(val_df[target_col], final_val_predictions)
print(f"Final Validation Performance: {final_validation_score}")

# Load test data
with open('./input/test.json', 'r') as f:
    test_data = json.load(f)

test_df = pd.DataFrame(test_data)

# Ensure we use the correct text column for test data
test_feature_columns_1 = feature_columns_1.copy()
test_feature_columns_2 = feature_columns_2.copy()

# Adjust feature columns for test data if needed
if "request_text_edit_aware" in test_df.columns:
    if "request_text" in test_feature_columns_1:
        test_feature_columns_1[test_feature_columns_1.index("request_text")] = "request_text_edit_aware"
    if "request_text" in test_feature_columns_2:
        test_feature_columns_2[test_feature_columns_2.index("request_text")] = "request_text_edit_aware"

# Generate predictions for test set using both models
pred1_test = learner1.predict({"data": test_df[test_feature_columns_1]})
pred2_test = learner2.predict({"data": test_df[test_feature_columns_2]})

# Combine predictions for meta-learner
meta_X_test = pd.DataFrame({
    "pred1": pred1_test,
    "pred2": pred2_test
})

# Final test predictions
final_test_predictions = meta_learner.predict_proba(meta_X_test)[:, 1]

# Create submission file
submission_df = pd.DataFrame({
    'request_id': test_df['request_id'],
    'requester_received_pizza': final_test_predictions
})

# Save to CSV
submission_df.to_csv('./final/submission.csv', index=False)