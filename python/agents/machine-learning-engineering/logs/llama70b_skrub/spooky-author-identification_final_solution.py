# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import numpy as np

# Load the dataset
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['author'], test_size=0.2, random_state=42)

# Define the vectorizer and model for solution 1
vectorizer_solution_1 = TfidfVectorizer()
model_solution_1 = MultinomialNB()

# Apply vectorization to the data for solution 1
X_train_vectorized_solution_1 = vectorizer_solution_1.fit_transform(X_train)
y_train_encoded_solution_1 = np.array([0 if x == 'EAP' else 1 if x == 'HPL' else 2 for x in y_train])

# Train the model on the vectorized data for solution 1
model_solution_1.fit(X_train_vectorized_solution_1, y_train_encoded_solution_1)

# Make predictions on the validation set for solution 1
X_val_vectorized_solution_1 = vectorizer_solution_1.transform(X_val)
y_pred_val_solution_1 = model_solution_1.predict_proba(X_val_vectorized_solution_1)

# Define the vectorizer and model for solution 2
vectorizer_solution_2 = TfidfVectorizer(stop_words='english')
model_solution_2 = LogisticRegression(max_iter=10000)

# Apply vectorization to the data for solution 2
X_train_vectorized_solution_2 = vectorizer_solution_2.fit_transform(X_train)
y_mapped_train_values_solution_2 = np.array([0 if x == 'EAP' else 1 if x == 'HPL' else 2 for x in y_train])

# Train the model on the training data for solution 2
model_solution_2.fit(X_train_vectorized_solution_2, y_mapped_train_values_solution_2)

# Make predictions on the validation set for solution 2
X_val_vectorized_solution_2 = vectorizer_solution_2.transform(X_val)
y_pred_val_solution_2 = model_solution_2.predict_proba(X_val_vectorized_solution_2)

# Define weights for both models
weight_solution_1 = 0.5
weight_solution_2 = 0.5

# Apply weighted average to combine predictions from both models on the validation set
weightedPred_val = weight_solution_1 * y_pred_val_solution_1 + weight_solution_2 * y_pred_val_solution_2

# Calculate log loss for the validation set using weighted predictions
yMappedValValues = np.array([0 if x == 'EAP' else 1 if x == 'HPL' else 2 for x in y_val])
final_validation_score = log_loss(yMappedValValues, weightedPred_val)
print(f'Final Validation Performance: {final_validation_score}')

# Make predictions on the test set
X_test_vectorized_solution_1 = vectorizer_solution_1.transform(test_df['text'])
test_pred_solution_1 = model_solution_1.predict_proba(X_test_vectorized_solution_1)

X_test_vectorized_solution_2 = vectorizer_solution_2.transform(test_df['text'])
test_pred_solution_2 = model_solution_2.predict_proba(X_test_vectorized_solution_2)

# Apply weighted average to combine predictions from both models on the test set
weightedPred_test = weight_solution_1 * test_pred_solution_1 + weight_solution_2 * test_pred_solution_2

# Create submission dataframes for each solution (keeping original format)
submission_df_solution_1 = pd.DataFrame({'id': test_df['id'], 
                                         'EAP': test_pred_solution_1[:, 0], 
                                         'HPL': test_pred_solution_1[:, 1], 
                                         'MWS': test_pred_solution_1[:, 2]})

submission_df_solution_2 = pd.DataFrame(test_pred_solution_2, columns=['EAP', 'HPL', 'MWS'])
submission_df_solution_2['id'] = test_df['id']
submission_df_solution_2 = submission_df_solution_2[['id', 'EAP', 'HPL', 'MWS']]

# Save the submission dataframes to csv files
submission_df_solution_1.to_csv('submission_solution_1.csv', index=False)
submission_df_solution_2.to_csv('submission_solution_2.csv', index=False)

# Create a final submission dataframe based on weighted test predictions
final_submission_df = pd.DataFrame({'id': test_df['id'], 
                                    'EAP': weightedPred_test[:, 0], 
                                    'HPL': weightedPred_test[:, 1], 
                                    'MWS': weightedPred_test[:, 2]})

# Save the final submission dataframe to a csv file
final_submission_df.to_csv('./final/submission.csv', index=False)