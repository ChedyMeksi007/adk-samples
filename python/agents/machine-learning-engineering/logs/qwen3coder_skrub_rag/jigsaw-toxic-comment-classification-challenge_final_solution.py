import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import skrub
import numpy as np

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Create target columns list
target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Create skrub variables
train_data = skrub.var("train_data", train_df)
test_data = skrub.var("test_data", test_df)

# Mark features and targets
X_train = train_data.skb.select("comment_text").skb.mark_as_X()
y_train = train_data.skb.select(target_columns).skb.mark_as_y()

# Apply TableVectorizer to process text features
X_train_vectorized = X_train.skb.apply(skrub.TableVectorizer())

# Create and apply the multi-output classifier
model = MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=1000))
fitted_model = X_train_vectorized.skb.apply(model, y=y_train)

# Create learner
learner = fitted_model.skb.make_learner(fitted=True)

# Predict on test set using the learner
test_predictions = learner.predict({"train_data": test_df})

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'toxic': test_predictions[:, 0],
    'severe_toxic': test_predictions[:, 1],
    'obscene': test_predictions[:, 2],
    'threat': test_predictions[:, 3],
    'insult': test_predictions[:, 4],
    'identity_hate': test_predictions[:, 5]
})

# Save submission to final directory
submission.to_csv('./final/submission.csv', index=False)