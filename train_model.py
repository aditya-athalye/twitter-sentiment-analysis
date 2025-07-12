import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re

def clean_tweet(tweet):
    # Remove usernames (@username)
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    # Remove non-letter characters
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    # Remove extra whitespace
    tweet = ' '.join(tweet.split())
    return tweet.lower()

# Load the training data
print("Loading training data...")
train_data = pd.read_csv('train_tweet.csv')

# Clean the tweets
print("Cleaning tweets...")
train_data['cleaned_tweet'] = train_data['tweet'].apply(clean_tweet)

# Create and fit the vectorizer
print("Creating and fitting vectorizer...")
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(train_data['cleaned_tweet'])
y = train_data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Save the model and vectorizer
print("Saving model and vectorizer...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Done! Model and vectorizer have been saved.") 