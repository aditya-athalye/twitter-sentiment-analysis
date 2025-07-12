# -*- coding: utf-8 -*-
"""Twitter Sentiment Analysis - Optimized Version"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
from tqdm import tqdm
import gensim
from gensim.models.doc2vec import TaggedDocument

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords')

# Read the data
print("Loading data...")
train = pd.read_csv('train_tweet.csv')
test = pd.read_csv('test_tweets.csv')

# Take a smaller sample for testing (20% of data)
train = train.sample(frac=0.2, random_state=42)
test = test.sample(frac=0.2, random_state=42)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Display first few rows
print("\nTrain data head:")
print(train.head())

print("\nTest data head:")
print(test.head())

# Check for null values
print("\nNull values in train:", train.isnull().any().any())
print("Null values in test:", test.isnull().any().any())

# Add length column
train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()

# Plot label distribution
plt.figure(figsize=(6, 4))
train['label'].value_counts().plot.bar(color='pink')
plt.title('Label Distribution')
plt.show()

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

print("Preprocessing text...")
# Preprocess tweets with progress bar
train['processed_tweet'] = [preprocess_text(tweet) for tweet in tqdm(train['tweet'], desc="Processing train tweets")]
test['processed_tweet'] = [preprocess_text(tweet) for tweet in tqdm(test['tweet'], desc="Processing test tweets")]

# Create bag of words
print("Creating bag of words...")
cv = CountVectorizer(max_features=1000)  # Reduced from 2500 to 1000
X = cv.fit_transform(train['processed_tweet']).toarray()
y = train['label']

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Train and evaluate models
def train_and_evaluate(model, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_valid)
    
    # Calculate metrics
    train_accuracy = model.score(X_train, y_train)
    valid_accuracy = model.score(X_valid, y_valid)
    f1 = f1_score(y_valid, y_pred)
    
    print(f"{model_name} Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {valid_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return model

# Train only two models for testing
print("\nTraining models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),  # Added n_jobs=-1 for parallel processing
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    train_and_evaluate(model, name)

# Generate word cloud for visualization
print("\nGenerating word cloud...")
wordcloud = WordCloud(width=800, height=500, random_state=0, max_font_size=110).generate(' '.join(train['processed_tweet']))
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Word Cloud of Processed Tweets')
plt.show()

print("\nScript completed successfully!")