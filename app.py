import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

# Add a title and description
st.title("🐦 Twitter Sentiment Analysis")
st.markdown("""
This app analyzes the sentiment of tweets about any topic. 
Enter a topic and some sample tweets to get started!
""")

# Helper to get cooldown remaining
COOLDOWN_SECONDS = 60  # 1 minute for demo

def get_cooldown_remaining():
    if 'cooldown_until' in st.session_state:
        now = time.time()
        remaining = st.session_state['cooldown_until'] - now
        return max(0, int(remaining))
    return 0

# Function to clean tweets
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

# Function to analyze sentiment
def analyze_sentiment(tweets):
    try:
        # Load the trained model and vectorizer
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        # Clean tweets
        cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]
        
        # Transform tweets using vectorizer
        X = vectorizer.transform(cleaned_tweets)
        
        # Predict sentiment
        predictions = model.predict(X)
        
        return predictions
        
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return None

# Create the main interface
st.sidebar.header("Input Parameters")
topic = st.sidebar.text_input("Enter tweet to analyze", "I love cricket! It's the best sport ever.")

# Analyze button
analyze_btn = st.sidebar.button("Analyze Tweet")

if analyze_btn:
    if not topic.strip():
        st.error("Please enter a tweet!")
    else:
        with st.spinner("Analyzing tweet..."):
            tweets = [topic.strip()]
            predictions = analyze_sentiment(tweets)
            if predictions is not None:
                sentiment = "Positive" if predictions[0] == 1 else "Negative"
                color = "green" if predictions[0] == 1 else "red"
                st.subheader("Result")
                st.markdown(f"**Tweet:** {topic.strip()}")
                st.markdown(f"**Sentiment:** :{color}[{sentiment}]")

# Add instructions in the sidebar
st.sidebar.markdown("""
### Instructions
1. Enter a tweet to analyze
2. Click 'Analyze Tweet' to see the result

### About the Model
This app uses a machine learning model trained on Twitter data to classify sentiment as positive or negative.
""")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit • Deployed on Hugging Face Spaces</p>
</div>
""", unsafe_allow_html=True) 