# Twitter Sentiment Analysis App
# This app fetches tweets about a topic and analyzes their sentiment

# Import required libraries
import streamlit as st  # For creating the web interface
import tweepy  # For accessing Twitter API
import pandas as pd  # For data manipulation
import re  # For text cleaning
import joblib  # For loading the trained model
import matplotlib.pyplot as plt  # For creating visualizations
from datetime import datetime  # For date handling
import os  # For environment variables
from dotenv import load_dotenv  # For loading environment variables
import time

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

# Add a title and description
st.title("🐦 Twitter Sentiment Analysis")
st.markdown("""
This app analyzes the sentiment of tweets about any topic on a specific date.
Enter a topic and date to get started!
""")

# Helper to get cooldown remaining
COOLDOWN_SECONDS = 900  # 15 minutes

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

# Function to fetch tweets
def fetch_tweets(topic, date):
    try:
        # Get Twitter API credentials from environment variables
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        if not bearer_token:
            st.error("Twitter Bearer Token not found! Please check your .env file.")
            return None
        
        # Initialize Twitter client
        client = tweepy.Client(bearer_token=bearer_token)
        
        # Convert date to Twitter's format
        date_str = date.strftime('%Y-%m-%d')
        
        # Search tweets
        tweets = client.search_recent_tweets(
            query=topic,
            start_time=f"{date_str}T00:00:00Z",
            end_time=f"{date_str}T23:59:59Z",
            max_results=100
        )
        
        if not tweets.data:
            st.warning("No tweets found for the given topic and date.")
            return None
            
        return tweets.data
        
    except Exception as e:
        if '429' in str(e):
            return '429'
        st.error(f"Error fetching tweets: {str(e)}")
        return None

# Function to analyze sentiment
def analyze_sentiment(tweets):
    try:
        # Load the trained model and vectorizer
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        # Clean tweets
        cleaned_tweets = [clean_tweet(tweet.text) for tweet in tweets]
        
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

cooldown_remaining = get_cooldown_remaining()

# Topic input
topic = st.sidebar.text_input("Enter topic to search", "cricket", disabled=cooldown_remaining > 0)

# Date input
date = st.sidebar.date_input("Select date", datetime.now(), disabled=cooldown_remaining > 0)

# Search button
disabled_btn = cooldown_remaining > 0
analyze_btn = st.sidebar.button("Analyze Tweets", disabled=disabled_btn)

if cooldown_remaining > 0:
    mins, secs = divmod(cooldown_remaining, 60)
    st.sidebar.warning(f"Too many requests to Twitter API. Please wait {mins:02d}:{secs:02d} before trying again.")
    st.sidebar.info("Note: Twitter's free API only allows a limited number of requests per 15 minutes. This is a restriction from Twitter, not this app.")

if analyze_btn and not disabled_btn:
    if not topic:
        st.error("Please enter a topic!")
    else:
        with st.spinner("Fetching and analyzing tweets..."):
            # Fetch tweets
            tweets = fetch_tweets(topic, date)
            
            if tweets == '429':
                st.session_state['cooldown_until'] = time.time() + COOLDOWN_SECONDS
                st.sidebar.warning("You have hit the Twitter API rate limit. Please wait before trying again.")
                st.sidebar.info("Note: Twitter's free API only allows a limited number of requests per 15 minutes. This is a restriction from Twitter, not this app.")
            elif tweets:
                # Analyze sentiment
                predictions = analyze_sentiment(tweets)
                
                if predictions is not None:
                    # Count positive and negative tweets
                    positive_count = sum(predictions == 1)
                    negative_count = sum(predictions == 0)
                    
                    # Create a DataFrame for visualization
                    sentiment_df = pd.DataFrame({
                        'Sentiment': ['Positive', 'Negative'],
                        'Count': [positive_count, negative_count]
                    })
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sentiment_df.plot(kind='bar', x='Sentiment', y='Count', ax=ax)
                    plt.title(f"Sentiment Analysis for '{topic}' on {date.strftime('%Y-%m-%d')}")
                    plt.xlabel("Sentiment")
                    plt.ylabel("Number of Tweets")
                    st.pyplot(fig)
                    
                    # Display counts
                    st.write(f"Positive Tweets: {positive_count}")
                    st.write(f"Negative Tweets: {negative_count}")
                    
                    # Display some example tweets
                    st.subheader("Example Tweets")
                    for i, tweet in enumerate(tweets[:5]):
                        sentiment = "Positive" if predictions[i] == 1 else "Negative"
                        st.write(f"{i+1}. {tweet.text} (Sentiment: {sentiment})")

# Add instructions in the sidebar
st.sidebar.markdown("""
### Instructions
1. Enter a topic to search for
2. Select a date
3. Click 'Analyze Tweets' to see results

### Note
The app uses a secure configuration file (.env) to store the Twitter API credentials.
""") 