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

cooldown_remaining = get_cooldown_remaining()

# Topic input
topic = st.sidebar.text_input("Enter topic to analyze", "cricket", disabled=cooldown_remaining > 0)

# Sample tweets input
st.sidebar.markdown("### Sample Tweets")
st.sidebar.markdown("Enter some sample tweets to analyze (one per line):")

sample_tweets_text = st.sidebar.text_area(
    "Sample tweets",
    value="I love cricket! It's the best sport ever.\nCricket is so boring, I can't watch it.\nThe match was amazing today!\nWhat a terrible game that was.",
    height=150,
    disabled=cooldown_remaining > 0
)

# Analyze button
disabled_btn = cooldown_remaining > 0
analyze_btn = st.sidebar.button("Analyze Tweets", disabled=disabled_btn)

if cooldown_remaining > 0:
    mins, secs = divmod(cooldown_remaining, 60)
    st.sidebar.warning(f"Please wait {mins:02d}:{secs:02d} before trying again.")

if analyze_btn and not disabled_btn:
    if not topic:
        st.error("Please enter a topic!")
    elif not sample_tweets_text.strip():
        st.error("Please enter some sample tweets!")
    else:
        with st.spinner("Analyzing tweets..."):
            # Split tweets by newlines and filter empty lines
            tweets = [tweet.strip() for tweet in sample_tweets_text.split('\n') if tweet.strip()]
            
            if tweets:
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
                    st.subheader(f"Results for '{topic}'")
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sentiment_df.plot(kind='bar', x='Sentiment', y='Count', ax=ax, color=['green', 'red'])
                    plt.title(f"Sentiment Analysis for '{topic}'")
                    plt.xlabel("Sentiment")
                    plt.ylabel("Number of Tweets")
                    plt.xticks(rotation=0)
                    st.pyplot(fig)
                    
                    # Display counts
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Positive Tweets", positive_count)
                    with col2:
                        st.metric("Negative Tweets", negative_count)
                    
                    # Calculate percentage
                    total = len(tweets)
                    positive_percent = (positive_count / total) * 100 if total > 0 else 0
                    negative_percent = (negative_count / total) * 100 if total > 0 else 0
                    
                    st.write(f"**Overall Sentiment:** {positive_percent:.1f}% Positive, {negative_percent:.1f}% Negative")
                    
                    # Display analyzed tweets
                    st.subheader("Analyzed Tweets")
                    for i, tweet in enumerate(tweets):
                        sentiment = "Positive" if predictions[i] == 1 else "Negative"
                        color = "green" if predictions[i] == 1 else "red"
                        st.markdown(f"**{i+1}.** {tweet} *(Sentiment: :{color}[{sentiment}])*")
                    
                    # Set cooldown
                    st.session_state['cooldown_until'] = time.time() + COOLDOWN_SECONDS

# Add instructions in the sidebar
st.sidebar.markdown("""
### Instructions
1. Enter a topic to analyze
2. Add some sample tweets (one per line)
3. Click 'Analyze Tweets' to see results

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