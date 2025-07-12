# Twitter Sentiment Analysis

A machine learning-powered web application that analyzes the sentiment of tweets using natural language processing techniques.

## 🚀 Live Demo

[View the live application on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/twitter-sentiment-analysis)

## 📊 Features

- **Sentiment Analysis**: Classifies tweets as positive or negative using a trained machine learning model
- **Interactive Interface**: User-friendly Streamlit web interface
- **Real-time Visualization**: Dynamic charts and metrics showing sentiment distribution
- **Sample Data**: Built-in sample tweets for testing and demonstration
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn, NLTK
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Model Persistence**: Joblib

## 📁 Project Structure

```
Twitter-Sentiment-Analysis/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── sentiment_model.pkl    # Trained sentiment model
├── vectorizer.pkl         # Text vectorizer
├── requirements.txt       # Python dependencies
├── train_tweet.csv        # Training dataset
├── test_tweets.csv        # Test dataset
└── README.md             # Project documentation
```

## 🎯 How It Works

1. **Text Preprocessing**: Tweets are cleaned by removing usernames, URLs, and special characters
2. **Feature Extraction**: Text is converted to numerical features using CountVectorizer
3. **Sentiment Classification**: A Logistic Regression model predicts sentiment (0=Negative, 1=Positive)
4. **Visualization**: Results are displayed with interactive charts and metrics

## 🚀 Deployment on Hugging Face Spaces

This project is deployed on Hugging Face Spaces, making it accessible to anyone with an internet connection.

### Key Files for Deployment:
- `app.py`: Main application entry point
- `requirements.txt`: Python dependencies
- `sentiment_model.pkl`: Pre-trained model
- `vectorizer.pkl`: Text vectorizer

## 📈 Model Performance

The sentiment analysis model was trained on a dataset of Twitter posts and achieves:
- Training Accuracy: ~85%
- Testing Accuracy: ~82%

## 🎨 Usage

1. Enter a topic to analyze
2. Add sample tweets (one per line)
3. Click "Analyze Tweets" to see results
4. View sentiment distribution and individual tweet classifications

## 🔧 Local Development

To run this project locally:

```bash
# Clone the repository
git clone <repository-url>
cd Twitter-Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or support, please open an issue on the project repository.

---

**Built with ❤️ using Streamlit and deployed on Hugging Face Spaces** 