# 🚀 Hugging Face Spaces Deployment Guide

This guide will walk you through deploying your Twitter Sentiment Analysis project to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Git**: Make sure you have Git installed on your system
3. **Project Files**: Ensure all required files are in your project directory

## 🎯 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Create a new repository on GitHub** (if you haven't already):
   - Go to GitHub and create a new repository
   - Name it something like `twitter-sentiment-analysis`
   - Make it public

2. **Push your project to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Twitter Sentiment Analysis"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/twitter-sentiment-analysis.git
   git push -u origin main
   ```

### Step 2: Create Hugging Face Space

1. **Go to Hugging Face Spaces**:
   - Visit [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"

2. **Configure your Space**:
   - **Owner**: Select your username
   - **Space name**: `twitter-sentiment-analysis`
   - **License**: Choose appropriate license (e.g., MIT)
   - **SDK**: Select **Streamlit**
   - **Space hardware**: Choose **CPU** (free tier)
   - **Visibility**: Choose **Public**

3. **Click "Create Space"**

### Step 3: Connect Your Repository

1. **In your new Space, click "Files" tab**
2. **Click "Add file" → "Upload files"**
3. **Upload these essential files**:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `sentiment_model.pkl` (trained model)
   - `vectorizer.pkl` (text vectorizer)
   - `README.md` (documentation)

### Step 4: Configure the Space

1. **Create a `.gitattributes` file** (optional):
   ```
   *.pkl filter=lfs diff=lfs merge=lfs -text
   ```

2. **Update the Space description**:
   - Go to Settings → Space metadata
   - Add a compelling description
   - Add relevant tags: `sentiment-analysis`, `nlp`, `machine-learning`, `streamlit`

### Step 5: Test Your Deployment

1. **Wait for the build to complete** (usually 2-5 minutes)
2. **Check the logs** if there are any errors
3. **Test the application** by visiting your Space URL

## 🔧 Troubleshooting

### Common Issues:

1. **Build fails**:
   - Check that all dependencies are in `requirements.txt`
   - Ensure `app.py` is the main entry point
   - Verify file names match exactly

2. **Model loading errors**:
   - Make sure `sentiment_model.pkl` and `vectorizer.pkl` are uploaded
   - Check file paths in your code

3. **Import errors**:
   - Verify all packages are listed in `requirements.txt`
   - Check for version conflicts

### Debug Steps:

1. **Check build logs** in the Space's "Logs" tab
2. **Test locally first**:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## 📊 Monitoring Your Space

1. **View analytics** in the Space's "Analytics" tab
2. **Monitor usage** and performance
3. **Check for errors** in the logs

## 🔗 Sharing Your Space

Once deployed, you can share your Space using:
- **Direct URL**: `https://huggingface.co/spaces/YOUR_USERNAME/twitter-sentiment-analysis`
- **Embed in websites**: Use the provided embed code
- **Social media**: Share the URL on LinkedIn, Twitter, etc.

## 🎉 Success!

Your Twitter Sentiment Analysis app is now live and accessible to anyone with an internet connection!

## 📝 Next Steps

1. **Add to your portfolio**: Include the link in your resume/LinkedIn
2. **Share on social media**: Promote your project
3. **Gather feedback**: Ask for user feedback to improve
4. **Iterate**: Make improvements based on usage and feedback

---

**Need help?** Check the [Hugging Face Spaces documentation](https://huggingface.co/docs/hub/spaces) or open an issue in your repository. 