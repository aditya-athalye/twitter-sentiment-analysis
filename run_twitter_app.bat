@echo off
cd /d "D:\Aditya\Clg\Python project\Twitter-Sentiment-Analysis"
call venv\Scripts\activate.bat
python -m streamlit run twitter_app.py
pause