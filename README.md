# PhishScan – AI Powered Phishing URL Detector
# Project Overview
PhishScan is a web-based application designed to detect and analyze phishing URLs using machine learning and heuristic techniques. With the rapid increase in phishing attacks and online scams, this system helps users verify whether a URL is safe or potentially malicious. The application combines machine learning models, URL feature extraction, and rule-based analysis to provide reliable predictions.

# Key Features
•	Real-time URL analysis
•	Machine learning based phishing detection
•	Heuristic fallback detection when the trained model is unavailable
•	Feature extraction from URLs
•	Interactive web interface built with Streamlit
•	Probability based phishing prediction

# How the System Works
1. Machine Learning Prediction
•	Extracts important structural features from the input URL
•	Converts URLs into vectorized format
•	Uses a trained XGBoost model to classify URLs
•	Returns a phishing probability score to the user
2. Heuristic Detection
•	Checks for suspicious tokens in URLs
•	Detects URLs containing IP addresses instead of domain names
•	Identifies excessive digits or hyphens
•	Detects unusually long URLs
•	Checks for absence of HTTPS

# Project Structure
Phishing Project
│
├── app.py                 // Streamlit web application
├── train_model.py         // Script to train phishing detection model
├── utils.py               // URL feature extraction functions
│
├── data/
│   └── phishing_urls.csv  // Dataset used for training
│
├── models/
│   ├── xgb_model.joblib   // Trained ML model
│   ├── vectorizer.joblib  // URL vectorizer
│   └── meta.json          // Model metadata
│
├── requirements.txt       // Project dependencies
├── Testing.xlsx           // Testing dataset
└── README.md

# Technologies Used
•	Python
•	Streamlit
•	XGBoost
•	Scikit-learn
•	Pandas
•	Joblib
•	SciPy


