# Flipkart-Sentiment-Analysis-Project
Developed an end-to-end Flipkart Sentiment Analysis system using NLP and Machine Learning, deployed as a Streamlit web app on AWS EC2 for real-time review classification.

🛒 Flipkart Product Review Sentiment Analysis
📌 Overview

This project is an end-to-end Sentiment Analysis system that classifies Flipkart product reviews as Positive or Negative using Natural Language Processing (NLP) and Machine Learning. The application is built with Streamlit for real-time predictions and uses TF-IDF + Logistic Regression for sentiment classification.

Users can enter any product review and instantly receive sentiment feedback through a web interface.

🎯 Objective

To analyze customer reviews and determine sentiment polarity (positive/negative) in order to understand customer satisfaction and identify potential pain points.

📂 Dataset

The dataset consists of Flipkart product reviews with fields such as:

Review Text

Rating

Reviewer Details (optional)

Sentiment labeling logic:

Rating ≥ 4 → Positive (1)

Rating < 4 → Negative (0)

The application automatically detects review and rating columns from the dataset.

🔄 Project Workflow

Load dataset using Pandas

Perform text preprocessing (lowercasing, removing special characters, stopwords removal)

Convert text into numerical features using TF-IDF Vectorization

Split data into training and testing sets

Train a Logistic Regression model

Evaluate performance using F1 Score

Save trained model and vectorizer using Joblib

Build a Streamlit web interface for real-time predictions

🧠 Model & Techniques

Text Cleaning with Regular Expressions and NLTK

Feature Extraction: TF-IDF

Machine Learning Model: Logistic Regression

Evaluation Metric: F1 Score

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

NLTK

Streamlit

Joblib

🚀 How to Run Locally
1️⃣ Clone the repository
git clone <your-repo-url>
cd <your-project-folder>

2️⃣ Create virtual environment
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Linux / Mac

source venv/bin/activate

3️⃣ Install dependencies
pip install pandas numpy scikit-learn nltk streamlit joblib

4️⃣ Run the application
streamlit run hello.py


Open in browser:

http://3.95.27.57:8501

🌐 Deployment

The application can be deployed on cloud platforms such as AWS EC2.
Steps include:

Launch Ubuntu EC2 instance

Install Python and dependencies

Upload project files

Open port 8501 in Security Group

Run Streamlit with:

streamlit run hello.py --server.port 8501 --server.address 0.0.0.0

✅ Features

Automatic dataset column detection

Real-time sentiment prediction

Clean web UI using Streamlit

F1-score based evaluation

Model persistence with Joblib

📈 Future Enhancements

Pain-point extraction from negative reviews

WordCloud visualization

BERT-based sentiment model

Database integration

Auto-start on server reboot

👤 Author

Developed by Siva Prasad
