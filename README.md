# Flipkart Sentiment Analysis Project

Developed an end-to-end Flipkart Sentiment Analysis system using Natural Language Processing and Machine Learning, deployed as a Streamlit web application on AWS EC2 for real-time review classification.

## Overview

This project is an end-to-end sentiment analysis application that classifies Flipkart product reviews as Positive or Negative using NLP and machine learning techniques. The system is built with Streamlit for real-time predictions and uses TF-IDF vectorization combined with Logistic Regression for sentiment classification.

Users can enter any product review and instantly receive sentiment feedback through a web interface.

## Objective

To analyze customer reviews and determine sentiment polarity (positive or negative) in order to understand customer satisfaction and identify potential pain points.

## Dataset

The dataset consists of Flipkart product reviews with the following fields:

- Review Text  
- Rating  
- Reviewer Details (optional)

Sentiment labeling logic:

- Rating ≥ 4 → Positive (1)  
- Rating < 4 → Negative (0)  

The application automatically detects review and rating columns from the dataset.

## Project Workflow

- Load dataset using Pandas  
- Perform text preprocessing (lowercasing, removing special characters, stopword removal)  
- Convert text into numerical features using TF-IDF Vectorization  
- Split data into training and testing sets  
- Train a Logistic Regression model  
- Evaluate performance using F1 Score  
- Save trained model and vectorizer using Joblib  
- Build a Streamlit web interface for real-time predictions  

## Model and Techniques

- Text cleaning using Regular Expressions and NLTK  
- Feature extraction using TF-IDF  
- Machine Learning model: Logistic Regression  
- Evaluation metric: F1 Score  

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- NLTK  
- Streamlit  
- Joblib  

## How to Run Locally

Clone the repository:

git clone <your-repo-url>  
cd <your-project-folder>

Create virtual environment:

python -m venv venv

Activate environment:

Windows:  
venv\Scripts\activate  

Linux / Mac:  
source venv/bin/activate  

Install dependencies:

pip install pandas numpy scikit-learn nltk streamlit joblib

Run the application:

streamlit run hello.py

Open in browser:

http://3.95.27.57:8501

## Deployment

The application can be deployed on AWS EC2.

Steps include:

- Launch Ubuntu EC2 instance  
- Install Python and required dependencies  
- Upload project files  
- Open port 8501 in the Security Group  
- Run Streamlit using:

streamlit run hello.py --server.port 8501 --server.address 0.0.0.0

## Features

- Automatic dataset column detection  
- Real-time sentiment prediction  
- Clean Streamlit-based web interface  
- F1-score based evaluation  
- Model persistence using Joblib  

## Future Enhancements

- Pain-point extraction from negative reviews  
- WordCloud visualization  
- BERT-based sentiment analysis  
- Database integration  
- Auto-start on server reboot  

## Author

Developed by Siva Prasad
