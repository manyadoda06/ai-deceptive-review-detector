# AI-Based Deceptive Review Detection System

## Overview

This project is a Machine Learning and Natural Language Processing (NLP) application that detects whether a review is genuine or deceptive based on its writing patterns.

The system uses TF-IDF vectorization and a Logistic Regression model to classify user-submitted reviews. An interactive Streamlit interface allows users to enter text and receive predictions with confidence scores.

## Features

* Detects deceptive (fake) reviews
* Identifies genuine reviews
* TF-IDF text vectorization
* Logistic Regression classification
* Confidence score prediction
* Interactive Streamlit web application

## Technologies Used

* Python
* Scikit-learn
* Streamlit
* Pandas
* TF-IDF Vectorization
* Logistic Regression

## Project Workflow

1. Load and preprocess review data
2. Convert text into numerical features using TF-IDF
3. Train a Logistic Regression classifier
4. Save the trained model and vectorizer
5. Deploy using Streamlit
6. Predict whether user-entered reviews are deceptive or genuine

## Applications

* Fake review detection
* E-commerce review analysis
* Customer feedback monitoring
* Online reputation management
* Sentiment and text analytics

## Files Included

* app.py
* train_text.py
* text_model.pkl
* tfidf.pkl

## Author

Manya Doda
