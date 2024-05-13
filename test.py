from flask import Flask, render_template, request
import sqlite3
import datetime
import joblib
import pickle
from textblob import TextBlob
from spellchecker import SpellChecker
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# Preprocess text function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Removing PII (Personal Identifiable Information)
    text = re.sub(r'\b(?:\d{10,}|(?:\w+\.?)*\w+@(?:\w+\.)+[a-zA-Z]{2,}\b)', 'PII_REMOVED', text)

    # Simple Spell Checking and Correction
    corrected_text = []
    words = word_tokenize(text)
    for word in words:
        if not word.isalpha():
            continue  # Skip non-alphabetic words
        corrected_text.append(word)  # Just keep the word as is
    text = ' '.join(corrected_text)

    # Removing Special Characters and Symbols
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Joining tokens back into text
    processed_text = ' '.join(tokens)

    return processed_text

# Load the random forest classifier
rf_classifier = joblib.load('rf_model.pkl')

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Sample text
sample_text = "Free money!!! Call now to claim your prize."

# Preprocess the text
processed_text = preprocess_text(sample_text)

# Transform the text using the TF-IDF vectorizer
vectorized_text = tfidf_vectorizer.transform([processed_text])

# Making a prediction
prediction = rf_classifier.predict(vectorized_text)
# Output the prediction
print("The text is classified as:", "Spam" if prediction[0] == 1 else "Not Spam")
