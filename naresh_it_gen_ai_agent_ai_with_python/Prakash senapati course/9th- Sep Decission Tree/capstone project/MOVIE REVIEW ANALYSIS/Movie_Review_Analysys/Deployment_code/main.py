import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Function to preprocess text
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [w for w in words if w.lower() not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    preprocessed_text = ' '.join(stemmed_words)
    return preprocessed_text

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Load CountVectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit app
def main():
    st.title("Sentiment Analysis of Reviews")
    review_text = st.text_area("Enter your review here:")

    if st.button("Predict"):
        # Preprocess the input text
        preprocessed_text = preprocess_text(review_text)

        # Convert text to numeric format
        preprocessed_text_numeric = vectorizer.transform([preprocessed_text])

        # Predict sentiment
        prediction = model.predict(preprocessed_text_numeric)

        # Display the result
        if prediction[0] == 1:
            st.success("Positive review!")
        else:
            st.error("Negative review!")

if __name__ == "__main__":
    main()
