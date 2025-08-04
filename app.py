import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()


# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# preprocessing function


def text_preprocessing(text):
    # lower case
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # removing special characters/ emojis
    new_text = []

    for i in text:
        if i.isalnum():
            new_text.append(i)

    text = new_text[:]
    new_text.clear()

    # removing stopwords and punctuations
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            new_text.append(i)

    # stemming
    text = new_text[:]
    new_text.clear()

    for i in text:
        new_text.append(ps.stem(i))

    return " ".join(new_text)


# title and description
st.title("Email/SMS Spam Detector")
st.write("This app detects whether an Email/SMS is spam or not using a trained machine learning model.")

# input text
input_text = st.text_area("Enter the text here:")

# button to predict
if st.button("Predict"):
    # preprocessing the input text
    preprocessed_text = text_preprocessing(input_text)

    # vectorizing the preprocessed text
    vectorized_text = vectorizer.transform([preprocessed_text])

    # prediction
    prediction = model.predict(vectorized_text)[0]

    # display the prediction
    if prediction == 1:
        st.header("This message is SPAM!")
    else:
        st.header("This message is NOT SPAM.")
