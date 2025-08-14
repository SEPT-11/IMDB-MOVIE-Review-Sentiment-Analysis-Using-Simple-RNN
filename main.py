##Import all the required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

#Load the imdb dataset and word index
word_index = imdb.get_word_index()
reverse_word_index = { value : key for key,value in word_index.items()}

#Load the pretrained model
model = load_model('simple_rnn.h5')


#Step 2:Helper Function
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get( i -3,'?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review =  [word_index.get(word ,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


##Prediction Function
def prediction_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction


##Design Streamlit App
import streamlit as st