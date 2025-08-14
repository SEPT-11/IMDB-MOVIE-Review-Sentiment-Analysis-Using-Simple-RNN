## 🎬 IMDB Movie Review Sentiment Analysis – Simple RNN

This project uses a Simple Recurrent Neural Network (RNN) to classify IMDB movie reviews as positive or negative sentiments.
It is built with TensorFlow and trained on the prebuilt IMDB dataset from tensorflow.keras.datasets.

## 📌 Project Overview

The IMDB dataset contains 50,000 movie reviews labeled as positive or negative. The goal is to build a simple RNN-based sentiment classifier that can learn the sequence patterns of words in a review and predict the sentiment.

## 📂 Dataset

Source: Prebuilt tensorflow.keras.datasets.imdb

Training Samples: 25,000

Testing Samples: 25,000

Preprocessing:

Tokenized and integer-encoded reviews

Word indices limited to top N frequent words (num_words)

Sequence padding to a fixed length

🛠 Tech Stack

Python

TensorFlow / Keras

NumPy

streamlit for deploying

## 📜 Model Architecture

Embedding Layer – Converts integer sequences to dense vectors

SimpleRNN Layer – Learns sequential dependencies in reviews

Dense Layer – Sigmoid activation for binary classification

Example:

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_length))
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

⚙️ Training

Loss: binary_crossentropy

Optimizer: adam

Metrics: accuracy

Epochs: Configurable (default: 5–10)

Validation Split: 20%

📊 Results

Training Accuracy: ~91.3%

Accuracy may vary based on hyperparameters and system setup.
