import ast
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# import custom modules
from data_pipeline import ETL_Pipeline
from metrics import Metrics

class RNN_Model():
    '''This class will implement an RNN model for sentiment analysis.'''
    def __init__(self):
        self.etl_pipeline = ETL_Pipeline()
        self.metrics = Metrics()
        self.tokenizer = Tokenizer()
        self.sentiment_dataset = None
        self.model = None
        self.data = None
    
    def load_data(self):
        # Load transformed data from pickle file
        if os.path.exists('data/transformed_data.pkl'):
            with open('data/transformed_data.pkl', 'rb') as f:
                self.data = pickle.load(f)
                print('Transformed data loaded successfully.')
        else:
            self.etl_pipeline.extract('./data/amazon_movie_reviews.csv')
            self.etl_pipeline.transform()
            self.data = self.etl_pipeline.data
            print('Data extracted and transformed successfully.')
    
    def build_model(self, vocab_size, embedding_dim, max_length):
        model = Sequential([Embedding(vocab_size, embedding_dim, input_length=max_length),
                            GRU(32),
                            Dense(1, activation='sigmoid')])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, epochs=5, batch_size=32):
        # Tokenize the text data
        self.tokenizer.fit_on_texts(X_train)

        vocab_size = len(self.tokenizer.word_index) + 1
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_train = pad_sequences(X_train, maxlen=100, padding='post')
        embedding_dim = 16

        self.model = self.build_model(vocab_size, embedding_dim, 100)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # Save the trained model in models directory
        self.model.save('models/rnn_model.h5')
        print('Model trained and saved successfully.')

        # Save tokenizer for inference
        with open('models/tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
            print('Tokenizer saved successfully.')
    
    def predict(self, input_text):
        # Load the trained model
        self.model = load_model('models/sentiment_analysis_rnn.h5')

        # Load the tokenizer
        with open('models/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)

        print('Model and tokenizer loaded successfully.')
        # Preprocess the input text
        input_text = pd.Series([input_text])
        _, tokens = self.etl_pipeline.preprocess(input_text)
        input_text = self.tokenizer.texts_to_sequences(input_text)
        input_text = pad_sequences(input_text, maxlen=100, padding='post')

        # Make prediction
        pred_prob = self.model.predict(input_text)[0][0]
        pred_label = 'positive' if pred_prob >= 0.5 else 'negative'

        return pred_label, pred_prob
    
    def get_model_stats(self):
        if self.model is not None:
            self.load_data()

            for fold in range(1, 6):
                print(f'Fold: {fold} processing...')

                # Split the data into train and test sets
                training_data = self.get_training_dataset(self.data, fold=fold)
                testing_data = self.get_testing_dataset(self.data, fold=fold)

                X_train = training_data['tokens'].apply(ast.literal_eval)
                y_train = training_data['sentiment']

                X_test = testing_data['tokens'].apply(ast.literal_eval)
                y_test = testing_data['sentiment']

                # Train the model
                self.train_model(X_train, y_train)
                
                # Evaluate the model
                self.metrics.evaluate(X_test, y_test, fold)
            self.metrics.save_metrics()
        else:
            print('Model not loaded. Please load the model first.')

    def get_training_dataset(self, data, fold):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for current_fold, (train_index, test_index) in enumerate(skf.split(data['tokens'], data['sentiment'])):
            if current_fold == fold:
                return data.iloc[train_index]

    def get_testing_dataset(self, data, fold):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for current_fold, (train_index, test_index) in enumerate(skf.split(data['tokens'], data['sentiment'])):
            if current_fold == fold:
                return data.iloc[test_index]
    
    def get_validation_dataset(self, training_data, val_size=0.2, random_state=42):
        train_data, val_data = train_test_split(training_data, test_size=val_size, random_state=random_state, stratify=training_data['sentiment'])
        return val_data, train_data
            