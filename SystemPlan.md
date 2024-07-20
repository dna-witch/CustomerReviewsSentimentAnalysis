# Sentiment Analysis Project

## Purpose

This project provides code for a microservice that uses a Recurrent Neural Network model to predict the sentiment (positive or negative) of an Amazon Movie product review. 
The sentiment of the review is determined by the user-awarded star-rating. Reviews with a rating of 3 to 5 are considered positive, while reviews with a rating of 1 or 2 are considered negative.

## Data
The Amazon Movie Reviews dataset used for this project contains 1 million samples of movie ratings, their associated plain text reviews, and features describing the products.

## Methods

### Exploratory Data Analysis
The exploratory data analysis revealed several trends in the data, including an imbalanced distribution of positive and negative ratings, skewed towards 5-star ratings. There were also many words that were commonly found throughout all rating brackets, and some words that seemed more exclusive to one or two rating brackets. There was also a disproportionate amount of unverified purchases compared to verified purchases. To counteract potential biases, the training data was stratified according to rating brackets. To improve performance, further stratification on verification status could be implemented, along with n-gram generation during the text preprocessing pipeline.

### Text Preprocessing
1. Data Cleaning: Remove instances with missing values and duplicates
2. Feature Extraction: Extract the sentiment labels of the review based on the rating bracket
3. Text Transformation: \
>* Remove HTML tags
>* Lowercase the text
>* Expand contractions
>* Remove special characters and punctuation
>* Remove stopwords
>* Lemmatize the words in the text
>* Tokenize the words in the text
>* Create an indexed vocabulary
>* Use Word2Vec to convert text into numerical vectors to be used as input for RNN model

### Modeling
The model chosen was a Recurrent Neural Network (RNN) built using Tensorflow and Keras. The model was trained on the dataset using the k-fold cross-validation technique to improve the performance.

### Model Evaluation 
The model's performance was evaluated on metrics such as accuracy, precision, recall, and F1-score.

## Endpoints

This container runs a script that listens to GET requests. \
The main script listens to port `8786`.

This microservice returns the following endpoints: \
`/stats`: returns the performance metrics of the RNN model. [GET method] \
http://localhost:8786/stats 

`/predict`: takes a text input with key "input_review" and returns the sentiment prediction as positive or negative, along with a probability. [GET method] \
http://localhost:8786/predict

## Usage Instructions

The Docker image for this project is hosted [here](https://hub.docker.com/repository/docker/shakuntalam/705.603spring24/general) under the tag *sentiment_analysis*.

Clone the Docker image \
`docker pull shakuntalam/705.603spring24:sentiment_analysis`

Find the IMAGE ID using \
`docker image ls`

Run the image using \
`docker run <container ID or container name>`

**Must install and use Postman Client to provide inference parameters and files.**