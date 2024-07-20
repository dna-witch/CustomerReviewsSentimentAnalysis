# Sentiment Analysis of Amazon Movie Reviews

This project provides code for a microservice that uses a Recurrent Neural Network model to predict the sentiment (positive or negative) of an Amazon Movie product review. 
The sentiment of the review is determined by the user-awarded star-rating. Reviews with a rating of 3 to 5 are considered positive, while reviews with a rating of 1 or 2 are considered negative.

### Endpoints

This container runs a script that listens to GET requests. \
The main script listens to port `8786`.

This microservice returns the following endpoints: \
`/stats`: returns the performance metrics of the RNN model. [GET method] \
http://localhost:8786/stats 

`/predict`: takes a text input with key "input_review" and returns the sentiment prediction as positive or negative, along with a probability. [GET method] \
http://localhost:8786/predict

### Usage Instructions

The Docker image for this project is hosted [here](https://hub.docker.com/repository/docker/shakuntalam/705.603spring24/general) under the tag *sentiment_analysis*.

Clone the Docker image \
`docker pull shakuntalam/705.603spring24:sentiment_analysis`

Find the IMAGE ID using \
`docker image ls`

Run the image using \
`docker run <container ID or container name>`

**Must install and use Postman Client to provide inference parameters and image files.**