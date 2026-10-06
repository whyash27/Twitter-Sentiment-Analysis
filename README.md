# Project: Twitter Sentiment Analysis using Multinomial Naive Bayes

## Overview

This project aims to perform sentiment analysis on Twitter data using the Multinomial Naive Bayes classifier. Sentiment analysis involves determining the sentiment (positive, negative, or neutral) associated with a piece of text, in this case, tweets from Twitter. Multinomial Naive Bayes classifier is used in this projects as it is a popular choice for text classification tasks due to its simplicity and effectiveness

## Installation

To run this project, you need to have Python installed on your system. Additionally, you'll need to install the following libraries:

- nltk: `pip install nltk`
- scikit-learn: `pip install scikit-learn`
- streamlit: `pip install streamlit`

## Usage

1. **Data Collection:** Obtain Twitter data by using a pre-existing dataset.

2. **Preprocessing:** Preprocess the text data by removing stopwords using the NLTK library. Count Vectorizer is used to convert text data into numerical feature vectors.

4. **Model Training:** Train the Multinomial Naive Bayes classifier using the preprocessed data.

5. **Prediction:** Use the trained model to predict the sentiment of new tweets.

6. **Application Deployment:** Use the provided `app.py` file to deploy the sentiment analysis application using Streamlit. This file contains the necessary code to create a user interface for interacting with the trained model.

## How to Run

1. Clone this repository to your local machine.
2. Ensure all required libraries are installed.
3. Run the provided Jupytrt nptebook` to preprocess data and train the model.
4. Run the Streamlit application using the command `streamlit run app.py`.

## Folder Structure

- `Twitter_data`: Twitter tweets data stored in csv format.
- `models`: Trained model in pkl format.
- `app.py`: Python scripts for the streamlit website for the model.
- `demo.mp4`: A demo video of how the model actually performs

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

## Credits

This project was created by Yash Pillewan.
