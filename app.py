import pickle
import streamlit as st
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Assuming you have these variables
model_path = "model.pkl"  # Your saved model path
vocabulary_path = "vocabulary.pkl"  # Path to vocabulary file (optional)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in stop_words]
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
    return " ".join(stemmed_tokens)

def main():
    st.title("Sentiment Analysis App")

    st.header("Enter a tweet to analyze its sentiment:")

    # Input field and analysis
    user_text = st.text_input("")

    if user_text:
        try:
            # Load the model
            with open(model_path, "rb") as f:
                clf = pickle.load(f)

            # Load Vocab
            if hasattr(clf, 'vectorizer_'):
                vectorizer = clf.vectorizer_
            elif vocabulary_path:
                with open(vocabulary_path, "rb") as f:
                    cv = CountVectorizer()  # Create an empty CountVectorizer
                    cv.vocabulary_ = pickle.load(f)  # Load vocabulary
                    vectorizer = cv
            else:
                raise ValueError("Unable to access vectorizer information.")

            # Preprocess text
            preprocessed_text = preprocess_text(user_text)

            # Convert text to features
            X = vectorizer.transform([preprocessed_text])

            # Make prediction and handle potential errors
            prediction = clf.predict(X)[0]
            if prediction == 0:
                st.success("The tweet is neutral.")
            elif prediction == 1:
                st.success("The tweet is positive!")
            else:
                st.error("The tweet is negative.")

        except Exception as e:
            st.error("An error occurred:{}".format(e))

    # Add an "About" section
    about_expander = st.expander("About")
    with about_expander:
        st.subheader("Model Description")
        st.markdown("This app uses a *Multinomial Naive Bayes* machine learning model trained on a dataset of labelled tweets.")
        st.markdown("The model was trained to classify tweets as *positive, neutral, or negative* based on their sentiment.")
        st.markdown("Training data consisted of a collection of tweets labelled with their corresponding sentiment, covering various topics and contexts.")
        st.markdown("While the model achieves high accuracy on the training data, its performance may vary on tweets with unconventional language usage or ambiguous sentiment.")

        st.subheader("Disclaimer")
        st.markdown("This app is intended for demonstration purposes only and may not accurately reflect the sentiment of all tweets.")
        st.markdown("The predictions made by the model are based on statistical patterns learned from the training data and should be interpreted with caution in real-world scenarios.")

        st.subheader("How to Use")
        st.markdown("Enter a tweet in the provided text box and click the 'Analyze' button to get the sentiment prediction.")
        st.markdown("The app will classify the tweet as positive, neutral, or negative based on its sentiment.")


if __name__ == "__main__":
    main()
