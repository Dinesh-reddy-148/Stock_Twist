import streamlit as st
import joblib

# Load trained model
model = joblib.load("logistic_regression_model.pkl")

# Load TF-IDF vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# Application title
st.title("💬 Sentiment Analysis")
st.write("Enter a text below to predict its sentiment.")


# Text input
text = st.text_area(
    "Enter your text:",
    placeholder="Example: I really enjoyed this product!"
)


# Prediction button
if st.button("Predict Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        # Convert input text into TF-IDF features
        text_tfidf = tfidf.transform([text])

        # Make prediction
        prediction = model.predict(text_tfidf)[0]

        # Display result
        st.subheader("Prediction")
        st.success(f"Sentiment: {prediction}")
