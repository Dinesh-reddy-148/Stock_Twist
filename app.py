import streamlit as st
import joblib

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_files():
    loaded_model = joblib.load("logistic_regression_model.pkl")
    loaded_tfidf = joblib.load("tfidf_vectorizer.pkl")
    return loaded_model, loaded_tfidf

try:
    model, tfidf = load_files()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

st.title("💬 Sentiment Analysis")
st.write("Enter text below to predict its sentiment.")

text = st.text_area(
    "Enter your text:",
    placeholder="Example: I really enjoyed this product!"
)

if st.button("Predict Sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            text_tfidf = tfidf.transform([text])
            prediction = model.predict(text_tfidf)[0]

            st.subheader("Prediction")
            st.success(f"Sentiment: {prediction}")

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.write("Loaded model type:", type(model).__name__)
            st.write("Loaded vectorizer type:", type(tfidf).__name__)
