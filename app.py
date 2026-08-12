import streamlit as st
import joblib

st.set_page_config(
    page_title="StockTwits Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_files():
    model = joblib.load("logistic_regression_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

try:
    model, tfidf = load_files()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

st.title("💬 StockTwits Sentiment Analysis")
st.write("Enter a StockTwits message below to predict its sentiment.")

text = st.text_area(
    "Enter your text:",
    placeholder="Example: This stock is going to rise strongly today!"
)

if st.button("Predict Sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            text_tfidf = tfidf.transform([text])
            prediction = model.predict(text_tfidf)[0]

            st.subheader("Prediction")

            if prediction == 1:
                st.success("Sentiment: Bullish 📈")
            elif prediction == 0:
                st.error("Sentiment: Bearish 📉")
            else:
                st.info(f"Sentiment: {prediction}")

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.write("Loaded model type:", type(model).__name__)
            st.write("Loaded vectorizer type:", type(tfidf).__name__)
