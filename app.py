import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StockTwits Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)


# =========================================================
# LOAD MODEL, VECTORIZER AND DASHBOARD DATA
# =========================================================

@st.cache_resource
def load_model_files():
    model = joblib.load("logistic_regression_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf


@st.cache_data
def load_dashboard_data():
    return pd.read_pickle("dashboard_data.pkl")


try:
    model, tfidf = load_model_files()
    dashboard_data = load_dashboard_data()

except Exception as error:
    st.error(f"Could not load application files: {error}")
    st.stop()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("💬 StockTwits Analytics")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "📊 Dashboard Overview",
        "📈 Market Activity",
        "💬 Sentiment & Text",
        "🤖 Model Performance",
        "🔮 Live Prediction"
    ]
)


# =========================================================
# DASHBOARD OVERVIEW
# =========================================================

if page == "📊 Dashboard Overview":

    st.title("📊 StockTwits Sentiment Analytics Dashboard")

    st.markdown(
        """
        Explore StockTwits investor discussions, sentiment patterns,
        market activity and machine-learning model performance.
        """
    )

    # Load data
    sentiment_df = dashboard_data["sentiment_distribution"]

    monthly_posts = dashboard_data["monthly_posts"]

    # Calculate KPIs
    total_messages = sentiment_df["Messages"].sum()

    bullish_messages = sentiment_df.loc[
        sentiment_df["Sentiment"] == "Bullish",
        "Messages"
    ].sum()

    bearish_messages = sentiment_df.loc[
        sentiment_df["Sentiment"] == "Bearish",
        "Messages"
    ].sum()

    bullish_percentage = (
        bullish_messages / total_messages * 100
    )

    bearish_percentage = (
        bearish_messages / total_messages * 100
    )

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Analysis Messages",
        f"{total_messages:,}"
    )

    col2.metric(
        "Bullish Messages",
        f"{bullish_percentage:.1f}%"
    )

    col3.metric(
        "Bearish Messages",
        f"{bearish_percentage:.1f}%"
    )

    col4.metric(
        "Time Period",
        f"{len(monthly_posts)} months"
    )

    st.markdown("---")

    # -----------------------------------------------------
    # SENTIMENT DISTRIBUTION
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Messages",
            hole=0.45,
            title="Overall Sentiment Distribution"
        )

        fig.update_layout(
            legend_title="Sentiment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # MONTHLY ACTIVITY
    # -----------------------------------------------------

    with col2:

        fig = px.line(
            monthly_posts,
            x="Month",
            y="Messages",
            markers=True,
            title="Monthly StockTwits Activity"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Messages",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# MARKET ACTIVITY
# =========================================================

elif page == "📈 Market Activity":

    st.title("📈 Market Activity Analysis")

    st.markdown(
        "Interactive analysis of StockTwits posting activity over time."
    )

    monthly_posts = dashboard_data["monthly_posts"]
    monthly_sentiment = dashboard_data["monthly_sentiment"]
    hourly_activity = dashboard_data["hourly_activity"]

    # -----------------------------------------------------
    # MONTHLY ACTIVITY
    # -----------------------------------------------------

    st.subheader("Monthly StockTwits Activity")

    fig = px.line(
        monthly_posts,
        x="Month",
        y="Messages",
        markers=True,
        title="Monthly Number of StockTwits Messages"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Messages",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # MONTHLY BULLISH VS BEARISH
    # -----------------------------------------------------

    st.subheader("Bullish vs Bearish Activity")

    fig = px.line(
        monthly_sentiment,
        x="Month",
        y="Messages",
        color="Sentiment",
        markers=True,
        title="Monthly Bullish vs Bearish Discussions"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Messages",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # HOURLY ACTIVITY
    # -----------------------------------------------------

    st.subheader("Hourly Investor Activity")

    fig = px.bar(
        hourly_activity,
        x="Hour",
        y="Messages",
        title="StockTwits Messages by Hour"
    )

    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Messages"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# SENTIMENT AND TEXT ANALYSIS
# =========================================================

elif page == "💬 Sentiment & Text":

    st.title("💬 Sentiment & Text Analysis")

    # -----------------------------------------------------
    # SENTIMENT FILTER
    # -----------------------------------------------------

    selected_sentiments = st.multiselect(
        "Select sentiment:",
        ["Bullish", "Bearish"],
        default=["Bullish", "Bearish"]
    )

    sentiment_df = dashboard_data["sentiment_distribution"]

    filtered_sentiment = sentiment_df[
        sentiment_df["Sentiment"].isin(selected_sentiments)
    ]

    # -----------------------------------------------------
    # SENTIMENT CHART
    # -----------------------------------------------------

    fig = px.pie(
        filtered_sentiment,
        names="Sentiment",
        values="Messages",
        hole=0.45,
        title="Selected Sentiment Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------
    # TOP TICKERS
    # -----------------------------------------------------

    st.subheader("📈 Top 20 Mentioned Ticker Symbols")

    top_tickers = dashboard_data["top_tickers"]

    fig = px.bar(
        top_tickers.sort_values("Frequency"),
        x="Frequency",
        y="Ticker",
        orientation="h",
        title="Most Frequently Mentioned Tickers"
    )

    fig.update_layout(
        xaxis_title="Frequency",
        yaxis_title="Ticker"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # TICKER SENTIMENT
    # -----------------------------------------------------

    st.subheader("📊 Sentiment by Top Tickers")

    ticker_sentiment = dashboard_data["ticker_sentiment"]

    filtered_ticker_sentiment = ticker_sentiment[
        ticker_sentiment["Sentiment"].isin(selected_sentiments)
    ]

    fig = px.bar(
        filtered_ticker_sentiment,
        x="Ticker",
        y="Messages",
        color="Sentiment",
        barmode="group",
        title="Bullish vs Bearish Sentiment by Ticker"
    )

    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Messages"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # TOP WORDS
    # -----------------------------------------------------

    st.subheader("🔤 Most Frequent Words")

    top_words = dashboard_data["top_words"]

    fig = px.bar(
        top_words.sort_values("Frequency"),
        x="Frequency",
        y="Word",
        orientation="h",
        title="Top 20 Most Frequent Words"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # TOP BIGRAMS
    # -----------------------------------------------------

    st.subheader("🔤 Most Frequent Word Pairs")

    top_bigrams = dashboard_data["top_bigrams"]

    fig = px.bar(
        top_bigrams.sort_values("Frequency"),
        x="Frequency",
        y="Bigram",
        orientation="h",
        title="Top 20 Most Frequent Bigrams"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    st.markdown(
        """
        The models were evaluated using Accuracy, Precision,
        Recall and F1 Score.
        """
    )

    # -----------------------------------------------------
    # MODEL RESULTS
    # -----------------------------------------------------

    results = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Linear SVM",
            "Multinomial Naive Bayes",
            "Random Forest"
        ],

        "Accuracy": [
            0.695680,
            0.692060,
            0.683460,
            0.620160
        ],

        "Precision": [
            0.691473,
            0.688529,
            0.673572,
            0.590791
        ],

        "Recall": [
            0.773508,
            0.769826,
            0.786656,
            0.932229
        ],

        "F1 Score": [
            0.730194,
            0.726912,
            0.725735,
            0.723237
        ]
    })

    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    best_model = results.loc[
        results["F1 Score"].idxmax()
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Best Model",
        best_model["Model"]
    )

    col2.metric(
        "Accuracy",
        f"{best_model['Accuracy']:.2%}"
    )

    col3.metric(
        "Recall",
        f"{best_model['Recall']:.2%}"
    )

    col4.metric(
        "F1 Score",
        f"{best_model['F1 Score']:.2%}"
    )

    st.markdown("---")

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    performance_long = results.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        performance_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        range_y=[0, 1],
        title="Model Performance Comparison"
    )

    fig.update_layout(
        yaxis_title="Score",
        xaxis_title="Model"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # RESULTS TABLE
    # -----------------------------------------------------

    st.subheader("Performance Metrics")

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}"
        }),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------
    # BASELINE VS TUNED
    # -----------------------------------------------------

    st.subheader("⚙️ Baseline vs Tuned Logistic Regression")

    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Baseline": [
            0.695680,
            0.691473,
            0.773508,
            0.730194
        ],

        "Tuned": [
            0.692220,
            0.684485,
            0.782636,
            0.730278
        ]
    })

    comparison_long = comparison.melt(
        id_vars="Metric",
        var_name="Model",
        value_name="Score"
    )

    fig = px.bar(
        comparison_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        range_y=[0, 1],
        title="Baseline vs Tuned Logistic Regression"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Hyperparameter tuning used C=0.1 and solver='liblinear'. "
        "The tuned model slightly improved recall and F1 score, "
        "while accuracy and precision decreased."
    )


# =========================================================
# LIVE PREDICTION
# =========================================================

elif page == "🔮 Live Prediction":

    st.title("🔮 StockTwits Sentiment Prediction")

    st.write(
        "Enter a StockTwits message below to predict whether "
        "the sentiment is Bullish or Bearish."
    )

    text = st.text_area(
        "Enter your StockTwits message:",
        placeholder=(
            "Example: This stock is going to rise strongly today!"
        ),
        height=150
    )

    if st.button(
        "Predict Sentiment",
        type="primary"
    ):

        if not text.strip():

            st.warning(
                "Please enter some text before making a prediction."
            )

        else:

            try:

                # Transform text using saved TF-IDF
                text_tfidf = tfidf.transform([text])

                # Prediction
                prediction = model.predict(
                    text_tfidf
                )[0]

                # Probability
                probabilities = model.predict_proba(
                    text_tfidf
                )[0]

                bearish_probability = probabilities[0]
                bullish_probability = probabilities[1]

                st.markdown("---")

                st.subheader("Prediction")

                if prediction == 1:

                    st.success(
                        "📈 Sentiment: BULLISH"
                    )

                elif prediction == 0:

                    st.error(
                        "📉 Sentiment: BEARISH"
                    )

                else:

                    st.info(
                        f"Sentiment: {prediction}"
                    )

                # -------------------------------------------------
                # PROBABILITY
                # -------------------------------------------------

                st.subheader(
                    "Prediction Confidence"
                )

                probability_df = pd.DataFrame({

                    "Sentiment": [
                        "Bearish",
                        "Bullish"
                    ],

                    "Probability": [
                        bearish_probability,
                        bullish_probability
                    ]
                })

                fig = px.bar(
                    probability_df,
                    x="Sentiment",
                    y="Probability",
                    range_y=[0, 1],
                    text="Probability",
                    title="Prediction Probability"
                )

                fig.update_traces(
                    texttemplate="%{text:.1%}",
                    textposition="outside"
                )

                fig.update_layout(
                    yaxis_title="Probability",
                    xaxis_title="Sentiment"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

                st.write(
                    "Loaded model type:",
                    type(model).__name__
                )

                st.write(
                    "Loaded vectorizer type:",
                    type(tfidf).__name__
                )
