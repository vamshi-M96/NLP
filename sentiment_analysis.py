# sentiment_analysis.py

import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

def run_sentiment_analysis_mode():
    st.subheader("📄 Upload CSV/XLSX for Sentiment Analysis")

    file = st.file_uploader("📤 Upload CSV or Excel File", type=["csv", "xlsx"])
    if file:
        # Load dataset
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
            return

        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        # Select column for sentiment
        text_col = st.selectbox("🧾 Select Text Column for Sentiment Analysis", df.columns)

        if st.button("🧠 Analyze Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                df["Polarity"], df["Subjectivity"] = zip(*df[text_col].astype(str).map(analyze_sentiment))

            st.subheader("📊 Sentiment Results")
            st.dataframe(df[[text_col, "Polarity", "Subjectivity"]])

            # Sentiment Summary
            st.subheader("📈 Polarity Distribution")
            fig, ax = plt.subplots()
            df["Polarity"].hist(bins=20, edgecolor='black', ax=ax)
            ax.set_title("Sentiment Polarity Distribution")
            ax.set_xlabel("Polarity")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            st.download_button("📥 Download Results as CSV", df.to_csv(index=False), file_name="sentiment_results.csv")

