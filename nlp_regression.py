import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re, string
from regression import regression

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def run_nlp_csv_upload_mode_regression():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub('\n', ' ', text)
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub("[0-9]+", '', text)
        return text

    def show_wordcloud(text):
        wc = WordCloud(background_color='white', width=800, height=400).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV/XLSX file", type=['csv', 'xlsx'])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.dataframe(df)

        col1, col2 = st.columns(2)
        with col1:
            text_column = st.selectbox("Select Text Column", df.columns)
        with col2:
            target_column = st.selectbox("Select Target Column (numeric)", df.columns)

        df.dropna(subset=[text_column, target_column], inplace=True)

        stop = st.text_area("Custom stopwords (comma-separated)", "a,an,the", height=80).split(",")
        stop = [s.strip().lower() for s in stop if s.strip()]

        df['clean_text'] = df[text_column].astype(str).apply(clean_text)
        df['clean_text'] = df['clean_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

        all_text = ' '.join(df['clean_text'])
        st.subheader("Word Cloud")
        show_wordcloud(all_text)

        vectorizer_choice = st.radio("Vectorizer", ["CountVectorizer", "TF-IDF"])
        max_features = st.slider("Max Features", 100, 10000, 1000, step=100)

        if vectorizer_choice == "CountVectorizer":
            vectorizer = CountVectorizer(max_features=max_features)
        else:
            vectorizer = TfidfVectorizer(max_features=max_features)

        X = vectorizer.fit_transform(df['clean_text'])
        y = df[target_column].astype(float)

        st.subheader("Training Regression Models")
        with st.spinner("Training regression models..."):
            results, best_model = regression(X, y)

        if results is not None:
            st.success("\U0001F389 Regression completed successfully!")
            st.dataframe(results)

            st.session_state.best_model = best_model

            with open("best_regression_model.pkl", "wb") as f:
                pickle.dump({"model": best_model, "vectorizer": vectorizer}, f)

            with open("best_regression_model.pkl", "rb") as f:
                st.sidebar.download_button(
                    label="Download Best Regression Model",
                    data=f,
                    file_name="best_regression_model.pkl"
                )

            st.subheader("Download Processed Data")
            csv_data = df[[text_column, target_column, 'clean_text']].to_csv(index=False).encode('utf-8')
            st.download_button("Download Final Processed CSV", data=csv_data, file_name="processed_text_data.csv", mime='text/csv')

    st.markdown("""
        <br><br><br><br>
        <style>
        .footer {
            position: relative;
            bottom: 0;
            width: 100%;
            margin-top: 50px;
            padding: 10px 0;
            font-size: 16px;
            text-align: center;
            color: grey;
        }
        .footer a {
            color: blue;
            text-decoration: none;
        }
        </style>

        <div class="footer">
            🧠 <strong>Automated NLP Regression App</strong> by 
            <a href="https://www.linkedin.com/in/meka-vamshi-/" target="_blank">
                <strong>Vamshi</strong>
            </a> | Built with Streamlit 💻
        </div>
    """, unsafe_allow_html=True)
