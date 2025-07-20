import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import os
import pickle
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from functools import partial
import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#st.set_page_config(page_title="Auto NLP App", layout="wide")

def run_nlp_csv_upload_mode():
    
    #st.title("📄  NLP in csv or xlsx ")

    # Initialize once
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Text Cleaning Function
    def clean_text(text):
        # Basic cleaning
        text = text.lower()
        text = re.sub('\n', ' ', text)
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub("[0-9]+", '', text)
        text = re.sub('[‘’“”…]', '', text)
        
        return text


    def show_wordcloud(text):
        wc = WordCloud(background_color='white', width=800, height=400).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)


    # Each function fits a model and returns train/test accuracy and best model

    def logistic_r(x_train, y_train, x_test, y_test):
        grid = GridSearchCV(LogisticRegression(max_iter=1000), 
                            param_grid={'C': [0.1, 1, 10]}, 
                            cv=3, n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        return accuracy_score(y_train, best_model.predict(x_train)) * 100, accuracy_score(y_test, best_model.predict(x_test)) * 100, best_model

    def random_forest(x_train, y_train, x_test, y_test):
        grid = GridSearchCV(RandomForestClassifier(), 
                            param_grid={'n_estimators': [100, 200], 'max_depth': [None, 20]}, 
                            cv=3, n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        return accuracy_score(y_train, best_model.predict(x_train)) * 100, accuracy_score(y_test, best_model.predict(x_test)) * 100, best_model

    def svc(x_train, y_train, x_test, y_test):
        grid = GridSearchCV(SVC(), 
                            param_grid={'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
                            cv=3, n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        return accuracy_score(y_train, best_model.predict(x_train)) * 100, accuracy_score(y_test, best_model.predict(x_test)) * 100, best_model

    def bagging(x_train, y_train, x_test, y_test):
        grid = GridSearchCV(BaggingClassifier(), 
                            param_grid={'n_estimators': [10, 50]}, 
                            cv=3, n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        return accuracy_score(y_train, best_model.predict(x_train)) * 100, accuracy_score(y_test, best_model.predict(x_test)) * 100, best_model

    def NB(x_train, y_train, x_test, y_test):
        grid = GridSearchCV(MultinomialNB(), 
                            param_grid={'alpha': [0.1, 1.0]}, 
                            cv=3, n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        return accuracy_score(y_train, best_model.predict(x_train)) * 100, accuracy_score(y_test, best_model.predict(x_test)) * 100, best_model

    def knn(x_train, y_train, x_test, y_test):
        grid = GridSearchCV(KNeighborsClassifier(), 
                            param_grid={'n_neighbors': [3, 5]}, 
                            cv=3, n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        return accuracy_score(y_train, best_model.predict(x_train)) * 100, accuracy_score(y_test, best_model.predict(x_test)) * 100, best_model

    def decision_tree(x_train, y_train, x_test, y_test, crit='gini'):
        grid = GridSearchCV(DecisionTreeClassifier(criterion=crit), 
                            param_grid={'max_depth': [None, 10, 20]}, 
                            cv=3, n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        return accuracy_score(y_train, best_model.predict(x_train)) * 100, accuracy_score(y_test, best_model.predict(x_test)) * 100, best_model

    @st.cache_resource(show_spinner=True)
    def train_all_models(_train_x, _train_y, _test_x, _test_y, crit='gini'):
        models = [
            logistic_r,
            random_forest,
            svc,
            bagging,
            NB,
            knn,
            partial(decision_tree, crit=crit)
        ]
        results = [m(_train_x, _train_y, _test_x, _test_y) for m in models]
        df = pd.DataFrame(results, columns=['Train Accuracy', 'Test Accuracy', 'Model'])
        df['Model Name'] = ['LogReg', 'RF', 'SVC', 'Bagging', 'NB', 'KNN', 'DT']
        return df

    # Streamlit UI
    tabs = st.tabs(["🔍 Text Classification", "📊 Predictions"])

    with tabs[0]:
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
                target_column = st.selectbox("Select Target Column", df.columns)

            df.dropna(subset=[text_column, target_column], inplace=True)

            use_custom_stopwords = st.toggle("✂️ Use custom stopwords?", value=True)

            if use_custom_stopwords:
                stop = st.text_area("✂️ Custom stopwords (comma-separated)", "a,an,the", height=80).split(",")
                stop = [s.strip().lower() for s in stop if s.strip()]
            else:
                stop = list(stop_words)

            # Clean Text
            df['clean_text'] = df[text_column].astype(str).apply(clean_text)
            df['clean_text'] = df['clean_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

            # WordCloud
            all_text = ' '.join(df['clean_text'])
            st.subheader("Word Cloud")
            show_wordcloud(all_text)

            # Encode labels
            le = LabelEncoder()
            df['encoded_target'] = le.fit_transform(df[target_column])

            # Vectorization
            vectorizer_choice = st.radio("Vectorizer", ["CountVectorizer", "TF-IDF"])
            max_features = st.slider("Max Features", 100, 10000, 1000, step=100)

            if vectorizer_choice == "CountVectorizer":
                vectorizer = CountVectorizer(max_features=max_features)
            else:
                vectorizer = TfidfVectorizer(max_features=max_features)

            X = vectorizer.fit_transform(df['clean_text'])
            y = df['encoded_target']

            train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

            crit = st.selectbox("Decision Tree Criterion", ["gini", "entropy"])

            st.subheader("Training Models")
            results = train_all_models(train_x.astype('float32'), train_y, test_x.astype('float32'), test_y, crit)

            st.dataframe(results[['Model Name', 'Train Accuracy', 'Test Accuracy']])

            best_model_row = results.loc[results['Test Accuracy'].idxmax()]
            st.success(f"🏆Best Model: {best_model_row['Model Name']} with Test Accuracy: {best_model_row['Test Accuracy']:.2f}%")
            st.session_state.best_model_row = best_model_row

            st.sidebar.subheader("Download Models")
            for idx, row in results.iterrows():
                model = row['Model']
                model_name = row['Model Name']
                with open(f"{model_name}_bundle.pkl", "wb") as f:
                    pickle.dump({
                        "model": model,
                        "vectorizer": vectorizer,
                        "label_encoder": le,
                        "Test Accuracy": row['Test Accuracy']
                    }, f)
                with open(f"{model_name}_bundle.pkl", "rb") as f:
                    st.sidebar.download_button(
                        label=f"Download {model_name} Model Bundle",
                        data=f,
                        file_name=f"{model_name}_bundle.pkl"
                    )

            st.subheader("Download Processed Data")
            csv_data = df[[text_column, target_column, 'clean_text']].to_csv(index=False).encode('utf-8')
            st.download_button("Download Final Processed CSV", data=csv_data, file_name="processed_text_data.csv", mime='text/csv')

    with tabs[1]:
        st.header("📈 Prediction & Evaluation")
        #st.success(f"Best Model: {best_model_row['Model Name']} with Test Accuracy: {best_model_row['Test Accuracy']:.2f}%")
        # ✅ Safely access best_model_row
        if "best_model_row" in st.session_state:
            best_model_row = st.session_state.best_model_row
            st.success(f"Best Model: {best_model_row['Model Name']} with Test Accuracy: {best_model_row['Test Accuracy']:.2f}%")
        else:
            st.warning("⚠️ No trained model found. Please train a model in Tab 0 first.")
            st.stop()


        available_models = [f for f in os.listdir() if f.endswith("_bundle.pkl")]
        if available_models:

            model_bundles = []
            for file in available_models:
                try:
                    with open(file, "rb") as f:
                        bundle = pickle.load(f)
                        test_acc = bundle.get("Test Accuracy")
                        if test_acc is not None:
                            model_bundles.append({
                                "file": file,
                                "Model Name": bundle.get("Model Name", file),
                                "Test Accuracy": test_acc,
                                "Bundle": bundle
                            })
                except Exception as e:
                    st.warning(f"⚠️ Error loading {file}: {e}")

            # Step 2: Suggest best model
            if model_bundles:
                best_model_info = max(model_bundles, key=lambda x: x["Test Accuracy"])
                st.success(f"🏆 Best Model: **{best_model_info['Model Name']}** ({best_model_info['Test Accuracy']:.2f}%)")


            selected_model_file = st.selectbox("📦 Select a model for prediction", available_models)
            st.write(f"✅ Selected Model: **{selected_model_file}**")

                    # Load the actual model bundle from the file
            with open(selected_model_file, "rb") as f:
                model_bundle = pickle.load(f)

            # Now you can safely access dictionary keys
            test_accuracy = model_bundle.get("Test Accuracy", None)
            if test_accuracy is not None:
                st.metric(label="Test Accuracy", value=f"{test_accuracy:.2f}%")
            else:
                st.warning("⚠️ 'Test Accuracy' not found in the model bundle.")
            
            # Load model bundle
            #model_name = best_model_row["Model Name"]
            bundle_path = selected_model_file

            if os.path.exists(bundle_path):
                with open(bundle_path, "rb") as f:
                    bundle = pickle.load(f)
                    best_model = bundle["model"]
                    vectorizer = bundle["vectorizer"]
                    label_encoder = bundle["label_encoder"]
            else:
                st.error(f"❌ Model bundle file '{bundle_path}' not found. Please re-download it from Tab 0.")
                st.stop()

            # Prediction mode selection
            prediction_mode = st.radio("Choose Prediction Mode", ["🔤 Single Text Prediction", "📂 Batch CSV Prediction"])

            # SINGLE TEXT PREDICTION
            if prediction_mode == "🔤 Single Text Prediction":
                user_input = st.text_area("✍️ Enter text for prediction")

                if st.button("🔮 Predict"):
                    if not user_input.strip():
                        st.warning("Please enter some text.")
                    else:
                        cleaned = clean_text(user_input)
                        cleaned = " ".join([word for word in cleaned.split() if word not in stop])
                        vec_input = vectorizer.transform([cleaned])
                        prediction = best_model.predict(vec_input)
                        predicted_label = label_encoder.inverse_transform(prediction)[0]
                        st.success(f"🧠 Predicted Label: **{predicted_label}**")

            # BATCH PREDICTION
            elif prediction_mode == "📂 Batch CSV Prediction":
                batch_file = st.file_uploader("📤 Upload CSV file for batch prediction", type=["csv"], key="batch_file")

                if batch_file:
                    df_pred = pd.read_csv(batch_file)

                    # Let user choose the text column
                    pred_text_col = st.selectbox("🧾 Select Text Column", df_pred.columns)

                    if st.button("🚀 Run Batch Prediction"):
                        df_pred['clean_text'] = df_pred[pred_text_col].astype(str).apply(clean_text)
                        df_pred['clean_text'] = df_pred['clean_text'].apply(
                            lambda x: " ".join([word for word in x.split() if word not in stop])
                        )

                        try:
                            X_pred = vectorizer.transform(df_pred['clean_text'])
                            predictions = best_model.predict(X_pred)
                            df_pred['Predicted_Label'] = label_encoder.inverse_transform(predictions)
                            st.success("✅ Predictions completed!")

                            st.dataframe(df_pred[[pred_text_col, 'Predicted_Label']].head())

                            csv = df_pred.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

                        except ValueError as ve:
                            if "features" in str(ve):
                                st.error("❌ Feature mismatch! Please ensure you're using the **same vectorizer and model** used during training.")
                            else:
                                st.error(f"❌ ValueError: {ve}")
        else:
            st.warning("⚠️ Please train and select a model in Tab 0 first.")



st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style="text-align: center; color: grey; font-size: 20px;">
        🧠 <strong>Automated Natural Language Processing (NLP) App</strong> by 
        <a href="https://www.linkedin.com/in/meka-vamshi-/" target="_blank" style="color: blue; text-decoration: none;">
            <strong>Vamshi</strong>
        </a> | Built with Streamlit 💻
    </div>
    """,
    unsafe_allow_html=True
)
