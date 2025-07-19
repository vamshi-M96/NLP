# Standard library
import os
import re
import numpy as np
import time
import string
import pickle
from glob import glob

# Third-party libraries
import pandas as pd
import streamlit as st
import docx2txt
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# Replacing spaCy with NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import NotFittedError
from sklearn.exceptions import FitFailedWarning
from sklearn.utils.validation import check_is_fitted


# Load NLTK-based text processing (instead of spaCy)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4') 
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


def text_clean(text):
    
    text = text.lower()
    text = re.sub('\n', '', text)
    text = re.sub('\t', '', text)
    text = re.sub('\[.]', '', text)
    text = re.sub('\[*!?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    return text
clean = lambda x: text_clean(x)

# Stopword Removal
def remove_stopwords(text, stopwords_set= STOP_WORDS):
    return " ".join([word for word in text.split() if word.lower() not in stopwords_set])

# WordCloud
def plot_wordcloud(text, stopwords):
    wc = WordCloud(width=800, height=400, stopwords=stopwords, background_color='white').generate(text)
    st.image(wc.to_array(), use_container_width=True)

def convert_to_docx(pth):
    files = glob(os.path.join(pth, '**', '*.doc'), recursive=True)
    for file in files:
        try:
            import doc2docx
            doc2docx.convert(file)
            os.remove(file)
        except Exception as e:
            st.warning(f"❌ Failed to convert {file}: {e}")

@st.cache_data(show_spinner=False)
def extract_from_folder(path):
    data = []
    for root, _, files in os.walk(path):
        for file in files:
            ext = file.lower().split('.')[-1]
            p = os.path.join(root, file)
            try:
                if ext == "docx":
                    text = docx2txt.process(p)
                elif ext == "pdf":
                    reader = PdfReader(p)
                    text = "".join(p.extract_text() for p in reader.pages if p.extract_text())
                elif ext == "txt":
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                else:
                    continue
                data.append((text, os.path.basename(root)))
            except:
                continue
    return pd.DataFrame(data, columns=["Content", "Category"])


#CLASSIFICATION
def classification(x,y):

    best_model = None
    st.subheader("📈 Classification")


    #logistic ression
    def logistic_r (train_x,train_y,test_x,test_y):
        le = LogisticRegression()
        model_lr = le.fit(train_x,train_y)

        lr_train_predict = model_lr.predict(train_x)
        lr_test_predict = model_lr.predict(test_x)

        lr_train_acc = accuracy_score(train_y,lr_train_predict)*100
        lr_test_acc = accuracy_score(test_y,lr_test_predict)*100

        return lr_train_acc,lr_test_acc,model_lr

    #Random forest
    def random_forest (train_x, train_y,test_x,test_y):
        kfold = KFold(n_splits=10, random_state=5,shuffle=True)
        n_estimators = np.array(range(10,50)) 
        max_feature = [2,3,4,5,6]
        param_grid = dict(n_estimators =n_estimators,max_features=max_feature)

        model_rfc = RandomForestClassifier()
        grid_rfc = GridSearchCV(estimator=model_rfc, param_grid=param_grid)
        grid_rfc.fit(train_x, train_y)

        best_model = grid_rfc.best_estimator_

        RFC_Model = RandomForestClassifier(n_estimators=grid_rfc.best_params_['n_estimators'],max_features=grid_rfc.best_params_['max_features'])
        RFC_Model.fit(train_x,train_y)

        RFC_train_predict = RFC_Model.predict(train_x)
        RFC_test_predict = RFC_Model.predict(test_x)

        rfc_train_acc = accuracy_score(train_y,RFC_train_predict)*100
        rfc_test_acc = accuracy_score(test_y,RFC_test_predict)*100

        return rfc_train_acc,rfc_test_acc, best_model

    #support vector clasifer

    #Support Vector clasiffiers
    def svc(train_x,train_y,test_x,test_y):


        clf = SVC()
        param_grid_svc = [{'kernel':['rbf','sigmoid','poly'],'gamma':[0.5,0.1,0.005],'C':[25,20,10,0.1,0.001] }]
        
        # Determine the minimum number of samples in any class
        train_y_series = pd.Series(train_y)

        # Safely compute number of samples per class
        min_class_samples = train_y_series.value_counts().min()
        safe_cv = min(5, min_class_samples)

        # Warn if dataset is too small
        if safe_cv < 2:
            raise ValueError("Not enough samples in some classes for cross-validation.")

        # Avoid warning spam from failed fits
        warnings.simplefilter('ignore', FitFailedWarning)
        
        svc= RandomizedSearchCV(clf,param_grid_svc,cv=safe_cv)
        svc.fit(train_x,train_y)

        svc_train_predict = svc.predict(train_x)
        svc_test_predict = svc.predict(test_x)

        svc_train_acc = accuracy_score(train_y,svc_train_predict)*100
        svc_test_acc = accuracy_score(test_y,svc_test_predict)*100

        return svc_train_acc,svc_test_acc,svc

    #bagging
    def bagging(train_x,train_y,test_x,test_y):
        cart = DecisionTreeClassifier()

        model_bag = BaggingClassifier(estimator=cart, n_estimators= 10, random_state=6)
        model_bag.fit(train_x,train_y)

        bag_train_predict = model_bag.predict(train_x)
        bag_test_predict = model_bag.predict(test_x)

        bag_train_acc = accuracy_score(train_y,bag_train_predict)*100
        bag_test_acc = accuracy_score(test_y,bag_test_predict)*100

        return bag_train_acc,bag_test_acc,model_bag

    #xgb
    def xgb(train_x,train_y,test_x,test_y):
        n_estimators =np.array(range(10,80,10))
        xgb_model = XGBClassifier(n_estimators=70,max_depth=5)
        xgb_model.fit(train_x,train_y)

        xgb_train_predict = xgb_model.predict(train_x)
        xgb_test_predict = xgb_model.predict(test_x)

        xgb_train_acc = accuracy_score(train_y,xgb_train_predict)*100
        xgb_test_acc = accuracy_score(test_y,xgb_test_predict)*100

        return xgb_train_acc,xgb_test_acc,xgb_model

    #LGBM
    def lgbm(train_x,train_y,test_x,test_y):

        params = {}
        params['learning_rate'] = 1
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params['sub_feature'] = 0.5
        params['num_leaves'] = 5
        params['min_data'] = 10
        params['max_depth'] = 5

        lgbm_model = lgb.LGBMClassifier()
        lgbm_model.fit(train_x,train_y)

        lgbm_train_predict = lgbm_model.predict(train_x)
        lgbm_test_predict = lgbm_model.predict(test_x)

        lgbm_train_acc = accuracy_score(train_y,lgbm_train_predict)*100
        lgbm_test_acc = accuracy_score(test_y,lgbm_test_predict)*100

        return lgbm_train_acc,lgbm_test_acc,lgbm_model

    #NaiveByaes
    def NB(train_x, train_y, test_x, test_y):
        # Convert only if needed
        if hasattr(train_x, "toarray"):
            train_x = train_x.toarray()
        if hasattr(test_x, "toarray"):
            test_x = test_x.toarray()

        nb_model = GaussianNB()
        nb_model.fit(train_x, train_y)

        nb_train_predict = nb_model.predict(train_x)
        nb_test_predict = nb_model.predict(test_x)

        nb_train_acc = accuracy_score(train_y, nb_train_predict) * 100
        nb_test_acc = accuracy_score(test_y, nb_test_predict) * 100

        return nb_train_acc, nb_test_acc, nb_model


    #KNN
    def knn(train_x,train_y,test_x,test_y):

        n_neighbors = np.array(range(2,30))
        param_grid = dict(n_neighbors=n_neighbors)

        model = KNeighborsClassifier()
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid.fit(train_x, train_y)

        knn_model = KNeighborsClassifier(grid.best_params_['n_neighbors'])
        knn_model.fit(train_x, train_y)

        knn_train_predict=knn_model.predict(train_x)
        knn_test_predict=knn_model.predict(test_x)

        knn_train_acc = accuracy_score(train_y,knn_train_predict)*100
        knn_test_acc = accuracy_score(test_y,knn_test_predict)*100

        return knn_train_acc,knn_test_acc,knn_model

    #Decision Tree
    def decision_tree(train_x, train_y, test_x, test_y):

        criterion_choice = st.selectbox("Decision Tree Criterion", options=["gini", "entropy"], index=0)

        if criterion_choice == 'gini':
            dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)
            dt_model.fit(train_x, train_y)
        return accuracy_score(train_y, dt_model.predict(train_x)) * 100, accuracy_score(test_y, dt_model.predict(test_x)) * 100, dt_model


    def df(train_x,train_y,test_x,test_y):

        list= [logistic_r (train_x,train_y,test_x,test_y), 
        random_forest (train_x, train_y,test_x,test_y),
        svc(train_x,train_y,test_x,test_y),
        bagging(train_x,train_y,test_x,test_y),
        xgb(train_x,train_y,test_x,test_y),
        lgbm(train_x,train_y,test_x,test_y),
        NB(train_x,train_y,test_x,test_y),
        knn(train_x,train_y,test_x,test_y),
        decision_tree(train_x, train_y, test_x, test_y) ]

        acc_data = pd.DataFrame(list,columns=('Train accuracy','Test accuracy','Model'),index=['logistic','Random_forest','SVC','Bagging','XGB','LGBM','NB',"KNN","Decission Tree"])

        return acc_data

    data = None
    best_model = None

    
        

    start_time= time.time()

    if x is not None and y is not None:
        

        
        st.dataframe(pd.DataFrame(x[:5].toarray()))
        st.dataframe(y.head(5))
              
        le = LabelEncoder()
        y = le.fit_transform(y)

        train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=10)
        
        # ✅ Ensure LightGBM gets float input
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        data = df(train_x,train_y,test_x,test_y)

        st.dataframe(data)
        


    # DOWNLOAD OF DIFFRENT MODELS
        st.sidebar.header('Download required model for Deployment')
        for model_name in data.index:
            model = data.loc[model_name, 'Model']
            file_name = f"{model_name}_model.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(model, f) 

            with open(file_name, "rb") as f:
                
                st.sidebar.download_button(
                    label=f"📥 Download {model_name.capitalize()} Model",
                    data=f.read(),
                    file_name=file_name,
                    mime="application/octet-stream",key=f"download_{model_name}"
                ) 

    # 🎯 Simple Best Classification Model Selector
        st.subheader("Best Classification Model")

        if 'Test accuracy' in data.columns:
            best_row = data.sort_values(by='Test accuracy', ascending=False).iloc[0]
            st.success(f"🏆 Best model based on Test Accuracy: **{best_row.name}**")
            st.write(best_row)
            best_model = best_row['Model']
        else:
            st.warning("⚠️ 'Test Accuracy' column not found in results.")
                
        end_time= time.time()

    
        time_taken = end_time-start_time
        st.session_state.label_encoder = le
        st.session_state.best_model = best_model 
        
        st.success(f"Task complited in {time_taken:.2f} seconds")

        

        return data, best_model        



# File upload or folder input
st.title("🧠 NLP Processor")

option = st.radio("Choose input type", ["📁 Folder", "📄 File Upload"])
df = pd.DataFrame()

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🧠 Modeling", "🔮 Prediction"])

with tab1:

    if option == "📁 Folder":
        folder_path = st.text_input("Enter folder path:")
        if folder_path and os.path.exists(folder_path):
            convert_to_docx(folder_path)
            df = extract_from_folder(folder_path)

    elif option == "📄 File Upload":
        uploaded = st.file_uploader("Upload CSV/XLSX/TXT/PDF", type=["csv", "xlsx", "txt", "pdf"], accept_multiple_files=False)
        if uploaded:
            ext = uploaded.name.split('.')[-1]
            try:
                if ext == "csv":
                    df = pd.read_csv(uploaded)
                elif ext == "xlsx":
                    df = pd.read_excel(uploaded)
                elif ext == "txt":
                    text = uploaded.read().decode("utf-8")
                    df = pd.DataFrame([(text, "Unknown")], columns=["Content", "Category"])
                elif ext == "pdf":
                    reader = PdfReader(uploaded)
                    text = "".join(p.extract_text() for p in reader.pages if p.extract_text())
                    df = pd.DataFrame([(text, "Unknown")], columns=["Content", "Category"])
            except Exception as e:
                st.error(f"❌ Failed to read file: {e}")
    
    # Proceed if DataFrame is valid
    if not df.empty:
        st.success("✅ Data loaded!")
        st.write(df)
        # Ask user to select columns if not default
        all_cols = df.columns.tolist()
        content_col = st.selectbox("📝 Select Content/Text Column", all_cols, index=0)
        category_col = st.selectbox("🏷️ Select Category Column (optional)", ["None"] + all_cols)

        df = df.rename(columns={content_col: "Content"})
        if category_col != "None":
            df["Category"] = df[category_col]
        else:
            df["Category"] = "Unknown"


        st.success("✅ Data loaded!")

        # Optional sampling
        if st.checkbox("🔀 Sample Data?"):
            frac = st.slider("Sample fraction", 0.01, 1.0, 0.2)

            if "Category" in df.columns:
                df_sampled = (
                    df.groupby("Category", group_keys=False)
                    .apply(lambda x: x.sample(frac=frac if len(x) > 1 else 1.0))
                    .reset_index(drop=True)
                )
                df = df_sampled
                st.success("✅ Sampled data while maintaining category balance.")
            else:
                df = df.sample(frac=frac).reset_index(drop=True)
                st.warning("⚠️ 'Category' column not found — used random sampling instead.")


        # Show category distribution
        if "Category" in df.columns:
            st.subheader("📊 Category Distribution")
            cat_counts = df["Category"].value_counts()
            fig, ax = plt.subplots()
            cat_counts.plot(kind='bar', ax=ax, color="skyblue")
            for i, v in enumerate(cat_counts.values):
                ax.text(i, v + 0.1, str(v), ha='center', fontsize=10)
            ax.set_title("Category-wise Distribution")
            st.pyplot(fig)

        # Text cleaning
            st.subheader("🧹 Text Cleaning & Stopword Removal")

            # Toggle switch to use custom stopwords
            use_custom_stopwords = st.toggle("✂️ Use custom stopwords?", value=True)

            if use_custom_stopwords:
                stop = st.text_area("✂️ Custom stopwords (comma-separated)", "a,an,the", height=80).split(",")
                stop = [s.strip().lower() for s in stop if s.strip()]
            else:
                stop = list(STOP_WORDS)

            with st.spinner("Cleaning text..."):
                df["Cleaned"] = df["Content"].astype(str).apply(clean)
                df["No_Stopwords"] = df["Cleaned"].apply(lambda x: remove_stopwords(x, stop))

            st.success("✅ Cleaning done")

        tokens = []
        if st.checkbox("🔡 Show NLP Tokens & Lemmas"):
            text = " ".join(df["No_Stopwords"])
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t.lower() not in stop and t.isalpha()]

        if st.checkbox("📍 Show Tokens"):
            st.write(tokens[:100])

        if st.checkbox("🧬 Show Lemmatized Text"):
            lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
            st.text_area("Lemmatized", " ".join(lemmatized[:500]), height=200)

        if st.checkbox("☁️ Show WordCloud"):
            text = " ".join(df["No_Stopwords"])
            plot_wordcloud(text, stop)
            # Sentiment
        # Show final processed DataFrame with only Category and Cleaned Text
        st.subheader("📄 Final Processed Data")
        final_df = df[["Category", "No_Stopwords"]].rename(columns={"No_Stopwords": "Processed_Text"})
        st.dataframe(final_df)

        # Save processed data and stopwords in session_state
        st.session_state.final_df = final_df
        st.session_state.stopwords_used = stop



with tab2:
    if "final_df" in st.session_state:
        final_df = st.session_state.final_df
        st.subheader("🔠 Vectorization & Modeling")

        # Vectorizer selection (pill-style)

        # Max Features Slider
        MAX_FEATURES = st.slider(
            "🔢 Max Features for Vectorizer",
            min_value=500,
            max_value=20000,
            value=5000,
            step=500,
            help="Limit the number of features to prevent performance issues."
        )
        
        # Vectorizer selection (pill-style)
        vectorizer_choice = st.radio(
            "🔘 Choose Vectorizer",
            options=["Count Vectorizer", "TF-IDF Vectorizer"],
            index=0,
            horizontal=True
        )
        
        if st.button("Enter"):
            # Select and apply the vectorizer with the chosen max_features
            if vectorizer_choice == "Count Vectorizer":
                vectorizer = CountVectorizer(max_features=MAX_FEATURES)
            else:
                vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
        
            st.success(f"{vectorizer_choice} initialized with max {MAX_FEATURES} features.")


            # Vectorize the Processed Text
            X = vectorizer.fit_transform(final_df["Processed_Text"])
            y = final_df["Category"]

            # Save vectorizer and X, y in session_state
            st.session_state.vectorizer = vectorizer
            st.session_state.X = X
            st.session_state.y = y

            st.success(f"✅ {vectorizer_choice} applied! Vocabulary size: `{len(vectorizer.get_feature_names_out())}`")


            results, best_model = classification( X ,y)

            if results is not None:
                st.success("🎉 Classification completed!")
                st.dataframe(results)

with tab3:
    st.subheader("🔮 Make Predictions on New File")

    if 'label_encoder' in st.session_state and 'final_df' in st.session_state and 'best_model' in st.session_state and 'vectorizer' in st.session_state:
        final_df = st.session_state.final_df
        vec = st.session_state.vectorizer
        pred_input = st.file_uploader("📄 Upload a new file for prediction", type=["txt", "pdf", "docx"])

        if pred_input:
            try:
                # Extract text from file
                ext = pred_input.name.split('.')[-1].lower()
                if ext == "txt":
                    raw_text = pred_input.read().decode("utf-8")
                elif ext == "pdf":
                    reader = PdfReader(pred_input)
                    raw_text = "".join(p.extract_text() for p in reader.pages if p.extract_text())
                elif ext == "docx":
                    with open("temp_uploaded.docx", "wb") as f:
                        f.write(pred_input.read())
                    raw_text = docx2txt.process("temp_uploaded.docx")
                    os.remove("temp_uploaded.docx")
                else:
                    st.error("Unsupported file type.")
                    raw_text = ""

                # Preprocess text
                cleaned = text_clean(raw_text)
                stop = list(STOP_WORDS)  # You can also use custom stopwords
                processed = remove_stopwords(cleaned, stop)

                # Vectorize
                if vectorizer_choice == "Count Vectorizer":
                    vec = CountVectorizer()
                else:
                    vec = TfidfVectorizer()

                # Refit vectorizer on entire dataset to match training vocabulary
                
                transformed_input = vec.transform([processed])

                # Predict
                prediction = st.session_state.best_model.predict(transformed_input)
                decoded = st.session_state.label_encoder.inverse_transform(prediction)

                st.success(f"✅ Predicted Category: **{decoded[0]}**")

                st.text_area("📄 Processed Text", processed, height=200)

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    else:
        st.warning("⚠️ Train a classification model first before predicting.")
