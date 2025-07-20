# 🧠 Automated Natural Language Processing (NLP) App

A Streamlit-based web app that allows users to upload raw text files or document datasets (PDF, DOCX, TXT, XLSX), preprocess the text using **NLTK**, visualize word patterns, train classification models using **TF-IDF** or **CountVectorizer**, and generate predictions — all in an interactive interface.

---

## 🚀 Features

- 📂 **Multi-format File Input**:
  - Supports uploading files or folders in PDF, DOCX, TXT, and XLSX format.
- 🧹 **Text Preprocessing**:
  - Clean text, remove stopwords, punctuation, digits, symbols
  - NLTK-based tokenization, lemmatization, and POS tagging
- ☁️ **Word Cloud Visualization**
  - Generates a word cloud from cleaned text
- 🧠 **Model Training & Evaluation**:
  - Train multiple models (Logistic Regression, Random Forest, SVC, XGBoost, LGBM, KNN, etc.)
  - Vectorize using **CountVectorizer** or **TF-IDF**
  - Auto model comparison with accuracy scores
- 🧾 **Prediction Module**:
  - Upload new text files for single or batch prediction using trained model
- 💾 **Model Download**:
  - Save and export trained models for deployment

---

## 🧰 Tech Stack

- **Python 3.8+**
- **Streamlit**
- **NLTK**
- **scikit-learn**
- **XGBoost / LightGBM**
- **Matplotlib / WordCloud**
- **PyPDF2 / docx2txt / Pandas**

---

## 📂 Folder Structure

