import streamlit as st
from transformers import pipeline

@st.cache_resource(show_spinner=True)
def load_toxicity_model():
    return pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)

def run_toxicity_detection():
    #st.title("🛡️ Toxicity / Hate Speech Detection")

    model = load_toxicity_model()

    input_mode = st.radio("Select input mode:", ["Single Text", "Batch CSV Upload"])

    if input_mode == "Single Text":
        user_text = st.text_area("Enter text to check toxicity", height=200)

        if st.button("Analyze"):
            if not user_text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing..."):
                    results = model(user_text)
                    # results is a list of lists: e.g. [[{'label': 'toxicity', 'score': 0.9}, {...}]]
                    scores = {res['label']: res['score'] for res in results[0]}
                    st.subheader("Toxicity Scores")
                    for label, score in scores.items():
                        st.write(f"**{label}**: {score:.2%}")

    else:  # Batch CSV Upload
        uploaded_file = st.file_uploader("Upload CSV file with text column", type=["csv"])
        if uploaded_file:
            import pandas as pd

            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            text_col = st.selectbox("Select text column for analysis", df.columns)

            if st.button("Analyze Batch"):
                with st.spinner("Analyzing..."):
                    df['toxicity_scores'] = df[text_col].apply(lambda x: model(str(x)))
                    
                    # Convert nested results to a dict of label->score
                    def extract_scores(results):
                        return {res['label']: res['score'] for res in results[0]}
                    
                    df['toxicity_dict'] = df['toxicity_scores'].apply(extract_scores)

                    st.write("Sample Results:")
                    st.dataframe(df[[text_col, 'toxicity_dict']].head())

                    # Optionally, save results to CSV for download
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download results CSV", data=csv, file_name="toxicity_results.csv")
