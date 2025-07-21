# zero_shot.py

import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_zero_shot_pipeline():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def run_zero_shot():
    st.title("🧠 Zero-Shot Text Classification")

    # Load model
    classifier = load_zero_shot_pipeline()

    # User input text
    text = st.text_area("📄 Enter text to classify", height=200)

    # Preset label groups
    preset_options = {
        "None (custom)": [],
        "Topic": ["Politics", "Sports", "Technology", "Business"],
        "Emotion": ["Joy", "Anger", "Sadness", "Fear"],
        "Intent": ["Order", "Cancel", "Refund", "Track"]
    }

    preset_choice = st.selectbox("🎯 Choose a preset or enter custom labels", list(preset_options.keys()))
    
    default_labels = ", ".join(preset_options[preset_choice]) if preset_choice != "None (custom)" else ""

    user_labels = st.text_input("🔖 Enter comma-separated labels:", value=default_labels)

    # Button
    if st.button("🔍 Classify"):
        if not text or not user_labels:
            st.warning("Please enter text and labels.")
        else:
            labels = [label.strip() for label in user_labels.split(",") if label.strip()]
            with st.spinner("Classifying..."):
                result = classifier(text, candidate_labels=labels)
                st.subheader("🔎 Classification Result")
                for lbl, score in zip(result['labels'], result['scores']):
                    st.write(f"**{lbl}**: {score:.2%}")
