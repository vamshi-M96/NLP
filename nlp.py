import streamlit as st
import os

# --- Page Config ---
st.set_page_config(page_title="🧠 NLP Launcher", layout="wide", page_icon="🧠")



# 💅 CSS for circular cards
st.markdown("""
<style>
.glass-card {
    width: 440px;
    height: 440px;
    border: 5px solid rgba(156, 39, 176, 0.8);;
    border-radius: 80%;
    background: rgba(0, 0, 0, 0);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    text-decoration: none;
    color: inherit;
    font-size: 1.5rem;
    font-weight: 600;
    padding: 0.5rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
.glass-card:hover {
    background: rgba(63, 81, 181, 0.25);
    transform: scale(1.07);
}
.card-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# --- Get Mode from URL ---
mode_selected = st.query_params.get("mode", None)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🧠 NLP Studio: Analyze, Train, Predict>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Select your input type to begin</p>", unsafe_allow_html=True)


# --- Card Selection ---
if not mode_selected:
    
    # First Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <a href="?mode=folder" class="glass-card">
            <div class="card-icon">📁</div>
            <strong>Zip File Path</strong><br>(PDF / DOCX / TXT)
        </a>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <a href="?mode=file" class="glass-card">
            <div class="card-icon">📄</div>
            <strong>Text Classification</strong><br>CSV / Excel (Text + Label)
        </a>
        """, unsafe_allow_html=True)

    # Second Row
    rcol1, rcol2, rcol3, rcol4 = st.columns(4)

    with rcol2:
        st.markdown("""
        <a href="?mode=file_Regression" class="glass-card">
            <div class="card-icon">📊</div>
            <strong>NLP Regression</strong><br>CSV / Excel (Text + Target)
        </a>
        """, unsafe_allow_html=True)

    with rcol4:
        st.markdown("""
        <a href="?mode=sentiment" class="glass-card">
            <div class="card-icon">💬</div>
            <strong>Sentiment Analysis</strong><br>CSV / Excel (Text Only)
        </a>
        """, unsafe_allow_html=True)


# --- Folder Workflow ---
if mode_selected == "folder":
    st.header("🤐 Zip-File-Based Classification")
       
    try:
        from nlp_filepath import run_nlp_filepath_mode
        run_nlp_filepath_mode()  # 🚫 Do NOT pass folder_path
    except Exception as e:
        st.error(f"❌ Error: {e}")

    if st.button("🔙 Back"):
        st.query_params.clear()

# --- File Workflow ---
if mode_selected == "file":
    st.header("📄 CSV/XLSX File Classification")
    try:
        from nlp_csv_xlsx import run_nlp_csv_upload_mode
        run_nlp_csv_upload_mode()  # 🔥 Let it handle upload + logic internally
    except Exception as e:
        st.error(f"❌ Error: {e}")

    if st.button("🔙 Back"):
        st.query_params.clear()

# --- File Workflow ---
if mode_selected == "file_Regression":
    st.header("📄 CSV/XLSX File Regression")
    try:
        from nlp_csv_xlsx import run_nlp_csv_upload_mode
        run_nlp_csv_upload_mode()  # 🔥 Let it handle upload + logic internally
    except Exception as e:
        st.error(f"❌ Error: {e}")

    if st.button("🔙 Back"):
        st.query_params.clear()

if mode_selected == "sentiment":
    st.header("🧾 Sentiment Analysis")
    try:
        from sentiment_analysis import run_sentiment_analysis_mode
        run_sentiment_analysis_mode()
    except Exception as e:
        st.error(f"❌ Error: {e}")





st.markdown("""
    <style>
    .footer {
        position: relative;
        bottom: 0;
        width: 100%;
        margin-top: 50px;
        padding: 10px 0;
        color: grey;
        font-size: 20px;
        text-align: center;
    }
    </style>

    <div class="footer">
        🧠 <strong>Automated Natural Language Processing (NLP) App</strong> by 
        <a href="https://www.linkedin.com/in/meka-vamshi-/" target="_blank" style="color: blue; text-decoration: none;">
            <strong>Vamshi</strong>
        </a> | Built with Streamlit 💻
    </div>
    """, unsafe_allow_html=True)
