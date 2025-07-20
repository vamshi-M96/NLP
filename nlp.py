import streamlit as st
import os

# --- Page Config ---
st.set_page_config(page_title="🧠 NLP Launcher", layout="wide", page_icon="🧠")

# --- CSS for Glass Cards ---
st.markdown("""
    <style>
    .glass-card {
        background: rgba(63, 81, 181, 0.25);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        color: White !important;
        font-weight: bold;
        font-size: 1.3rem;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        text-decoration: none;
        display: block;
    }
    .glass-card:hover {
        transform: scale(1.05);
        background: rgba(244, 67, 54, 0.25);
    }
    .card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Get Mode from URL ---
mode_selected = st.query_params.get("mode", None)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🧠 NLP Classification App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Select your input type to begin</p>", unsafe_allow_html=True)

# --- Card Selection ---
if not mode_selected:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <a href="?mode=folder" class="glass-card">
                <div class="card-icon">📁</div>
                Folder Path <br>(PDF / DOCX / TXT)
            </a>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <a href="?mode=file" class="glass-card">
                <div class="card-icon">📄</div>
                CSV / Excel File <br>(Text + Label)
            </a>
        """, unsafe_allow_html=True)

# --- Folder Workflow ---
if mode_selected == "folder":
    st.header("📁 Folder-Based Classification")
       
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
