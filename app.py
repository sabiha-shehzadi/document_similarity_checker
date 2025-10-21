import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="📚 Document Similarity Checker",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# Custom Page Styling
# ---------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f9fafb;
            padding: 2rem;
            border-radius: 12px;
        }
        .title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: bold;
            color: #2b6cb0;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #4a5568;
            margin-bottom: 2rem;
        }
        /* Sticky Footer Styling */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #edf2f7;
            color: #2d3748;
            text-align: center;
            padding: 10px;
            font-size: 0.9rem;
            border-top: 1px solid #cbd5e0;
            box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
            z-index: 100;
        }
        .footer b {
            color: #2b6cb0;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header Section
# ---------------------------
st.markdown("<div class='title'>🧠 Document Similarity Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload multiple text or PDF files to visualize and compare their similarity using TF-IDF and cosine similarity.</div>", unsafe_allow_html=True)

# ---------------------------
# File Upload Section
# ---------------------------
uploaded_files = st.file_uploader(
    "📂 Upload your documents (TXT, CSV, or PDF):",
    type=["txt", "csv", "pdf"],
    accept_multiple_files=True
)

# ---------------------------
# Main Logic
# ---------------------------
if uploaded_files:
    docs = []
    filenames = []

    for file in uploaded_files:
        try:
            text = file.read().decode("utf-8", errors="ignore")
        except Exception:
            text = str(file.read())
        docs.append(text)
        filenames.append(file.name)

    # Vectorize text data
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)

    # Compute similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    similarity_df = pd.DataFrame(cosine_sim, index=filenames, columns=filenames)

    # Display results
    st.markdown("### 📊 Similarity Matrix")
    st.dataframe(similarity_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    # 🔥 Heatmap Visualization (Compact + Expandable)
    st.markdown("### 🔥 Heatmap Visualization")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
    plt.title("Document Similarity Heatmap", fontsize=14)
    st.pyplot(fig, use_container_width=True)


    # Summary
    st.markdown("### 🧾 Summary Insights")
    st.write(f"Total documents uploaded: **{len(uploaded_files)}**")
    st.write("The darker the red color in the heatmap, the higher the similarity between two files. Blue indicates low similarity.")

# ---------------------------
# Sticky Footer (always visible)
# ---------------------------
st.markdown("<div class='footer'>Developed by <b>Sabiha Khan</b> | Powered by Streamlit & Scikit-Learn</div>", unsafe_allow_html=True)
