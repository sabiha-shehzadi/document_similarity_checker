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
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #718096;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header Section
# ---------------------------
st.markdown("<div class='title'>🧠 Document Similarity Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload multiple text files to visualize and compare their similarity using TF-IDF and cosine similarity.</div>", unsafe_allow_html=True)

# ---------------------------
# File Upload Section
# ---------------------------
uploaded_files = st.file_uploader(
    "📂 Upload your text files (TXT format preferred):",
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

    # Heatmap Visualization
    st.markdown("### 🔥 Heatmap Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(similarity_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Document Similarity Heatmap", fontsize=14)
    st.pyplot(fig)

    # Summary
    st.markdown("### 🧾 Summary Insights")
    st.write(f"Total documents uploaded: **{len(uploaded_files)}**")
    st.write("The darker the red color in the heatmap, the higher the similarity between two files. Blue indicates low similarity.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("<div class='footer'>Developed by <b>Sabiha Khan</b> | Powered by Streamlit & Scikit-Learn</div>", unsafe_allow_html=True)
