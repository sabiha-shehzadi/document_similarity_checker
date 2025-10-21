import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.helper_function import extract_text_from_pdf, clean_text, calculate_similarity

st.set_page_config(page_title="Document Similarity Checker", page_icon="📚")

st.title("📚 Document Similarity & Plagiarism Checker")
st.write("Upload 2 or 3 student PDFs and check their similarity using TF-IDF & Cosine Similarity.")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.info("Processing uploaded PDFs... ⏳")
    
    # Extract and clean text
    docs = []
    file_names = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        cleaned = clean_text(text)
        docs.append(cleaned)
        file_names.append(file.name)
    
    # Calculate similarity
    sim_matrix = calculate_similarity(docs)
    
    st.success("✅ Similarity analysis completed!")
    
    # Show as table
    df = pd.DataFrame(np.round(sim_matrix, 2), columns=file_names, index=file_names)
    st.dataframe(df)

    # --- 🔥 Add Similarity Heatmap Visualization ---
    st.write("### 🔥 Visual Similarity Heatmap")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    st.caption("Darker colors indicate higher similarity between documents.")
    # --- End of Heatmap Section ---
    
    # Highlight remarks
    st.write("### 📊 Interpretation:")
    for i in range(len(file_names)):
        for j in range(i+1, len(file_names)):
            percent = round(sim_matrix[i][j]*100, 2)
            if percent > 80:
                remark = "⚠️ Very High Similarity (Possible Plagiarism)"
            elif percent > 50:
                remark = "🟡 Moderate Similarity"
            else:
                remark = "🟢 Low Similarity"
            st.write(f"**{file_names[i]} ↔ {file_names[j]}:** {percent}% → {remark}")
