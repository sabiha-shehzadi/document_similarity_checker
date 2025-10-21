# 📚 Document Similarity & Plagiarism Checker

A simple Streamlit web app that compares multiple PDF documents (like student assignments) and calculates their similarity using TF-IDF and Cosine Similarity. It helps detect possible plagiarism or reused content.

🚀 Features

Upload 2 or 3 PDF files

Automatic text extraction & cleaning

View similarity matrix in a table

Clear plagiarism remarks (Low, Moderate, High)

Easy deployment via Streamlit Cloud

# ⚙️ How to Run
 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # for Windows

 2. Install dependencies
pip install -r requirements.txt

 3. Run app
streamlit run app.py

# 🧠 Tech Stack

Python • Streamlit • PyPDF2 • Scikit-learn • NLTK • Pandas • NumPy

🌍 Streamlit Deployment: share.streamlit.io
