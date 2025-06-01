import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import string
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords jika belum ada
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('indonesian'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load dataset
@st.cache_data
def load_data():
    if not os.path.exists("berita_politik.xlsx"):
        st.error("File 'berita_politik.xlsx' tidak ditemukan.")
        st.stop()
    df = pd.read_excel('berita_politik.xlsx')
    df['text'] = df['judul'].astype(str) + " " + df['isi'].astype(str)
    df['clean_text'] = df['text'].apply(preprocess)
    return df

# Load data
df = load_data()

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

# Fit Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(tfidf_matrix)

# Search function
def search_news(query):
    query_clean = preprocess(query)
    query_vec = vectorizer.transform([query_clean])
    distances, indices = knn_model.kneighbors(query_vec)
    
    results = df.iloc[indices[0]][['judul', 'isi', 'link']].copy()
    results['score'] = 1 - distances[0]  # cosine similarity
    return results

# Streamlit App
st.title("Search Engine Berita Politik (K-NN + TF-IDF)")

query = st.text_input("Masukkan kata kunci (misal: pemilu presiden)", "")

if query:
    with st.spinner("Mencari berita..."):
        results = search_news(query)
        st.success(f"Ditemukan {len(results)} berita relevan:")
        for _, row in results.iterrows():
            link = row['link'] if pd.notna(row['link']) else "#"
            st.markdown(f"### [{row['judul']}]({link})")
            st.write(row['isi'][:300] + "...")
            st.caption(f"Skor Kemiripan: {row['score']:.4f}")
            st.markdown("---")
