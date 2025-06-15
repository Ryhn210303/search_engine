import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi, BM25Plus

# Download stopwords jika belum
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('indonesian'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def tokenize(text):
    stop_words = set(stopwords.words('indonesian'))
    tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    return [t for t in tokens if t not in stop_words]

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("berita_politik.xlsx")
    df['text'] = df['judul'].astype(str) + " " + df['isi'].astype(str)
    df['clean_text'] = df['text'].apply(preprocess)
    df['tokens'] = df['text'].apply(tokenize)
    return df

# Search function
def search_news(query, method="Cosine"):
    k = 5  # selalu ambil 5 berita
    query_clean = preprocess(query)

    if method == "Cosine Similarity":
        query_vec = tfidf_vectorizer.transform([query_clean])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    elif method == "BM25":
        query_tokens = tokenize(query)
        similarity = bm25.get_scores(query_tokens)
    elif method == "BM25+":
        query_tokens = tokenize(query)
        similarity = bm25_plus.get_scores(query_tokens)
    else:
        st.error("Metode tidak dikenali.")
        return pd.DataFrame()

    top_indices = similarity.argsort()[-k:][::-1]
    results = df.iloc[top_indices][['judul', 'isi', 'link']].copy()
    results['score'] = similarity[top_indices]
    return results

# Load data dan vektor
df = load_data()

# Cosine - TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

# BM25 dan BM25+
bm25 = BM25Okapi(df['tokens'].tolist())
bm25_plus = BM25Plus(df['tokens'].tolist())

# Streamlit UI
st.title("Search Engine Berita Politik")

query = st.text_input("Masukkan kata kunci (misal: pemilu presiden)")
method = st.selectbox("Pilih metode similarity", ["Cosine Similarity", "BM25", "BM25+"])


if query:
    with st.spinner("Mencari berita..."):
        results = search_news(query, method=method)
        if not results.empty:
            st.success(f"Hasil pencarian teratas dengan metode {method}:")
            for i, row in results.iterrows():
                st.markdown(f"### [{row['judul']}]({row['link']})")
                st.write(row['isi'][:300] + "...")
                st.caption(f"Skor Kemiripan: {row['score']:.4f}")
                st.markdown("---")
        else:
            st.warning("Tidak ditemukan hasil relevan.")
