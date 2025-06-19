import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi, BM25Plus
from datetime import datetime

try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

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
@st.cache_data
def load_data():
    df = pd.read_excel("berita_politik.xlsx")
    df['text'] = df['judul'].astype(str) + " " + df['isi'].astype(str)
    df['clean_text'] = df['text'].apply(preprocess)
    df['tokens'] = df['text'].apply(tokenize)
    if 'tanggal' not in df.columns:
        df['tanggal'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(range(len(df)), unit='d')
    return df

def search_news(query, method="Cosine Similarity", k=10):
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
    available_cols = ['judul', 'isi', 'tanggal']
    if 'url' in df.columns:
        available_cols.append('url')

    results = df.iloc[top_indices][available_cols].copy()
    results['score'] = similarity[top_indices]
    return results
def evaluate_precision(queries, method):
    correct = 0
    for query in queries:
        results = search_news(query, method, k=10)
        if not results.empty:
            correct += 1
    return correct / len(queries)

df = load_data()
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])
bm25 = BM25Okapi(df['tokens'].tolist())
bm25_plus = BM25Plus(df['tokens'].tolist())

st.title("🔍 Search Engine Berita Politik")
query = st.text_input("Masukkan kata kunci (misal: pemilu presiden)")
method = st.selectbox("Pilih metode similarity", ["Cosine Similarity", "BM25", "BM25+"])
search_button = st.button("Cari Berita")

if search_button and query:
    with st.spinner("Mencari berita..."):
        results = search_news(query, method=method, k=10)
        if not results.empty:
            st.success(f"Hasil pencarian teratas dengan metode {method}:")
            for _, row in results.iterrows():
                st.markdown(f"### {row['judul']}")
                st.write(f"🗓️ Tanggal: {row['tanggal'].strftime('%d-%m-%Y')}")
                st.write(row['isi'][:500] + "...")
                st.caption(f"Skor Kemiripan: {row['score']:.4f}")
                st.markdown("---")
        else:
            st.warning("Tidak ditemukan hasil relevan.")

if st.checkbox("Lihat evaluasi presisi dari 10 query"):
    test_queries = [
        "pemilu presiden", "demokrasi", "kebijakan pemerintah", "politik luar negeri", "korupsi",
        "parlemen", "kandidat", "kampanye", "pemerintah daerah", "isu HAM"
    ]
    st.write("Presisi pencarian dari 10 query:")
    for method in ["Cosine Similarity", "BM25", "BM25+"]:
        precision = evaluate_precision(test_queries, method)
        st.write(f"- {method}: {precision:.2f}")
