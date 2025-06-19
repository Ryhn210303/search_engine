import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi, BM25Plus

# Download stopwords jika belum ada
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
def search_news(query, method="Cosine Similarity"):
    k = 10
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

    available_cols = ['judul', 'isi']
    if 'url' in df.columns:
        available_cols.append('url')
    if 'tanggal' in df.columns:
        available_cols.append('tanggal')

    results = df.iloc[top_indices][available_cols].copy()
    results['score'] = similarity[top_indices]
    return results

# Evaluation: Precision at k
def precision_at_k(results, k=10):
    if 'label' not in results.columns:
        return None
    return results['label'].head(k).sum() / k

# Load data dan buat vektor
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
        results = search_news(query, method=method)
        if not results.empty:
            st.success(f"Hasil pencarian teratas dengan metode {method}:")
            for _, row in results.iterrows():
                url = row.get('url', '')
                judul = row['judul']
                tanggal = row.get('tanggal', '')
                isi = row['isi'][:300] + "..."

                if isinstance(url, str) and pd.notna(url) and url.strip() != "":
                    if not url.startswith("http"):
                        url = "https://" + url
                    st.markdown(f"### [{judul}]({url})")
                else:
                    st.markdown(f"### {judul}")

                if pd.notna(tanggal):
                    st.caption(f"Tanggal: {tanggal}")
                st.write(isi)
                st.caption(f"Skor Kemiripan: {row['score']:.4f}")
                st.markdown("---")
            
            # Precision if label exists
            prec = precision_at_k(results)
            if prec is not None:
                st.info(f"🎯 Presisi@10: {prec:.2f}")
        else:
            st.warning("Tidak ditemukan hasil relevan.")

# Evaluasi Semua Metode dengan 10 Query (opsional)
with st.expander("📊 Evaluasi Presisi 10 Query"):
    st.markdown("Jika dataset memiliki kolom `label`, sistem dapat mengevaluasi presisi dari 10 query terhadap 3 metode.")
    test_queries = [
        "pemilu presiden", "kpu dan dpt", "kampanye politik", "partai politik baru",
        "pemilihan legislatif", "isu korupsi", "undang-undang pemilu", "politik uang",
        "netralitas ASN", "hasil quick count"
    ]

    eval_data = []
    for q in test_queries:
        for m in ["Cosine Similarity", "BM25", "BM25+"]:
            res = search_news(q, method=m)
            prec = precision_at_k(res)
            eval_data.append({"Query": q, "Metode": m, "Presisi@10": round(prec, 2) if prec is not None else "N/A"})

    eval_df = pd.DataFrame(eval_data)
    st.dataframe(eval_df)
