import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi, BM25Plus
import string
import re
import nltk
from nltk.corpus import stopwords
import datetime

# Download stopwords
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

# --- Preprocessing ---
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

def clean_tanggal(text):
    if isinstance(text, str):
        text = re.sub(r'^[A-Za-z]+,\s*', '', text)
        text = text.replace('WIB', '').strip()
        return text
    return text

@st.cache_data

def load_data():
    df = pd.read_excel("berita_politik.xlsx")
    df['text'] = df['judul'].astype(str) + " " + df['isi'].astype(str)
    df['clean_text'] = df['text'].apply(preprocess)
    df['tokens'] = df['text'].apply(tokenize)
    if 'tanggal' in df.columns:
        df['tanggal'] = df['tanggal'].apply(clean_tanggal)
        df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d %b %Y %H:%M', errors='coerce')
    return df

# --- Search Engine Setup ---
df = load_data()
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])
bm25 = BM25Okapi(df['tokens'].tolist())
bm25_plus = BM25Plus(df['tokens'].tolist())

# --- Query List ---
query_list = [
    "pemilu 2024",
    "kampanye presiden",
    "partai golkar",
    "zulkifli hasan",
    "politik identitas",
    "dukungan NU dalam pemilu",
    "psi dan anak muda",
    "prabowo gibran debat",
    "politik dan agama",
    "caleg korupsi"
]

# --- Streamlit App ---
st.title("🧪 Evaluasi Presisi Search Engine Berita Politik")

selected_query = st.selectbox("Pilih Query Uji", query_list)
method = st.radio("Pilih Metode Similarity", ["Cosine Similarity", "BM25", "BM25+"])

# Search Function
def search(query, method):
    query_clean = preprocess(query)
    if method == "Cosine Similarity":
        query_vec = tfidf_vectorizer.transform([query_clean])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    elif method == "BM25":
        query_tokens = tokenize(query)
        similarity = bm25.get_scores(query_tokens)
    else:
        query_tokens = tokenize(query)
        similarity = bm25_plus.get_scores(query_tokens)

    top_indices = similarity.argsort()[-10:][::-1]
    results = df.iloc[top_indices].copy()
    results['score'] = similarity[top_indices]
    return results

# Run Search and Evaluation
if st.button("Tampilkan Hasil & Evaluasi"):
    results = search(selected_query, method)
    st.subheader("📋 Hasil Pencarian dan Evaluasi")

    relevance_list = []
    for i, row in results.iterrows():
        st.markdown(f"### {row['judul']}")
        if pd.notna(row.get('tanggal')):
            st.caption(f"📅 {row['tanggal'].strftime('%d %B %Y %H:%M')}")
        st.write(row['isi'][:300] + "...")
        st.caption(f"🔍 Skor Kemiripan: {row['score']:.4f}")

        relevant = st.radio(f"Apakah hasil ke-{i} relevan?", ["Belum Dinilai", "Relevan", "Tidak Relevan"], key=f"rel_{i}")
        relevance_list.append({
            "query": selected_query,
            "metode": method,
            "judul": row['judul'],
            "skor": row['score'],
            "relevan": 1 if relevant == "Relevan" else (0 if relevant == "Tidak Relevan" else None)
        })
        st.markdown("---")

    if st.button("💾 Simpan Hasil Evaluasi"):
        df_eval = pd.DataFrame(relevance_list)
        df_eval.dropna(subset=['relevan'], inplace=True)
        presisi = df_eval['relevan'].mean() if not df_eval.empty else 0.0
        filename = f"evaluasi_presisi_{method.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_eval.to_csv(filename, index=False)
        st.success(f"Hasil evaluasi disimpan sebagai {filename}.")
        st.info(f"Presisi: {presisi:.2f} ({int(presisi*100)}%)")
