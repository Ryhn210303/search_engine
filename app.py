import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.corpus import stopwords

# Cek stopwords
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel('berita_politik.xlsx')
    df['text'] = df['judul'].astype(str) + " " + df['isi'].astype(str)
    df['clean_text'] = df['text'].apply(preprocess)
    return df

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('indonesian'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load and process TF-IDF
df = load_data()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

# Search function
def search_news(query, k=5):
    query_clean = preprocess(query)
    query_vec = vectorizer.transform([query_clean])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[-k:][::-1]
    results = df.iloc[related_docs_indices][['judul', 'isi', 'link']]
    scores = cosine_similarities[related_docs_indices]
    results = results.assign(score=scores)
    return results

# Streamlit App
st.title("Search Engine Berita Politik")

query = st.text_input("Masukkan kata kunci (misal: pemilu presiden)", "")

k = st.slider("Jumlah hasil berita yang ditampilkan", 1, 10, 5)

if query:
    with st.spinner("Mencari berita..."):
        results = search_news(query, k=k)
        st.success(f"Ditemukan {len(results)} berita relevan:")
        for i, row in results.iterrows():
            st.markdown(f"### [{row['judul']}]({row['link']})")
            st.write(row['isi'][:300] + "...")
            st.caption(f"Skor Kemiripan: {row['score']:.4f}")
            st.markdown("---")
