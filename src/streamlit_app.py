from pathlib import Path
import json

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = BASE_DIR / "indexes" / "pdf.index"
CHUNKS_PATH = BASE_DIR / "indexes" / "chunks.json"


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_index():
    return faiss.read_index(
        str(INDEX_PATH)
    )


@st.cache_data
def load_chunks():
    with open(
        CHUNKS_PATH,
        encoding="utf-8"
    ) as f:
        return json.load(f)


st.set_page_config(
    page_title="PDF Semantic Search",
    layout="wide"
)

st.title("PDF Semantic Search")
st.write(
    "Search PDF content using embeddings and FAISS"
)

model = load_model()
index = load_index()
chunks = load_chunks()

query = st.text_input(
    "Ask a question"
)

if query:
    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        5
    )

    st.subheader("Results")

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):
        with st.expander(
            f"Result #{rank} | Distance: {distances[0][rank-1]:.4f}"
        ):
            st.write(chunks[idx])