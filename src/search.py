from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = BASE_DIR / "indexes" / "pdf.index"
CHUNKS_PATH = BASE_DIR / "indexes" / "chunks.json"

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_PATH)
)

with open(
    CHUNKS_PATH,
    encoding="utf-8"
) as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")


while True:
    query = input("\nAsk a question (type exit to quit): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        3
    )

    print("\nTop Results:\n")

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):
        print(f"Result #{rank}")
        print(
            f"Distance: {distances[0][rank-1]:.4f}"
        )
        print("-" * 60)
        print(chunks[idx][:500])
        print("\n")