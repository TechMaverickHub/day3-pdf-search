from pathlib import Path
import json

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "sample.pdf"
INDEX_DIR = BASE_DIR / "indexes"

INDEX_DIR.mkdir(exist_ok=True)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def main():
    print("Loading PDF...")

    text = extract_text(PDF_PATH)

    print(f"Extracted {len(text)} characters")

    chunks = chunk_text(text)

    print(f"Generated {len(chunks)} chunks")

    print("Loading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        str(INDEX_DIR / "pdf.index")
    )

    with open(
        INDEX_DIR / "chunks.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nIndex saved successfully")
    print(f"Chunks saved: {len(chunks)}")


if __name__ == "__main__":
    main()