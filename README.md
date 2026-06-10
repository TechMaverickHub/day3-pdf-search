# PDF Semantic Search with FAISS

A semantic search application that allows users to search PDF documents using embeddings and vector search.

This project extracts text from PDFs, splits the content into chunks, generates embeddings using Sentence Transformers, stores them in a FAISS vector index, and retrieves the most relevant chunks based on semantic meaning.

This is the first step toward building a complete Retrieval-Augmented Generation (RAG) system.

---

## Project Goal

Traditional PDF search relies on exact keyword matching.

Example:

### Query

```text
database optimization
```

Traditional search may fail if the PDF contains:

```text
Use PostgreSQL indexes and query optimization techniques
```

Semantic search understands the meaning behind the query and retrieves relevant content even when exact keywords are missing.

---

## Demo Pipeline

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS Index
 ↓
Semantic Search
```

---

## Features

- PDF text extraction
- Document chunking
- Embedding generation
- Vector indexing with FAISS
- Semantic retrieval
- Interactive CLI search
- Streamlit web interface
- UV-based dependency management

---

## Tech Stack

- Python 3.12+
- UV
- PyPDF
- Sentence Transformers
- FAISS
- NumPy
- Streamlit

---

## Project Structure

```text
day3-pdf-search/
│
├── data/
│   └── sample.pdf
│
├── indexes/
│   ├── pdf.index
│   └── chunks.json
│
├── src/
│   ├── ingest.py
│   ├── search.py
│   └── streamlit_app.py
│
├── screenshots/
│
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd day3-pdf-search
```

Install dependencies:

```bash
uv sync
```

---

## Build Vector Index

Extract text, chunk content, generate embeddings, and create a FAISS index:

```bash
uv run src/ingest.py
```

Expected Output:

```text
Loading PDF...
Extracted 12500 characters

Generated 32 chunks

Index saved successfully

Chunks saved: 32
```

---

## Run CLI Search

```bash
uv run src/search.py
```

Example:

```text
Ask a question:
database indexing
```

Output:

```text
Result #1

Distance: 0.8123

PostgreSQL indexes improve query performance by reducing
the amount of data scanned during execution...
```

---

## Run Streamlit UI

```bash
uv run streamlit run src/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## How It Works

### Step 1: PDF Text Extraction

Text is extracted from every page using PyPDF.

```python
reader = PdfReader(pdf_path)
```

---

### Step 2: Chunking

Large documents are split into smaller chunks.

Example:

```text
Chunk 1
Django ORM Optimization...
```

```text
Chunk 2
Database Indexing...
```

```text
Chunk 3
Caching Strategies...
```

Chunking improves retrieval quality and reduces irrelevant context.

---

### Step 3: Generate Embeddings

Each chunk is converted into a dense vector representation.

```python
model.encode(chunks)
```

---

### Step 4: Create Vector Index

Embeddings are stored inside a FAISS vector database.

```python
index = faiss.IndexFlatL2(dimension)
```

---

### Step 5: Semantic Retrieval

User queries are converted into embeddings and compared against stored vectors.

```python
distances, indices = index.search(
    query_embedding,
    3
)
```

---

## Understanding Distance Scores

This project uses:

```python
faiss.IndexFlatL2()
```

FAISS returns Euclidean (L2) distances.

Example:

```text
Distance: 0.81
Distance: 1.22
Distance: 1.46
```

Interpretation:

```text
Lower Distance = Better Match
Higher Distance = Less Similar
```

The chunk with the smallest distance is considered the most relevant result.

---

## Example Searches

### Query

```text
database optimization
```

### Results

```text
Distance: 0.81

PostgreSQL indexes improve query performance...
```

---

### Query

```text
django orm
```

### Results

```text
Distance: 0.72

Use select_related and prefetch_related to reduce
database queries...
```

---

### Query

```text
caching
```

### Results

```text
Distance: 0.93

Redis can be used to cache expensive queries...
```

---

## Why Chunk Documents?

Large PDFs cannot be efficiently searched as a single block of text.

Instead:

```text
PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
Search
```

Benefits:

- Better retrieval accuracy
- Reduced noise
- Faster search
- Foundation for RAG systems

---

## Interview Questions

### What is chunking?

Chunking is the process of splitting large documents into smaller sections before generating embeddings.

---

### Why do we chunk documents?

Smaller chunks improve retrieval precision and reduce irrelevant context.

---

### What is FAISS?

FAISS is a vector similarity search library developed by Meta that enables efficient nearest-neighbor search.

---

### What do the returned scores represent?

The scores returned by FAISS IndexFlatL2 are Euclidean distances.

Lower distance means higher similarity.

---

### How does this relate to RAG?

Semantic retrieval is the retrieval layer of a RAG pipeline.

Retrieved chunks are later passed to an LLM as context for answer generation.

---

## What I Learned

### Day 1

- Embeddings
- Cosine Similarity
- Semantic Search

### Day 2

- FAISS
- Vector Search
- Distance-Based Retrieval

### Day 3

- PDF Processing
- Chunking
- Document Embeddings
- Semantic Retrieval
- Knowledge Base Search

---

## Future Improvements

- PDF Upload Support
- Multiple PDF Search
- Metadata Filtering
- Cosine Similarity Search
- RAG Chatbot
- FastAPI Backend
- Docker Deployment
- Qdrant Integration
- Production Monitoring

---

## Learning Roadmap

- [x] Embeddings
- [x] Semantic Search
- [x] FAISS Vector Search
- [x] PDF Processing
- [x] Chunking
- [ ] RAG Pipeline
- [ ] FastAPI API
- [ ] Docker Deployment
- [ ] Production AI System

---

## Author

**Abhiroop Bhattacharyya**

Backend Engineer → AI Engineer

### Skills

- Python
- Django
- FastAPI
- Generative AI
- RAG
- FAISS
- Vector Databases
- AI Applications

---

## Connect

Building in public while transitioning from Backend Engineering to AI Engineering.

⭐ If you found this project useful, consider giving it a star.