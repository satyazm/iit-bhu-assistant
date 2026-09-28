"""Loads PDFs from data/documents/<category>/, chunks them, embeds them with
Gemini, and stores them in a local Chroma collection."""

import sys

import chromadb
import google.generativeai as genai
from pypdf import PdfReader
from tqdm import tqdm

from config import (
    CATEGORIES,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
)

def read_pdf_text(pdf_path):
    """pdf_path can be a filesystem path or a file-like object (e.g. an uploaded file)."""
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_number, text))
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_batch(texts):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"]


def collect_chunks(pdf_path, category, source_name):
    """Reads a PDF (path or file-like) and returns (ids, texts, metadatas) for its chunks."""
    ids, texts, metadatas = [], [], []
    for page_number, page_text in read_pdf_text(pdf_path):
        for chunk_index, chunk in enumerate(chunk_text(page_text)):
            doc_id = f"{category}/{source_name}/p{page_number}/c{chunk_index}"
            ids.append(doc_id)
            texts.append(chunk)
            metadatas.append(
                {"category": category, "source": source_name, "page": page_number}
            )
    return ids, texts, metadatas


def index_pdf(collection, pdf_path, category, source_name):
    """Chunks, embeds, and upserts a single PDF into the collection. Returns chunk count."""
    ids, texts, metadatas = collect_chunks(pdf_path, category, source_name)
    if not texts:
        return 0

    batch_size = 50
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        batch_meta = metadatas[start : start + batch_size]
        embeddings = embed_batch(batch_texts)
        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_meta,
            embeddings=embeddings,
        )
    return len(texts)


def build_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("iit_bhu_docs")

    total_chunks = 0
    total_files = 0

    for category in CATEGORIES:
        category_dir = DOCUMENTS_DIR / category
        if not category_dir.exists():
            continue
        pdf_files = sorted(category_dir.glob("*.pdf"))
        for pdf_path in tqdm(pdf_files, desc=f"Indexing {category}"):
            count = index_pdf(collection, pdf_path, category, pdf_path.name)
            if count:
                total_chunks += count
                total_files += 1

    if total_files == 0:
        print("No PDFs found under data/documents/<category>/. Add some and rerun.")
        return

    print(f"Indexed {total_chunks} chunks from {total_files} PDF(s).")


if __name__ == "__main__":
    if not GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    genai.configure(api_key=GEMINI_API_KEY)
    build_collection()
