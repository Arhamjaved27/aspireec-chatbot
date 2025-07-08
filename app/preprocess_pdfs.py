import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

PDF_DIR = os.path.join(os.path.dirname(__file__), '..', 'Aspire Data')
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'cache', 'faiss_index.bin')
META_PATH = os.path.join(os.path.dirname(__file__), '..', 'cache', 'faiss_meta.pkl')
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # characters of overlap
EMBED_MODEL = 'all-MiniLM-L6-v2'


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # If the text contains many newlines, treat as a list and split by line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 10:  # Heuristic: treat as list if >10 lines
        return lines
    # Otherwise, use overlapping chunking
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start += chunk_size - overlap
    return chunks


def main():
    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = []
    meta = []
    total_chars = 0
    for fname in os.listdir(PDF_DIR):
        if fname.lower().endswith('.pdf'):
            pdf_path = os.path.join(PDF_DIR, fname)
            print(f"Processing {fname}...")
            text = extract_text_from_pdf(pdf_path)
            total_chars += len(text)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            meta.extend([{'source': fname, 'chunk_id': i, 'text': chunk} for i, chunk in enumerate(chunks)])
    print(f"Total characters extracted: {total_chars}")
    print(f"Total chunks: {len(all_chunks)}")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, 'wb') as f:
        pickle.dump(meta, f)
    print(f"FAISS index and metadata saved to cache/.")

if __name__ == "__main__":
    main()
