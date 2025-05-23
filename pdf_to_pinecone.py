import os
import sys
import platform

# Handle Windows-specific readline issue
if platform.system() == 'Windows':
    try:
        import readline
    except ImportError:
        # Create a dummy readline module for Windows
        class DummyReadline:
            def __init__(self):
                pass
            def get_completer(self):
                return None
            def set_completer(self, *args):
                pass
        sys.modules['readline'] = DummyReadline()

import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from typing import List
from datetime import datetime
import time

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'spiritual')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone v2
pc = Pinecone(api_key=PINECONE_API_KEY)

# Constants
CHUNK_SIZE = 1000  # tokens
OVERLAP = 200      # tokens
EMBEDDING_MODEL = "text-embedding-ada-002"

# Tokenizer
def get_token_count(text: str) -> int:
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    return len(encoding.encode(text))

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> List[str]:
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str) -> List[float]:
    # OpenAI API rate limits: add retry logic
    for attempt in range(5):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}. Retrying in 5s...")
            time.sleep(5)
    raise RuntimeError("Failed to get embedding after retries.")

def process_pdf(pdf_path: str, index_name: str = PINECONE_INDEX):
    # Create or get Pinecone index
    index_names = [idx.name for idx in pc.list_indexes()]
    if index_name not in index_names:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
    index = pc.Index(index_name)

    # Read PDF (page by page for memory efficiency)
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        all_chunks = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if not text:
                continue
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append((page_num, i, chunk))

    print(f"Total chunks to embed: {len(all_chunks)}")
    # Batch upsert to Pinecone
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        vectors = []
        for page_num, chunk_idx, chunk in batch:
            embedding = get_embedding(chunk)
            meta = {
                "source": os.path.basename(pdf_path),
                "page": page_num,
                "chunk": chunk_idx,
                "timestamp": datetime.now().isoformat(),
            }
            vectors.append({
                "id": f"{os.path.basename(pdf_path)}-p{page_num}-c{chunk_idx}",
                "values": embedding,
                "metadata": meta
            })
        index.upsert(vectors=vectors)
        print(f"Upserted {i+len(batch)}/{len(all_chunks)} chunks...")
    print(f"Done! Stored {len(all_chunks)} chunks from {pdf_path} in Pinecone.")

def main():
    pdf_dir = "pdfs"
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            print(f"Processing {filename}...")
            process_pdf(os.path.join(pdf_dir, filename))

if __name__ == "__main__":
    main() 