# config.py
PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # or whichever you use
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
DOCS_PATH = "docs"