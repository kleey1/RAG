from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import PERSIST_DIR, EMBEDDING_MODEL, TOP_K

# Laod DB and Search

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return db.as_retriever(search_kwargs={"k": TOP_K})

def retrieve(query: str):
    retriever = get_retriever()
    return retriever.invoke(query)