import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from config import PERSIST_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, DOCS_PATH

load_dotenv()

#### INDEXING ####

def load_documents():
    """Load all text files from the docs directory"""

    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"The directory {DOCS_PATH} does not exist. ")
    
    # Load all .txt files from the docs directory
    loader = PyPDFDirectoryLoader(path=DOCS_PATH)
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {DOCS_PATH}. ")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def create_vector_store(chunks):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
    
    embeddings = get_embedding_model()

    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {PERSIST_DIR}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")

    documents = load_documents()
    chunks = split_documents(documents)

    if os.path.exists(PERSIST_DIR):
      shutil.rmtree(PERSIST_DIR)

    vectorstore = create_vector_store(chunks)

    print("\nIngestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()
    
    