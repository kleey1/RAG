import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#### INDEXING ####

def load_documents():
    """Load all text files from the docs directory"""

    documents = []

    urls = [
        "https://cms.epoka.edu.al/api/CMS_Archive/assets/Reg-Undergraduate-studies-and-examinations-pdf-361",
        "https://cms.epoka.edu.al/api/CMS_Archive/assets/Regulation-On-Student-Discipline-pdf-361"
    ]

    print("Loading remote PDFs...")

    for url in urls:
        try:
            loader = PyPDFLoader(url)
            pdf_docs = loader.load()

            # add metadata
            for doc in pdf_docs:
                doc.metadata["source"] = url

            documents.extend(pdf_docs)

        except Exception as e:
            print(f"Failed to load {url}: {e}")

    if len(documents) == 0:
        raise ValueError("No documents loaded!")

    print(f"\nTotal documents loaded: {len(documents)}")

    for i, doc in enumerate(documents[0:2]):
        print(f"\nDocument {i+1}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Preview: {doc.page_content[:150]}...")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
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

def create_vector_store(chunks, embedding_model, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    documents = load_documents()
    chunks = split_documents(documents)

    persistent_directory = "db/chroma_db"

    if os.path.exists(persistent_directory):
      shutil.rmtree(persistent_directory)

    vectorstore = create_vector_store(chunks, embedding_model, persistent_directory)

    print("\nIngestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()
    
    