from ingestion_pipeline import main as ingest
from answer_generation import ask

if __name__ == "__main__":
    ingest()  # no-op if DB already exists

    while True:
        query = input("\nAsk a question (or 'quit'): ")
        if query.lower() == "quit":
            break
        print("\n" + ask(query))