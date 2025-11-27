import os
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")

docs = [
    {"id":"d1", "text":"How to reset my password? Go to settings -> account -> reset password."},
    {"id":"d2", "text":"Delivery time is 3-5 business days for domestic orders."},
    {"id":"d3", "text":"Refunds are processed within 7 days after approval."},
    {"id":"d4", "text":"We offer B2B integrations via our API. Contact sales for API keys and onboarding."},
    {"id":"d5", "text":"To connect via OAuth follow these steps: register app, set callback URL, exchange code for token."}
]

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("knowledge_base")

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source":"faq"} for _ in texts]
    )

    print("Inserted docs into ChromaDB:", len(texts))

if __name__ == "__main__":
    main()
