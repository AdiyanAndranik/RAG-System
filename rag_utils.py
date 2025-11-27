import os
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")

def chunk_text(text, chunk_size=500, overlap=50):
    """Simple sliding-window chunker (characters)."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks

class RAG:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(name="knowledge_base")

    def embed(self, texts):
        """Return normalized embeddings list for given list of texts."""
        embs = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
        return embs

    def upsert_documents(self, docs: list):
        """
        docs: list of dicts with keys 'id' and 'text' and optional 'metadata'
        """
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        embeddings = self.embed(texts)
        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def retrieve(self, query, k=3):
        q_emb = self.model.encode([query], normalize_embeddings=True).tolist()[0]

        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            retrieved.append({
                "text": doc,
                "metadata": meta,
                "distance": dist
            })

        return retrieved

def build_rag_prompt(user_query, docs):
    context = "\n\n---\n\n".join([f"Doc {i+1}: {d}" for i, d in enumerate(docs)])
    prompt = f"""
You are a helpful assistant that answers user questions using only the provided context. 
If the context does not contain the answer, say "I don't know, please provide more details."

CONTEXT:
{context}

USER QUESTION:
{user_query}

Return a short answer in plain text.
"""
    return prompt