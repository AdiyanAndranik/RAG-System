import os
import uuid
import json
import requests
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from rag_utils import RAG, build_rag_prompt, chunk_text

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:latest")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")

app = FastAPI()
rag = RAG()

class Query(BaseModel):
    text: str
    top_k: Optional[int] = 3


def call_llm(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=60
    )

    text = response.text.strip()
    first_line = text.split("\n")[0]
    try:
        data = json.loads(first_line)
        return data.get("response", "").strip()
    except Exception as e:
        print("Ollama raw response:", text)
        raise RuntimeError(f"Failed to parse Ollama response: {e}")


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...), namespace: Optional[str] = Form("knowledge")):
    contents = await file.read()
    try:
        import fitz
    except Exception as e:
        return {"error": "PyMuPDF not installed", "detail": str(e)}

    tmp_path = f"tmp_{uuid.uuid4().hex}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    doc = fitz.open(tmp_path)
    full_text = []
    for page in doc:
        txt = page.get_text("text")
        if txt:
            full_text.append(txt)
    doc.close()
    os.remove(tmp_path)

    text = "\n".join(full_text).strip()
    if not text:
        return {"error": "No text extracted from PDF"}

    chunks = chunk_text(text, chunk_size=800, overlap=120)
    docs = []
    for idx, chunk in enumerate(chunks):
        doc_id = f"{namespace}_{uuid.uuid4().hex}_{idx}"
        docs.append({"id": doc_id, "text": chunk, "metadata": {"source_file": file.filename, "namespace": namespace}})

    rag.upsert_documents(docs)

    return {"message": "ingested", "collection": COLLECTION_NAME, "chunks": len(docs)}


@app.post("/rag_query")
def rag_query(q: Query):
    """Retrieve relevant documents from uploaded files"""
    results = rag.retrieve(q.text, k=q.top_k or 3)
    return {"retrieved": results}


@app.post("/generate")
def generate(q: Query):
    """Pure LLM generation without RAG - for general questions"""
    prompt = f"""You are a helpful assistant. Answer the following question clearly and concisely.

Question: {q.text}

Answer:"""
    
    answer = call_llm(prompt)
    return {"answer": answer, "sources": []}


@app.post("/agent")
def agent_route(q: Query):
    """
    Smart agent that decides:
    - If question is relevant to uploaded documents -> use RAG
    - If question is general knowledge -> use pure LLM
    """
    user_text = q.text
    
    retrieved = rag.retrieve(user_text, k=3)
    

    is_relevant = any(doc.get("distance", 999) < 0.5 for doc in retrieved)
    
    if is_relevant:

        docs_text = [doc["text"] for doc in retrieved]
        prompt = build_rag_prompt(user_text, docs_text)
        answer = call_llm(prompt)
        
        return {
            "action": "rag_answer",
            "answer": answer,
            "sources": retrieved,
            "reason": "Answer based on uploaded documents"
        }
    else:
        prompt = f"""You are a helpful assistant. Answer the following question clearly and concisely.

Question: {user_text}

Answer:"""
        
        answer = call_llm(prompt)
        
        return {
            "action": "llm_answer",
            "answer": answer,
            "sources": [],
            "reason": "No relevant documents found, using general knowledge"
        }