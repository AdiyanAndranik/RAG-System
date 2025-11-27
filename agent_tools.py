import requests
from typing import List

BASE_URL = "http://127.0.0.1:8000"

def tool_retrieve(text: str, k: int = 3) -> dict:
    resp = requests.post(
        f"{BASE_URL}/rag_query",
        json={"text": text}
    )
    return resp.json()

def tool_generate_answer(prompt: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/generate",
        json={"text": prompt}
    )
    return resp.json()

def tool_n8n_webhook(workflow_name: str, payload: dict):
    url = f"http://localhost:5678/webhook/{workflow_name}"
    resp = requests.post(url, json=payload)
    return resp.json() if resp.text else {"status": "sent"}