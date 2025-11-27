from agent_tools import tool_retrieve, tool_generate_answer, tool_n8n_webhook

def decide_action(query: str):
    q = query.lower()

    if any(w in q for w in ["reset", "forgot password", "change password"]):
        return "retrieve_answer"

    if any(w in q for w in ["refund", "delivery", "order"]):
        return "retrieve_answer"

    if any(w in q for w in ["integration", "api", "b2b"]):
        return "retrieve_answer"

    if "start workflow" in q or "trigger" in q:
        return "workflow"

    return "chat"
    

def agent(query: str):
    action = decide_action(query)

    if action == "retrieve_answer":

        retrieved = tool_retrieve(query)
        context_docs = retrieved["retrieved"]
        prompt = f"Context:\n{context_docs}\n\nQuestion: {query}\nAnswer clearly:"
        answer = tool_generate_answer(prompt)
        return {
            "action": "rag_answer",
            "answer": answer,
            "sources": context_docs
        }

    if action == "workflow":

        workflow = "example_workflow"
        res = tool_n8n_webhook(workflow, {"query": query})
        return {
            "action": "n8n_workflow",
            "workflow": workflow,
            "result": res
        }

    answer = tool_generate_answer(query)
    return {
        "action": "llm_chat",
        "answer": answer
    }