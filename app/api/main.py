from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.rag_service import answer_policy_question

app = FastAPI(
    title="PaySafe Policy & Compliance RAG Copilot API",
    description="Enterprise RAG service powered by PostgreSQL pgvector and Groq LLM.",
    version="1.0.0"
)

class PolicyQuestion(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "paysafe-policy-rag-copilot"}

@app.post("/ask-policy")
def ask_policy(payload: PolicyQuestion):
    """Answers employee policy questions with strict citations and anti-hallucination guardrails."""
    try:
        result = answer_policy_question(payload.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
