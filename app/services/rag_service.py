import json
from groq import Groq
from app.config import settings
from app.database.vector_db import search_policy_chunks

client = Groq(api_key=settings.groq_api_key)

def answer_policy_question(question: str) -> dict:
    """Retrieves relevant policy chunks and generates a strictly grounded answer."""
    # 1. RETRIEVE: Search PostgreSQL pgvector for the top 2 closest policy paragraphs
    relevant_chunks = search_policy_chunks(question, top_k=2)
    context_text = "\n\n".join(relevant_chunks)

    # 2. AUGMENT & GUARDRAIL: Force the AI to answer ONLY from the retrieved text
    system_prompt = """
    You are PaySafe India's strict HR & Compliance Policy Assistant.
    Answer the employee's question using ONLY the Policy Context provided below.
    
    STRICT RULES:
    1. If the answer is NOT explicitly written in the Policy Context, set "grounded" to false and set "answer" to: "I cannot answer this based on PaySafe's official policy documents."
    2. Never use outside knowledge or guess numbers.
    3. Always list the exact section header(s) you used inside "sources".
    
    Return ONLY valid JSON with these exact 3 keys:
    - "answer" (string)
    - "sources" (list of strings, e.g., ["Internet & WFH Policy - Section 2"])
    - "grounded" (boolean: true if answered from context, false if not found)
    """

    user_message = f"Policy Context:\n{context_text}\n\nEmployee Question: {question}"

    # 3. GENERATE: Call Groq LLM with temperature=0.0 for zero creativity
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    return json.loads(response.choices.message.content)
