# 🛡️ PaySafe India: Enterprise Policy & Compliance RAG Copilot

A pure backend Forward-Deployed Engineer (FDE) Retrieval-Augmented Generation (RAG) API built to answer employee policy questions strictly from company handbooks with exact citations and zero hallucinations.

## 🏗️ Architecture & Data Flow
```text
[Policy Handbook (.txt)] ──▶ [Chunking + FastEmbed Vectors] ──▶ [PostgreSQL + pgvector]
                                                                          │
[Employee Query (JSON)] ──▶ [FastAPI /ask-policy] ──▶ [Top-K Vector Search] ──▶ [Guardrailed LLM + Citations]
```

## 🛠️ Tech Stack
* **API Framework:** Python, FastAPI, Uvicorn, Pydantic
* **Vector Database:** PostgreSQL with `pgvector` extension (Neon Cloud)
* **Embeddings:** `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dimensions)
* **LLM & Guardrails:** Groq API (`openai/gpt-oss-120b`, `temperature=0.0`, strict JSON grounding)
* **Containerization:** Docker

## 🚀 Quick Start
1. Add `DATABASE_URL` and `GROQ_API_KEY` to `.env`.
2. Ingest policy documents into `pgvector`:
   ```bash
   python -c "from app.database.vector_db import ingest_policy_file; ingest_policy_file()"
   ```
3. Start the API server:
   ```bash
   uvicorn app.api.main:app --host 0.0.0.0 --port 8000
   ```
