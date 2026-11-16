import psycopg2
from pgvector.psycopg2 import register_vector
from fastembed import TextEmbedding
from app.config import settings

# Load a tiny, fast, free embedding model (turns text into 384 meaning numbers)
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_vector_conn():
    """Connects to Neon PostgreSQL and enables pgvector support."""
    conn = psycopg2.connect(settings.database_url)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cur.close()
    register_vector(conn)
    return conn

def ingest_policy_file(file_path: str = "data/paysafe_policy.txt"):
    """Reads the policy file, chunks it by paragraph, creates vectors, and saves to SQL."""
    conn = get_vector_conn()
    cur = conn.cursor()

    # 1. Create the vector table (384 matches the size of our embedding model)
    cur.execute("DROP TABLE IF EXISTS policy_chunks;")
    cur.execute("""
        CREATE TABLE policy_chunks (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(384)
        );
    """)

    # 2. Chunking: Split the document by double newlines (each paragraph = 1 chunk)
    with open(file_path, "r") as f:
        chunks = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    # 3. Convert each chunk into meaning numbers (vectors) and insert into PostgreSQL
    embeddings = list(embed_model.embed(chunks))
    for chunk_text, vector in zip(chunks, embeddings):
        cur.execute(
            "INSERT INTO policy_chunks (content, embedding) VALUES (%s, %s);",
            (chunk_text, vector.tolist())
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"SUCCESS: Ingested {len(chunks)} policy chunks into PostgreSQL pgvector!")

def search_policy_chunks(question: str, top_k: int = 2) -> list[str]:
    """Finds the top_k most relevant policy chunks by meaning (Vector Search)."""
    conn = get_vector_conn()
    cur = conn.cursor()

    # Convert the user's question into meaning numbers
    question_vector = list(embed_model.embed([question]))[0].tolist()

    # <-> is pgvector's distance operator: smallest distance = closest meaning!
    cur.execute("""
        SELECT content
        FROM policy_chunks
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """, (question_vector, top_k))

    results = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results
