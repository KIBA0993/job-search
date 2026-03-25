"""
Phase 4: Q&A store management — CRUD, similarity matching, answer tailoring.

Q&A entries live in the qa_store SQLite table.  Embeddings are computed
via Mammoth AI (OpenAI-compatible) and serialized as float vectors in the
embedding BLOB column.  Cosine similarity is used to find the best match
for a new question, with a keyword-overlap fallback when no client is
available.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SKILL_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two float vectors."""
    try:
        import numpy as np

        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    except ImportError:
        # Pure-Python fallback — slower but no numpy required
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        denom = norm_a * norm_b
        return dot / denom if denom else 0.0


def _get_embedding(
    text: str,
    client,
    model: str = "text-embedding-3-small",
) -> Optional[list]:
    """Fetch embedding vector from Mammoth AI (OpenAI-compatible)."""
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as exc:
        logger.warning("Embedding request failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core Q&A operations
# ---------------------------------------------------------------------------

def find_best_match(
    question: str,
    db_path=None,
    client=None,
    embedding_model: str = "text-embedding-3-small",
    threshold: float = 0.80,
) -> Optional[dict]:
    """
    Find the best-matching stored Q&A entry for a question.

    1. If client is provided, embed the question and compare cosine similarity
       against all stored embeddings that have been previously computed.
    2. Fall back to keyword-overlap ratio if no client or no embeddings exist.

    Returns a dict with keys id, question_pattern, answer, source, similarity
    if a match above threshold is found; else None.
    """
    import db as _db

    if db_path is None:
        db_path = _db.DB_PATH

    with _db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, question_pattern, answer, source, embedding "
            "FROM qa_store ORDER BY used_count DESC"
        ).fetchall()

    if not rows:
        return None

    # --- Embedding-based similarity ---
    if client:
        q_vec = _get_embedding(question.lower(), client, embedding_model)
        if q_vec:
            best_score = -1.0
            best_row = None
            for row in rows:
                if not row["embedding"]:
                    continue
                try:
                    stored_vec = pickle.loads(row["embedding"])
                    sim = _cosine_similarity(q_vec, stored_vec)
                    if sim > best_score:
                        best_score = sim
                        best_row = row
                except Exception:
                    continue
            if best_row and best_score >= threshold:
                return {
                    "id": best_row["id"],
                    "question_pattern": best_row["question_pattern"],
                    "answer": best_row["answer"],
                    "source": best_row["source"],
                    "similarity": best_score,
                }

    # --- Keyword-overlap fallback ---
    q_words = set(question.lower().split())
    best_overlap = 0.0
    best_row = None
    for row in rows:
        p_words = set(row["question_pattern"].split())
        if not p_words:
            continue
        overlap = len(q_words & p_words) / len(p_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_row = row

    keyword_threshold = 0.50
    if best_row and best_overlap >= keyword_threshold:
        return {
            "id": best_row["id"],
            "question_pattern": best_row["question_pattern"],
            "answer": best_row["answer"],
            "source": best_row["source"],
            "similarity": best_overlap,
        }

    return None


def tailor_answer(
    base_answer: str,
    target_question: str,
    client=None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Adapt a stored answer's phrasing to the specific question wording.

    Factual content is kept identical; only phrasing is adjusted.
    Returns the original answer if client is None or tailoring fails.
    """
    if not client or not base_answer.strip():
        return base_answer
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Adapt the following answer to fit the specific question below. "
                        "Keep all factual content identical. Only adjust phrasing. "
                        "Use only commas and periods. No dashes, asterisks, or bullet points.\n\n"
                        f"Question: {target_question}\n"
                        f"Answer: {base_answer}\n\n"
                        "Return only the adapted answer, nothing else."
                    ),
                }
            ],
            max_tokens=300,
            temperature=0.2,
        )
        adapted = response.choices[0].message.content.strip()
        return adapted if adapted else base_answer
    except Exception as exc:
        logger.warning("Answer tailoring failed: %s", exc)
        return base_answer


def add_answer(
    question: str,
    answer: str,
    source: str = "user_provided",
    client=None,
    embedding_model: str = "text-embedding-3-small",
    db_path=None,
) -> int:
    """
    Store a Q&A entry in qa_store, computing an embedding if client is provided.
    Uses ON CONFLICT to upsert — updating answer, source, and embedding.

    Returns the row id.
    """
    import db as _db

    if db_path is None:
        db_path = _db.DB_PATH

    embedding_blob = None
    if client:
        vec = _get_embedding(question.lower(), client, embedding_model)
        if vec:
            embedding_blob = pickle.dumps(vec)

    with _db.get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO qa_store (question_pattern, answer, source, embedding) "
            "VALUES (lower(?), ?, ?, ?) "
            "ON CONFLICT(question_pattern) DO UPDATE SET "
            "answer=excluded.answer, source=excluded.source, "
            "embedding=COALESCE(excluded.embedding, embedding)",
            (question, answer, source, embedding_blob),
        )
        return cur.lastrowid


def update_used_count(qa_id: int, db_path=None) -> None:
    """Increment used_count and record last_used date for a Q&A entry."""
    import db as _db

    if db_path is None:
        db_path = _db.DB_PATH

    with _db.get_connection(db_path) as conn:
        conn.execute(
            "UPDATE qa_store "
            "SET used_count = used_count + 1, last_used = date('now') "
            "WHERE id = ?",
            (qa_id,),
        )


def requeue_jobs_for_question(qa_id: int, db_path=None) -> int:
    """
    When a new Q&A answer is stored, re-queue any jobs that were blocked
    by that unanswered question.

    Marks the relevant pending_questions as answered, then sets the job
    status back to 'scoring' if all its pending questions are now answered.

    Returns the count of re-queued jobs.
    """
    import db as _db

    if db_path is None:
        db_path = _db.DB_PATH

    with _db.get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT j.id
            FROM pending_questions pq
            JOIN jobs j ON j.id = pq.job_id
            WHERE pq.matched_qa_id = ? AND pq.answered = 0
              AND j.status = 'queued'
            """,
            (qa_id,),
        ).fetchall()

        job_ids = [r["id"] for r in rows]
        if not job_ids:
            return 0

        conn.execute(
            "UPDATE pending_questions "
            "SET answered = 1, answered_at = datetime('now') "
            "WHERE matched_qa_id = ? AND answered = 0",
            (qa_id,),
        )

        requeued = 0
        for jid in job_ids:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM pending_questions "
                "WHERE job_id = ? AND answered = 0",
                (jid,),
            ).fetchone()[0]
            if remaining == 0:
                conn.execute(
                    "UPDATE jobs "
                    "SET status = 'scoring', updated_at = datetime('now') "
                    "WHERE id = ?",
                    (jid,),
                )
                requeued += 1

    return requeued
