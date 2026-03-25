"""
Database initialization and common queries for the job-search skill.
All data stored locally in SQLite (data/jobs.db).
"""

import sqlite3
import os
from pathlib import Path
from datetime import date, datetime
from typing import Optional

SKILL_ROOT = Path(__file__).parent.parent
DB_PATH = SKILL_ROOT / "data" / "jobs.db"

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    platform        TEXT NOT NULL,
    title           TEXT NOT NULL,
    company         TEXT NOT NULL,
    url             TEXT UNIQUE NOT NULL,
    description     TEXT,
    relevance_score REAL,
    matched_resume  TEXT,
    posted_date     DATE,
    discovered_date DATE NOT NULL DEFAULT (date('now')),
    status          TEXT NOT NULL DEFAULT 'new',
    skip_reason     TEXT,
    url_hash        TEXT,
    title_hash      TEXT,
    updated_at      DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS applications (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(id),
    applied_date     DATE NOT NULL DEFAULT (date('now')),
    resume_path      TEXT NOT NULL,
    tailored_path    TEXT,
    status           TEXT NOT NULL,
    failure_reason   TEXT,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS qa_store (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    question_pattern TEXT NOT NULL UNIQUE,
    answer           TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'user_provided',
    embedding        BLOB,
    used_count       INTEGER DEFAULT 0,
    last_used        DATE,
    created_at       DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pending_questions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(id),
    question_text    TEXT NOT NULL,
    question_context TEXT,
    question_type    TEXT,
    asked_in_digest  DATE,
    answered         INTEGER DEFAULT 0,
    answered_at      DATETIME,
    matched_qa_id    INTEGER REFERENCES qa_store(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status      ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_url_hash    ON jobs(url_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_title_hash  ON jobs(title_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_discovered  ON jobs(discovered_date);
CREATE INDEX IF NOT EXISTS idx_applications_job ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_pending_answered ON pending_questions(answered);
"""

SEED_QA_SQL = """
INSERT OR IGNORE INTO qa_store (question_pattern, answer, source) VALUES
    ('are you authorized to work in the united states', '{work_authorized}', 'onboarding'),
    ('will you now or in the future require sponsorship', '{sponsorship}', 'onboarding'),
    ('what is your desired salary', '{salary}', 'onboarding'),
    ('what are your salary expectations', '{salary}', 'onboarding'),
    ('what is your notice period', '{notice_period}', 'onboarding'),
    ('were you referred by anyone', '{referral}', 'onboarding'),
    ('do you have any referrals', '{referral}', 'onboarding'),
    ('are you open to relocation', 'I am open to relocation depending on the role and location.', 'onboarding');
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode and foreign keys enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all tables and indexes if they don't already exist."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    print(f"Database initialized at {db_path}")


def seed_qa_store(profile: dict, db_path: Path = DB_PATH) -> None:
    """Populate Q&A store with standard answers from the user profile."""
    app = profile.get("application_defaults", {})
    prefs = profile.get("job_preferences", {})

    work_authorized = "Yes" if app.get("work_authorized", True) else "No"
    sponsorship = "Yes" if app.get("visa_sponsorship_required", False) else "No"
    salary_min = prefs.get("salary_min", 0)
    salary_preferred = prefs.get("salary_preferred", 0)
    currency = prefs.get("salary_currency", "USD")
    notice_days = app.get("notice_period_days", 14)
    referral = "No" if not app.get("referral", False) else "Yes"

    salary_answer = (
        f"My salary expectation is {salary_preferred:,} {currency}, "
        f"with a minimum of {salary_min:,} {currency}."
        if salary_preferred > 0
        else "My salary expectation is negotiable based on the total compensation package."
    )
    notice_answer = (
        f"My notice period is {notice_days} days."
        if notice_days > 0
        else "I am available to start immediately."
    )

    sql = SEED_QA_SQL.format(
        work_authorized=work_authorized,
        sponsorship=sponsorship,
        salary=salary_answer,
        notice_period=notice_answer,
        referral=referral,
    )
    with get_connection(db_path) as conn:
        conn.executescript(sql)
    print("Q&A store seeded with onboarding defaults.")


def get_daily_stats(db_path: Path = DB_PATH) -> dict:
    """Return a summary of today's activity."""
    today = date.today().isoformat()
    with get_connection(db_path) as conn:
        applied = conn.execute(
            "SELECT COUNT(*) FROM applications WHERE applied_date = ? AND status = 'submitted'",
            (today,),
        ).fetchone()[0]
        queued = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'queued'",
        ).fetchone()[0]
        skipped = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'skipped' AND discovered_date = ?",
            (today,),
        ).fetchone()[0]
        pending_q = conn.execute(
            "SELECT COUNT(*) FROM pending_questions WHERE answered = 0",
        ).fetchone()[0]
        total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    return {
        "date": today,
        "applied_today": applied,
        "queued": queued,
        "skipped_today": skipped,
        "pending_questions": pending_q,
        "total_jobs_tracked": total_jobs,
    }


def list_queued_jobs(db_path: Path = DB_PATH) -> list[sqlite3.Row]:
    """Return all jobs in queued status, newest first."""
    with get_connection(db_path) as conn:
        return conn.execute(
            """
            SELECT j.id, j.title, j.company, j.platform, j.url,
                   j.relevance_score, j.discovered_date,
                   GROUP_CONCAT(pq.question_text, ' | ') AS unanswered_questions
            FROM jobs j
            LEFT JOIN pending_questions pq ON pq.job_id = j.id AND pq.answered = 0
            WHERE j.status = 'queued'
            GROUP BY j.id
            ORDER BY j.relevance_score DESC, j.discovered_date DESC
            """
        ).fetchall()


def mark_job_status(job_id: str, status: str, reason: str = None, db_path: Path = DB_PATH) -> None:
    """Update a job's status."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, skip_reason = ?, updated_at = datetime('now') WHERE id = ?",
            (status, reason, job_id),
        )


def job_exists(url_hash: str, title_hash: str, db_path: Path = DB_PATH) -> bool:
    """Check if a job is already in the database (deduplication)."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM jobs WHERE url_hash = ? OR title_hash = ?",
            (url_hash, title_hash),
        ).fetchone()
    return row is not None


def insert_job(job: dict, db_path: Path = DB_PATH) -> bool:
    """
    Insert a new job. Returns True if inserted, False if duplicate.
    job dict keys: id, platform, title, company, url, description,
                   posted_date, url_hash, title_hash
    """
    if job_exists(job["url_hash"], job["title_hash"], db_path):
        return False
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO jobs
                (id, platform, title, company, url, description, posted_date, url_hash, title_hash)
            VALUES
                (:id, :platform, :title, :company, :url, :description,
                 :posted_date, :url_hash, :title_hash)
            """,
            job,
        )
    return True


def update_job_score(job_id: str, score: float, resume_path: str, db_path: Path = DB_PATH) -> None:
    """Store relevance score and matched resume after scoring."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET relevance_score = ?, matched_resume = ?, status = 'scoring', "
            "updated_at = datetime('now') WHERE id = ?",
            (score, resume_path, job_id),
        )


def record_application(job_id: str, resume_path: str, tailored_path: Optional[str],
                        status: str, failure_reason: str = None,
                        db_path: Path = DB_PATH) -> int:
    """Insert an application record and return its id."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO applications (job_id, resume_path, tailored_path, status, failure_reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, resume_path, tailored_path, status, failure_reason),
        )
        return cur.lastrowid


def get_unscored_jobs(limit: int = 100, db_path: Path = DB_PATH) -> list[sqlite3.Row]:
    """Return jobs that have been discovered but not yet scored."""
    with get_connection(db_path) as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE status = 'new' AND relevance_score IS NULL LIMIT ?",
            (limit,),
        ).fetchall()


def get_jobs_for_dispatch(threshold: float = 0.40, db_path: Path = DB_PATH) -> list[sqlite3.Row]:
    """Return scored jobs above threshold that haven't been acted on yet."""
    with get_connection(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'scoring' AND relevance_score >= ?
            ORDER BY relevance_score DESC
            """,
            (threshold,),
        ).fetchall()


def count_applications_today(db_path: Path = DB_PATH) -> int:
    """Count submitted applications today (for rate limiting)."""
    today = date.today().isoformat()
    with get_connection(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM applications WHERE applied_date = ? AND status = 'submitted'",
            (today,),
        ).fetchone()[0]


def add_pending_question(job_id: str, question_text: str,
                          question_context: str = None, question_type: str = "text",
                          db_path: Path = DB_PATH) -> int:
    """Log a question that couldn't be answered automatically."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO pending_questions (job_id, question_text, question_context, question_type)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, question_text, question_context, question_type),
        )
        return cur.lastrowid


def get_unanswered_questions(db_path: Path = DB_PATH) -> list[sqlite3.Row]:
    """Return all unanswered pending questions with their job info."""
    with get_connection(db_path) as conn:
        return conn.execute(
            """
            SELECT pq.*, j.title, j.company, j.platform, j.url
            FROM pending_questions pq
            JOIN jobs j ON j.id = pq.job_id
            WHERE pq.answered = 0
            ORDER BY j.relevance_score DESC, pq.id ASC
            """
        ).fetchall()


if __name__ == "__main__":
    init_db()
    print("Schema created. Run 'python3 scripts/main.py setup' to configure your profile.")
