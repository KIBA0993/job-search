"""
Phase 2: Job relevance scoring via Mammoth AI (OpenAI-compatible API).

For each unscored job in the database:
  1. Read all configured resume versions (PDF, DOCX, TXT)
  2. Call the AI API to rate match between each resume and the job description
  3. Keep the highest score and the resume that produced it
  4. Update the database: score < threshold → skipped, else → status 'scoring'

Requires MAMMOTH_API_KEY (or the env var named in user_profile.json ai.mammoth_api_key_env).
If the key is missing or the API call fails, the job stays 'new' and a warning is logged.
"""

import logging
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional, Union

SKILL_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resume text extraction
# ---------------------------------------------------------------------------

def read_resume_text(path: Union[str, Path], max_chars: int = 8000) -> str:
    """
    Extract plain text from a resume file.
    Supports: .txt, .pdf (requires pypdf), .docx (requires python-docx).
    Returns '' if the file is missing or unreadable.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Resume file not found: %s", path)
        return ""

    suffix = path.suffix.lower()

    if suffix == ".txt":
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except Exception as exc:
            logger.warning("Could not read .txt resume %s: %s", path, exc)
            return ""

    if suffix == ".pdf":
        # Method 1: pypdf (pure Python, preferred)
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages).strip()
            if text:
                return text[:max_chars]
            logger.debug("pypdf returned empty text for %s — trying fallbacks", path)
        except ImportError:
            logger.debug("pypdf not installed, trying fallbacks for %s", path)
        except Exception as exc:
            logger.debug("pypdf failed for %s: %s — trying fallbacks", path, exc)

        # Method 2: pdftotext CLI (from poppler-utils / brew install poppler)
        if shutil.which("pdftotext"):
            try:
                result = subprocess.run(
                    ["pdftotext", "-layout", str(path), "-"],
                    capture_output=True, text=True, timeout=30,
                )
                text = result.stdout.strip()
                if text:
                    logger.debug("pdftotext succeeded for %s", path)
                    return text[:max_chars]
            except Exception as exc:
                logger.debug("pdftotext failed for %s: %s", path, exc)

        # Method 3: pdfminer.six
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract  # type: ignore
            text = pdfminer_extract(str(path)).strip()
            if text:
                logger.debug("pdfminer succeeded for %s", path)
                return text[:max_chars]
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("pdfminer failed for %s: %s", path, exc)

        logger.warning(
            "Could not extract text from PDF: %s\n"
            "  Fix: pip3 install pypdf\n"
            "  Or:  brew install poppler  (adds pdftotext CLI)",
            path,
        )
        return ""

    if suffix in (".docx", ".doc"):
        try:
            from docx import Document  # type: ignore
            doc = Document(str(path))
            text = "\n".join(p.text for p in doc.paragraphs)
            return text[:max_chars]
        except ImportError:
            logger.warning(
                "python-docx not installed — cannot read DOCX resume %s. "
                "Install with: pip install python-docx", path,
            )
            return ""
        except Exception as exc:
            logger.warning("Could not read .docx resume %s: %s", path, exc)
            return ""

    logger.warning("Unsupported resume format '%s' for file: %s", suffix, path)
    return ""


def load_resume_versions(resume_map: dict, skill_root: Path = SKILL_ROOT) -> list[tuple[str, str]]:
    """
    Load text for every resume version in the resume_map config.
    Returns list of (absolute_path_str, text) tuples for non-empty results.
    """
    versions: list[tuple[str, str]] = []
    seen_paths: set[str] = set()

    def _add(raw_path: str):
        if not raw_path:
            return
        p = Path(raw_path) if Path(raw_path).is_absolute() else skill_root / raw_path
        key = str(p.resolve())
        if key in seen_paths:
            return
        seen_paths.add(key)
        text = read_resume_text(p)
        if text.strip():
            versions.append((str(p), text))
        else:
            logger.debug("No text extracted from resume: %s", p)

    # Add domain-specific resumes
    for domain_val in resume_map.get("domains", {}).values():
        _add(domain_val.get("resume", ""))

    # Add default resume (may already be covered by a domain)
    _add(resume_map.get("default", ""))

    if not versions:
        logger.warning(
            "No resume text could be loaded. Place resume files in data/resumes/ "
            "and run 'python3 main.py setup' to configure the paths."
        )
    return versions


# ---------------------------------------------------------------------------
# AI client
# ---------------------------------------------------------------------------

def create_ai_client(profile: dict):
    """
    Return an OpenAI-compatible client configured for Mammoth AI.
    Returns None if the API key is missing or the openai package is absent.
    """
    ai_cfg = profile.get("ai", {})
    key_env = ai_cfg.get("mammoth_api_key_env", "MAMMOTH_API_KEY")
    api_key = os.environ.get(key_env, "")

    if not api_key:
        logger.warning(
            "%s not set — AI scoring disabled. "
            "Export %s=<your-key> or re-run setup.", key_env, key_env,
        )
        return None

    try:
        import openai  # type: ignore
        base_url = ai_cfg.get("mammoth_api_base", "https://api.openai.com/v1")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return client
    except ImportError:
        logger.warning(
            "openai package not installed — AI scoring disabled. "
            "Install with: pip install openai"
        )
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_SCORE_PROMPT = (
    "You are a recruiter evaluating resume-job fit. "
    "Rate how well the following resume matches the job description. "
    "Respond with a single decimal number between 0.0 and 1.0 — nothing else.\n\n"
    "RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job}"
)


def score_single(
    job_description: str,
    resume_text: str,
    client,
    model: str = "gpt-4o-mini",
) -> float:
    """
    Call the AI API to score one resume against one job description.
    Returns a float in [0.0, 1.0], or 0.0 on any failure.
    """
    if not job_description.strip() or not resume_text.strip():
        return 0.0

    # Truncate to keep prompt within token limits
    resume_trunc = resume_text[:4000]
    jd_trunc = job_description[:3000]

    prompt = _SCORE_PROMPT.format(resume=resume_trunc, job=jd_trunc)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        # Extract first numeric value (may be negative) from the response
        match = re.search(r"-?\d+\.?\d*", raw)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
        logger.debug("score_single: unexpected response: %s", raw)
        return 0.0
    except Exception as exc:
        logger.warning("score_single API call failed: %s", exc)
        return 0.0


def score_job(
    job_description: str,
    resume_versions: list[tuple[str, str]],
    client,
    model: str = "gpt-4o-mini",
) -> tuple[float, str]:
    """
    Score a job description against all resume versions.
    Returns (best_score, best_resume_path).
    If no AI client or no resumes, returns (0.0, '').
    """
    if not client or not resume_versions:
        return 0.0, ""

    best_score = 0.0
    best_path = resume_versions[0][0] if resume_versions else ""

    for resume_path, resume_text in resume_versions:
        score = score_single(job_description, resume_text, client, model)
        logger.debug("score_job: %s → %.2f", Path(resume_path).name, score)
        if score > best_score:
            best_score = score
            best_path = resume_path

    return best_score, best_path


# ---------------------------------------------------------------------------
# Scoring runner
# ---------------------------------------------------------------------------

def run_scoring(
    profile: dict,
    resume_map: dict,
    db_path=None,
    skill_root: Path = SKILL_ROOT,
    batch_size: int = 50,
) -> dict:
    """
    Score all unscored jobs in the database.

    Returns:
        {
          "scored": int,       # jobs that received a score
          "skipped_low": int,  # jobs scored below the threshold
          "skipped_no_ai": int,# jobs skipped because AI is unavailable
          "errors": int,       # jobs that failed to score
        }
    """
    import db as _db  # local import to avoid circular deps in tests

    if db_path is None:
        db_path = _db.DB_PATH

    stats = {"scored": 0, "skipped_low": 0, "skipped_no_ai": 0, "errors": 0}

    # Load AI client
    client = create_ai_client(profile)
    if not client:
        unscored = _db.get_unscored_jobs(limit=1, db_path=db_path)
        if unscored:
            logger.warning(
                "AI client unavailable — %d jobs remain unscored.", len(unscored)
            )
        stats["skipped_no_ai"] = -1  # sentinel: check total separately
        return stats

    # Load resume versions
    resume_versions = load_resume_versions(resume_map, skill_root)
    if not resume_versions:
        logger.warning("No resume text available — scoring skipped.")
        return stats

    ai_cfg = profile.get("ai", {})
    model = ai_cfg.get("scoring_model", "gpt-4o-mini")
    threshold = float(
        profile.get("automation", {}).get("relevance_threshold", 0.40)
    )

    jobs = _db.get_unscored_jobs(limit=batch_size, db_path=db_path)
    logger.info("Scoring %d unscored jobs (threshold=%.2f)...", len(jobs), threshold)

    for job in jobs:
        job_id = job["id"]
        description = job["description"] or ""

        if not description.strip():
            logger.debug("No description for job %s — assigning score 0", job_id)
            _db.mark_job_status(job_id, "skipped", reason="no description", db_path=db_path)
            stats["skipped_low"] += 1
            continue

        try:
            score, resume_path = score_job(description, resume_versions, client, model)
        except Exception as exc:
            logger.error("Error scoring job %s: %s", job_id, exc)
            stats["errors"] += 1
            continue

        if score < threshold:
            _db.mark_job_status(
                job_id, "skipped",
                reason=f"low relevance score: {score:.2f}",
                db_path=db_path,
            )
            stats["skipped_low"] += 1
            logger.debug("Skipped %s — score %.2f < %.2f", job_id, score, threshold)
        else:
            _db.update_job_score(job_id, score, resume_path, db_path)
            stats["scored"] += 1
            logger.info("Scored %s: %.2f → %s", job_id, score, Path(resume_path).name)

    logger.info(
        "Scoring done — scored: %d, skipped_low: %d, errors: %d",
        stats["scored"], stats["skipped_low"], stats["errors"],
    )
    return stats
