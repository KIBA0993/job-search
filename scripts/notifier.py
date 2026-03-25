"""
Phase 5: Notification System — Gmail digest sender + reply ingestor.

Daily digest email (HTML) has 4 sections:
  A. Applications submitted today
  B. Queued applications needing manual review
  C. Unanswered questions (from pending_questions table)
  D. Up to 5 perfect-match jobs for manual apply (manual_notified status)

Reply ingestion:
  - Checks Gmail INBOX for emails with subject starting with
    "Re: Job Search Daily Digest"
  - Parses "ANSWER: <question_id>: <your answer>" lines
  - Stores answers in qa_store via question_handler.add_answer()
  - Calls requeue_jobs_for_question() to re-dispatch blocked jobs

Gmail OAuth:
  - Credentials JSON from Google Cloud Console → config/.gmail_credentials.json
  - Token cached to config/.gmail_token.json
  - Scopes: gmail.send, gmail.readonly
  - Falls back to writing digest to data/logs/digest_YYYY-MM-DD.txt if
    Gmail is unavailable or not configured.
"""

import base64
import logging
import re
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SKILL_ROOT = Path(__file__).parent.parent

DIGEST_SUBJECT_PREFIX = "Job Search Daily Digest"

# Matches lines like: "ANSWER: 12: Yes, I am authorized."
# Case-insensitive; supports extra spaces around colons.
ANSWER_PATTERN = re.compile(
    r"ANSWER\s*:\s*(\d+)\s*:\s*(.+)", re.IGNORECASE | re.MULTILINE
)

MAX_MANUAL_NOTIFY = 5

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]


# ---------------------------------------------------------------------------
# Gmail OAuth helpers
# ---------------------------------------------------------------------------

def get_gmail_service(credentials_path: str, token_path: str):
    """
    Build and return an authenticated Gmail API service object.

    Returns None if:
    - google-api-python-client is not installed
    - credentials file does not exist
    - OAuth flow cannot complete
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        logger.warning(
            "Gmail API libraries not installed. "
            "Run: pip install google-api-python-client google-auth-oauthlib"
        )
        return None

    creds_path = Path(credentials_path)
    tok_path = Path(token_path)

    if not creds_path.exists():
        logger.warning("Gmail credentials file not found: %s", creds_path)
        return None

    creds = None
    if tok_path.exists():
        creds = Credentials.from_authorized_user_file(str(tok_path), GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                logger.warning("Token refresh failed: %s", exc)
                creds = None
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(creds_path), GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)

        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tok_path.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# HTML digest builder
# ---------------------------------------------------------------------------

def build_digest_html(
    stats: dict,
    applied_jobs: list,
    queued_jobs: list,
    unanswered_questions: list,
    notify_jobs: list,
    today: str = None,
) -> str:
    """
    Build the HTML body for the daily digest email.

    Parameters
    ----------
    stats               : dict from db.get_daily_stats()
    applied_jobs        : list of rows with title, company, platform, url
    queued_jobs         : list of rows with title, company, url, relevance_score
    unanswered_questions: list of rows with id, question_text, title, company
    notify_jobs         : list of rows with title, company, url, relevance_score
                          (capped internally to MAX_MANUAL_NOTIFY)
    today               : ISO date string (defaults to today)
    """
    today = today or date.today().isoformat()

    def _e(s) -> str:
        """HTML-escape a value."""
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    hdr_style = (
        "font-family:sans-serif;font-size:18px;color:#333;"
        "border-bottom:2px solid #4CAF50;padding-bottom:6px;margin-top:24px;"
    )
    item_style = "font-family:sans-serif;font-size:14px;margin:6px 0;color:#444;"
    link_style = "color:#1a73e8;text-decoration:none;"
    mono_style = (
        "font-family:monospace;background:#f5f5f5;"
        "padding:2px 6px;border-radius:3px;font-size:13px;"
    )
    dim_style = "font-family:sans-serif;font-size:12px;color:#888;"

    sections: list[str] = []

    # ── Section A: Applications Submitted Today ───────────────────────────
    sections.append(f'<h2 style="{hdr_style}">✅ Applied Today ({len(applied_jobs)})</h2>')
    if applied_jobs:
        rows = "".join(
            f'<li style="{item_style}">'
            f'<a href="{_e(j["url"])}" style="{link_style}">{_e(j["title"])}</a>'
            f" at <strong>{_e(j['company'])}</strong>"
            f' <span style="{dim_style}">({_e(j["platform"])})</span>'
            f"</li>"
            for j in applied_jobs
        )
        sections.append(f"<ul>{rows}</ul>")
    else:
        sections.append(f'<p style="{dim_style}">No applications submitted today.</p>')

    def _fmt_score(row) -> str:
        s = row["relevance_score"]
        return f"{s:.0%}" if s is not None else "N/A"

    # ── Section B: Queued Applications ───────────────────────────────────
    sections.append(f'<h2 style="{hdr_style}">⏳ Queued for Review ({len(queued_jobs)})</h2>')
    if queued_jobs:
        rows = "".join(
            f'<li style="{item_style}">'
            f'<a href="{_e(j["url"])}" style="{link_style}">{_e(j["title"])}</a>'
            f" at <strong>{_e(j['company'])}</strong>"
            f' <span style="{dim_style}">— score: {_fmt_score(j)}</span>'
            f"</li>"
            for j in queued_jobs
        )
        sections.append(f"<ul>{rows}</ul>")
    else:
        sections.append(f'<p style="{dim_style}">No queued jobs. All caught up!</p>')

    # ── Section C: Unanswered Questions ──────────────────────────────────
    sections.append(
        f'<h2 style="{hdr_style}">❓ Unanswered Questions ({len(unanswered_questions)})</h2>'
    )
    if unanswered_questions:
        sections.append(
            f'<p style="{item_style}">Reply to this email using the format shown '
            f"to provide each answer.</p>"
        )
        rows = "".join(
            f'<li style="{item_style}">'
            f'<em>{_e(q["title"])}</em> at <strong>{_e(q["company"])}</strong><br>'
            f"<strong>Q:</strong> {_e(q['question_text'])}<br>"
            f'<span style="{mono_style}">ANSWER: {q["id"]}: your answer here</span>'
            f"</li>"
            for q in unanswered_questions
        )
        sections.append(f"<ul>{rows}</ul>")
    else:
        sections.append(f'<p style="{dim_style}">No outstanding questions.</p>')

    # ── Section D: Perfect-Match Manual Apply ────────────────────────────
    capped = list(notify_jobs)[:MAX_MANUAL_NOTIFY]
    sections.append(
        f'<h2 style="{hdr_style}">⭐ Perfect Matches — Apply Manually ({len(capped)})</h2>'
    )
    if capped:
        sections.append(
            f'<p style="{item_style}">These high-relevance jobs require a cover letter '
            f"or essay. Apply manually when you have time:</p>"
        )
        rows = "".join(
            f'<li style="{item_style}">'
            f'<a href="{_e(j["url"])}" style="{link_style}">{_e(j["title"])}</a>'
            f" at <strong>{_e(j['company'])}</strong>"
            f' <span style="{dim_style}">— score: {_fmt_score(j)}'
            f" — requires essay/cover letter</span>"
            f"</li>"
            for j in capped
        )
        sections.append(f"<ul>{rows}</ul>")
    else:
        sections.append(f'<p style="{dim_style}">No perfect-match manual suggestions today.</p>')

    # ── Footer ────────────────────────────────────────────────────────────
    footer = (
        f'<hr style="margin-top:24px;border:none;border-top:1px solid #eee;">'
        f'<p style="{dim_style}">'
        f"Job Search Automation &nbsp;|&nbsp; {_e(today)}"
        f" &nbsp;|&nbsp; Applied: {stats.get('applied_today', 0)}"
        f" &nbsp;|&nbsp; Queued: {stats.get('queued', 0)}"
        f" &nbsp;|&nbsp; Skipped: {stats.get('skipped_today', 0)}"
        f"</p>"
    )

    body = "\n".join(sections) + "\n" + footer
    return (
        '<!DOCTYPE html><html><body style="max-width:700px;margin:0 auto;padding:16px;">'
        + body
        + "</body></html>"
    )


# ---------------------------------------------------------------------------
# Send digest
# ---------------------------------------------------------------------------

def send_digest(profile: dict, db_path=None) -> bool:
    """
    Build and send the daily digest email.

    Falls back to writing the HTML to data/logs/digest_YYYY-MM-DD.txt
    if Gmail is not configured or the send fails.

    Returns True if the email was sent successfully via Gmail.
    """
    import db as _db

    if db_path is None:
        db_path = _db.DB_PATH

    today = date.today().isoformat()
    notif_cfg = profile.get("notifications", {})
    to_addr = notif_cfg.get("gmail_digest_address", "")

    # Gather all data needed for the digest
    stats = _db.get_daily_stats(db_path)
    queued_jobs = _db.list_queued_jobs(db_path)
    unanswered_qs = _db.get_unanswered_questions(db_path)

    with _db.get_connection(db_path) as conn:
        applied_jobs = conn.execute(
            """
            SELECT j.title, j.company, j.platform, j.url
            FROM applications a
            JOIN jobs j ON j.id = a.job_id
            WHERE a.applied_date = ? AND a.status = 'submitted'
            ORDER BY a.id DESC
            """,
            (today,),
        ).fetchall()

        notify_jobs = conn.execute(
            """
            SELECT id, title, company, url, relevance_score
            FROM jobs
            WHERE status = 'manual_notified'
            ORDER BY relevance_score DESC
            LIMIT ?
            """,
            (MAX_MANUAL_NOTIFY,),
        ).fetchall()

    html = build_digest_html(
        stats=stats,
        applied_jobs=list(applied_jobs),
        queued_jobs=list(queued_jobs),
        unanswered_questions=list(unanswered_qs),
        notify_jobs=list(notify_jobs),
        today=today,
    )

    # Mark unanswered questions as included in today's digest (only first time)
    if unanswered_qs:
        q_ids = [q["id"] for q in unanswered_qs]
        placeholders = ",".join("?" * len(q_ids))
        with _db.get_connection(db_path) as conn:
            conn.execute(
                f"UPDATE pending_questions SET asked_in_digest = ? "
                f"WHERE id IN ({placeholders}) AND asked_in_digest IS NULL",
                [today] + q_ids,
            )

    # Attempt to send via Gmail
    sent = False
    if to_addr:
        creds_path = str(
            SKILL_ROOT / notif_cfg.get("gmail_credentials_path",
                                       "config/.gmail_credentials.json")
        )
        token_path = str(
            SKILL_ROOT / notif_cfg.get("gmail_token_path",
                                       "config/.gmail_token.json")
        )
        service = get_gmail_service(creds_path, token_path)
        if service:
            subject = (
                f"{DIGEST_SUBJECT_PREFIX} — {today} "
                f"({stats['applied_today']} applied, {stats['queued']} queued)"
            )
            sent = _gmail_send_html(service, to_addr, subject, html)
            if sent:
                logger.info("Daily digest sent to %s", to_addr)
            else:
                logger.warning("Gmail send failed — writing fallback log.")

    if not sent:
        _write_digest_log(html, today)

    return sent


def _gmail_send_html(service, to: str, subject: str, html_body: str) -> bool:
    """Send an HTML email via Gmail API. Returns True on success."""
    try:
        msg = MIMEMultipart("alternative")
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()
        return True
    except Exception as exc:
        logger.error("Gmail send error: %s", exc)
        return False


def _write_digest_log(html: str, today: str) -> None:
    """Write the digest HTML to a fallback log file."""
    log_dir = SKILL_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"digest_{today}.txt"
    log_path.write_text(html, encoding="utf-8")
    logger.info("Digest written to %s", log_path)


# ---------------------------------------------------------------------------
# Reply ingestion
# ---------------------------------------------------------------------------

def _parse_answer_lines(text: str) -> list:
    """
    Parse "ANSWER: <id>: <text>" lines from an email body.

    Returns a list of (question_id: int, answer_text: str) tuples.
    Handles extra whitespace and is case-insensitive.
    """
    results = []
    for match in ANSWER_PATTERN.finditer(text):
        try:
            qid = int(match.group(1))
            answer = match.group(2).strip()
            if answer:
                results.append((qid, answer))
        except (ValueError, IndexError):
            continue
    return results


def _get_email_body(payload: dict) -> str:
    """Recursively extract the plain-text body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        text = _get_email_body(part)
        if text:
            return text
    return ""


def poll_replies(
    profile: dict,
    db_path=None,
    client=None,
    since_date: str = None,
) -> dict:
    """
    Poll Gmail inbox for replies to the daily digest email.

    For each reply found, parse "ANSWER: N: text" lines and store them
    in qa_store. Then trigger re-queue for any jobs blocked by those questions.

    Parameters
    ----------
    profile     : user_profile dict (for Gmail credentials paths)
    db_path     : path to SQLite DB (defaults to SKILL_ROOT/data/jobs.db)
    client      : optional OpenAI-compatible client for embedding new answers
    since_date  : search Gmail for replies after this date (YYYY/MM/DD format);
                  defaults to yesterday

    Returns
    -------
    dict with keys: replies_found, answers_stored, jobs_requeued
    """
    import db as _db
    from question_handler import add_answer, requeue_jobs_for_question

    if db_path is None:
        db_path = _db.DB_PATH

    notif_cfg = profile.get("notifications", {})
    creds_path = str(
        SKILL_ROOT / notif_cfg.get("gmail_credentials_path",
                                   "config/.gmail_credentials.json")
    )
    token_path = str(
        SKILL_ROOT / notif_cfg.get("gmail_token_path",
                                   "config/.gmail_token.json")
    )

    service = get_gmail_service(creds_path, token_path)
    if not service:
        logger.warning("Gmail service unavailable — skipping reply poll.")
        return {"replies_found": 0, "answers_stored": 0, "jobs_requeued": 0}

    if since_date is None:
        since_date = (datetime.now() - timedelta(days=1)).strftime("%Y/%m/%d")

    query = f'subject:"Re: {DIGEST_SUBJECT_PREFIX}" after:{since_date}'
    stats_out = {"replies_found": 0, "answers_stored": 0, "jobs_requeued": 0}

    try:
        result = service.users().messages().list(
            userId="me", q=query, maxResults=20
        ).execute()
    except Exception as exc:
        logger.error("Gmail list error: %s", exc)
        return stats_out

    messages = result.get("messages", [])
    stats_out["replies_found"] = len(messages)

    for msg_meta in messages:
        try:
            msg = service.users().messages().get(
                userId="me", id=msg_meta["id"], format="full"
            ).execute()
        except Exception as exc:
            logger.warning("Failed to fetch message %s: %s", msg_meta["id"], exc)
            continue

        body = _get_email_body(msg.get("payload", {}))
        parsed = _parse_answer_lines(body)

        for qid, answer_text in parsed:
            with _db.get_connection(db_path) as conn:
                q_row = conn.execute(
                    "SELECT question_text FROM pending_questions WHERE id = ?",
                    (qid,),
                ).fetchone()

            if not q_row:
                logger.warning(
                    "No pending question with id=%d — skipping.", qid
                )
                continue

            question_text = q_row["question_text"]
            qa_id = add_answer(
                question=question_text,
                answer=answer_text,
                source="email_reply",
                client=client,
                db_path=db_path,
            )
            stats_out["answers_stored"] += 1
            logger.info(
                "Stored answer for question #%d: '%s'", qid, question_text[:60]
            )

            # Mark the pending_question as answered and link it to the new qa entry
            with _db.get_connection(db_path) as conn:
                conn.execute(
                    "UPDATE pending_questions "
                    "SET answered=1, answered_at=datetime('now'), matched_qa_id=? "
                    "WHERE id=?",
                    (qa_id, qid),
                )

            requeued = requeue_jobs_for_question(qa_id, db_path)
            stats_out["jobs_requeued"] += requeued
            if requeued:
                logger.info(
                    "Re-queued %d job(s) after answering question #%d",
                    requeued, qid,
                )

    return stats_out
