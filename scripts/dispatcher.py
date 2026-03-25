"""
Phase 4: Application dispatcher.

Decides whether to auto-apply, queue, skip, or flag a job for manual
notification. Runs the full fill_application flow and handles the result.

Decision rules
--------------
SKIP        score < relevance_threshold  OR  daily limit already reached
AUTO        score ≥ threshold AND no unanswerable fields in pre-flight scan
QUEUE       score ≥ threshold AND ≥1 unanswerable field / essay found, OR
            browser fill came back INCOMPLETE / CAPTCHA / ERROR
NOTIFY      score ≥ perfect_match_threshold AND unanswerable (essay required)

Rate limiting
-------------
- Pause rate_limit_seconds (default 150 s = ~2.5 min) between live submissions
- Stop when count_applications_today() reaches max_applications_per_day
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SKILL_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class DispatchDecision(Enum):
    AUTO   = "auto"
    QUEUE  = "queue"
    SKIP   = "skip"
    NOTIFY = "notify"


@dataclass
class DispatchResult:
    decision: DispatchDecision
    job_id: str
    reason: str = ""
    missing_fields: list = field(default_factory=list)
    fill_status: str = ""


# ---------------------------------------------------------------------------
# Pre-flight complexity check (no browser required)
# ---------------------------------------------------------------------------

_ESSAY_TRIGGERS = [
    "cover letter required",
    "include a cover letter",
    "write a brief",
    "answer the following questions",
    "please describe",
    "tell us about yourself",
    "why do you want to work",
    "describe your experience",
]


def score_complexity(job: dict) -> tuple[int, list[str]]:
    """
    Estimate how complex the application is by scanning the job description
    for signals that require a human-written essay (cover letter, etc.).

    Returns (unanswerable_count, list_of_detected_labels).
    This is a cheap heuristic — no browser required.
    """
    description = (job.get("description") or "").lower()
    found = []
    for trigger in _ESSAY_TRIGGERS:
        if trigger in description:
            found.append(trigger[:60])
    return len(found), found


# ---------------------------------------------------------------------------
# Single-job decision (pre-browser)
# ---------------------------------------------------------------------------

def decide(
    job: dict,
    profile: dict,
    applied_today: int = 0,
) -> DispatchResult:
    """
    Return a DispatchResult based on score + complexity without opening a browser.
    Used as a fast filter before `apply_single` fires up bb-browser.
    """
    job_id = job.get("id", "?")
    score = float(job.get("relevance_score") or 0.0)

    automation = profile.get("automation", {})
    threshold       = float(automation.get("relevance_threshold", 0.40))
    perfect_thresh  = float(automation.get("perfect_match_threshold", 0.85))
    max_apps        = int(automation.get("max_applications_per_day", 20))

    if applied_today >= max_apps:
        return DispatchResult(
            decision=DispatchDecision.SKIP,
            job_id=job_id,
            reason=f"Daily limit reached ({applied_today}/{max_apps})",
        )

    if score < threshold:
        return DispatchResult(
            decision=DispatchDecision.SKIP,
            job_id=job_id,
            reason=f"Score {score:.2f} below threshold {threshold:.2f}",
        )

    unanswerable, missing = score_complexity(job)
    if unanswerable > 0:
        if score >= perfect_thresh:
            return DispatchResult(
                decision=DispatchDecision.NOTIFY,
                job_id=job_id,
                reason=f"Perfect match ({score:.2f}) but requires essay/cover letter",
                missing_fields=missing,
            )
        return DispatchResult(
            decision=DispatchDecision.QUEUE,
            job_id=job_id,
            reason=f"Has {unanswerable} unanswerable field(s) detected in description",
            missing_fields=missing,
        )

    return DispatchResult(
        decision=DispatchDecision.AUTO,
        job_id=job_id,
        reason=f"All fields answerable, score {score:.2f}",
    )


# ---------------------------------------------------------------------------
# Full apply flow for a single job
# ---------------------------------------------------------------------------

def apply_single(
    job_id: str,
    profile: dict,
    resume_path: str,
    dry_run: bool = False,
    client=None,
    db_path=None,
    rate_limit_seconds: int = 150,
) -> DispatchResult:
    """
    Run the complete application flow for one job:
      1. Pre-flight decision (decide())
      2. fill_application() via bb-browser
      3. Persist result to DB

    Returns a DispatchResult.
    """
    import db as _db
    import bot_filter as _bf
    from form_filler import fill_application, FillStatus

    if db_path is None:
        db_path = _db.DB_PATH

    with _db.get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()

    if not row:
        return DispatchResult(
            decision=DispatchDecision.SKIP,
            job_id=job_id,
            reason="Job not found in database",
        )

    job = dict(row)
    applied_today = _db.count_applications_today(db_path)

    # --- Pre-flight ---
    pre = decide(job, profile, applied_today)
    if pre.decision == DispatchDecision.SKIP:
        _db.mark_job_status(job_id, "skipped", reason=pre.reason, db_path=db_path)
        return pre

    if pre.decision == DispatchDecision.NOTIFY:
        _db.mark_job_status(job_id, "manual_notified", reason=pre.reason, db_path=db_path)
        return pre

    # --- Browser fill ---
    def bot_filter_fn(text: str) -> str:
        return _bf.scan_and_strip(text, llm_client=client)

    result = fill_application(
        job=job,
        resume_path=resume_path,
        profile=profile,
        dry_run=dry_run,
        client=client,
        bot_filter_fn=bot_filter_fn,
    )

    # --- Persist result ---
    if result.status == FillStatus.SUCCESS:
        tailored = job.get("matched_resume")
        _db.record_application(
            job_id=job_id,
            resume_path=resume_path,
            tailored_path=tailored if tailored and tailored != resume_path else None,
            status="submitted",
            db_path=db_path,
        )
        _db.mark_job_status(job_id, "auto_applied", db_path=db_path)
        logger.info("Applied to %s @ %s", job.get("title"), job.get("company"))
        if not dry_run and rate_limit_seconds > 0:
            time.sleep(rate_limit_seconds)
        return DispatchResult(
            decision=DispatchDecision.AUTO,
            job_id=job_id,
            reason="Submitted successfully",
            fill_status="success",
        )

    if result.status == FillStatus.DRY_RUN:
        _db.record_application(
            job_id=job_id,
            resume_path=resume_path,
            tailored_path=None,
            status="dry_run",
            db_path=db_path,
        )
        return DispatchResult(
            decision=DispatchDecision.AUTO,
            job_id=job_id,
            reason="Dry run — form filled, not submitted",
            fill_status="dry_run",
        )

    if result.status == FillStatus.CAPTCHA:
        _db.mark_job_status(job_id, "queued", reason="CAPTCHA detected", db_path=db_path)
        return DispatchResult(
            decision=DispatchDecision.QUEUE,
            job_id=job_id,
            reason="CAPTCHA blocked the form",
            fill_status="captcha",
        )

    if result.status == FillStatus.INCOMPLETE:
        for question_text in result.missing_fields:
            _db.add_pending_question(
                job_id=job_id,
                question_text=question_text,
                question_type="essay" if len(question_text) >= 80 else "text",
                db_path=db_path,
            )
        _db.mark_job_status(
            job_id, "queued",
            reason=f"Missing: {', '.join(result.missing_fields[:3])}",
            db_path=db_path,
        )
        return DispatchResult(
            decision=DispatchDecision.QUEUE,
            job_id=job_id,
            reason=f"Incomplete — {len(result.missing_fields)} unanswered field(s)",
            missing_fields=result.missing_fields,
            fill_status="incomplete",
        )

    # FillStatus.ERROR
    _db.mark_job_status(
        job_id, "queued",
        reason=f"Fill error: {result.error_message[:80]}",
        db_path=db_path,
    )
    return DispatchResult(
        decision=DispatchDecision.QUEUE,
        job_id=job_id,
        reason=f"Error: {result.error_message}",
        fill_status="error",
    )


# ---------------------------------------------------------------------------
# Batch pipeline dispatcher
# ---------------------------------------------------------------------------

def run_dispatch(
    profile: dict,
    dry_run: bool = False,
    client=None,
    db_path=None,
    rate_limit_seconds: int = 150,
) -> dict:
    """
    Dispatch all jobs that are ready for application (status='scoring').

    Respects the daily application cap and rate limit between submissions.
    Returns a stats dict: {auto, queued, skipped, notify, errors}.
    """
    import db as _db

    if db_path is None:
        db_path = _db.DB_PATH

    automation = profile.get("automation", {})
    threshold = float(automation.get("relevance_threshold", 0.40))
    max_apps  = int(automation.get("max_applications_per_day", 20))

    jobs  = _db.get_jobs_for_dispatch(threshold=threshold, db_path=db_path)
    stats = {"auto": 0, "queued": 0, "skipped": 0, "notify": 0, "errors": 0}

    for job in jobs:
        if _db.count_applications_today(db_path) >= max_apps:
            logger.info("Daily limit reached (%d). Stopping dispatch.", max_apps)
            break

        job_dict     = dict(job)
        resume_path  = job_dict.get("matched_resume") or ""

        try:
            result = apply_single(
                job_id=job_dict["id"],
                profile=profile,
                resume_path=resume_path,
                dry_run=dry_run,
                client=client,
                db_path=db_path,
                rate_limit_seconds=rate_limit_seconds if not dry_run else 0,
            )
        except Exception as exc:
            logger.error("Dispatch error for job %s: %s", job_dict.get("id"), exc)
            stats["errors"] += 1
            continue

        stats[result.decision.value] = stats.get(result.decision.value, 0) + 1

    return stats
