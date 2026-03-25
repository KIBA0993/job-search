#!/usr/bin/env python3
"""
Job Search Automation — CLI entry point.

Usage:
  python3 main.py setup                  # First-run wizard
  python3 main.py run [--dry-run]        # Full pipeline
  python3 main.py status                 # Today's stats
  python3 main.py queue                  # List queued jobs
  python3 main.py apply <job_id>         # Manually apply to one job
  python3 main.py skip <job_id>          # Manually skip a job
  python3 main.py qa add                 # Add a Q&A entry interactively
  python3 main.py qa list                # List all Q&A entries
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import date

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

# Load .env if present
_env_file = SKILL_ROOT / "config" / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import db


# ---------------------------------------------------------------------------
# Retry utility
# ---------------------------------------------------------------------------

def retry_with_backoff(fn, max_retries: int = 3, backoff_base: float = 5.0, exceptions=(Exception,)):
    """
    Call fn() up to max_retries times, sleeping backoff_base * attempt seconds
    between attempts.  Returns the result of the first successful call.
    Raises the last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except exceptions as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_time = backoff_base * attempt
                logging.getLogger(__name__).warning(
                    "Attempt %d/%d failed (%s). Retrying in %.0fs …",
                    attempt, max_retries, exc, sleep_time,
                )
                time.sleep(sleep_time)
    raise last_exc


# ---------------------------------------------------------------------------
# Run-log file handler setup
# ---------------------------------------------------------------------------

def _setup_file_logging(log_dir: Path, verbose: bool = False):
    """Add a FileHandler that writes to data/logs/run_YYYY-MM-DD.log."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_{date.today().isoformat()}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG if verbose else logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        logging.getLogger().addHandler(fh)
        return fh
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not open log file: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_profile() -> dict:
    path = SKILL_ROOT / "config" / "user_profile.json"
    if not path.exists():
        print("No configuration found. Run: python3 main.py setup")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_resume_map() -> dict:
    path = SKILL_ROOT / "config" / "resume_map.json"
    if not path.exists():
        return {"domains": {}, "default": ""}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_setup(args) -> None:
    import onboarding
    from_json = getattr(args, "from_json", None)
    if from_json:
        onboarding.run_from_json(from_json)
    else:
        onboarding.run(reconfigure_gmail=getattr(args, "reconfigure_gmail", False))


def cmd_parse_resume(args) -> None:
    """
    Extract structured profile info from a resume PDF/DOCX/TXT using Mammoth AI.
    Outputs JSON that the chatbot can use to populate the setup config.

    Usage:
        python3 main.py parse-resume /path/to/resume.pdf
        python3 main.py parse-resume /path/to/resume.pdf --pretty
    """
    import json as _json
    from pathlib import Path as _Path
    from scorer import read_resume_text

    resume_path = _Path(args.resume_path).expanduser()
    if not resume_path.exists():
        print(_json.dumps({"error": f"File not found: {resume_path}"}))
        sys.exit(1)

    text = read_resume_text(resume_path)
    if not text.strip():
        print(_json.dumps({"error": "Could not extract text from resume. Install pypdf: pip3 install pypdf"}))
        sys.exit(1)

    # Try AI extraction if key is available
    ai_cfg_path = SKILL_ROOT / "config" / "user_profile.json"
    profile_for_ai = {}
    if ai_cfg_path.exists():
        with open(ai_cfg_path) as f:
            profile_for_ai = json.load(f)

    # Load .env for API key
    _env_file = SKILL_ROOT / "config" / ".env"
    if _env_file.exists():
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    from scorer import create_ai_client
    client = create_ai_client(profile_for_ai) if profile_for_ai else None

    extracted = _parse_resume_with_ai(text, client)
    extracted["resume_text_preview"] = text[:500] + ("…" if len(text) > 500 else "")
    extracted["source_file"] = str(resume_path)

    indent = 2 if getattr(args, "pretty", False) else None
    print(json.dumps(extracted, indent=indent))


def _parse_resume_with_ai(resume_text: str, client) -> dict:
    """
    Call Mammoth AI to extract structured fields from resume text.
    Falls back to regex heuristics if AI is unavailable.
    """
    import re as _re

    EXTRACT_PROMPT = (
        "Extract the following fields from this resume and return ONLY valid JSON "
        "(no markdown, no explanation). Use null for any field you cannot find.\n\n"
        "Fields to extract:\n"
        "  name          - full name (string)\n"
        "  email         - email address (string)\n"
        "  phone         - phone number (string)\n"
        "  linkedin_url  - LinkedIn profile URL (string)\n"
        "  github_url    - GitHub profile URL (string)\n"
        "  portfolio_url - portfolio or personal website URL (string)\n"
        "  city          - city (string)\n"
        "  state         - state/province (string)\n"
        "  country       - country (string, default 'United States')\n"
        "  job_titles    - list of job titles this person is targeting or currently holds (array of strings)\n"
        "  skills        - top technical skills (array of strings, max 20)\n"
        "  years_of_experience - total years of professional experience (integer)\n"
        "  domains       - which of these domains fit this resume best "
        "(array from: ml_engineering, backend_engineering, product_management, data_engineering, general)\n\n"
        f"RESUME:\n{resume_text[:6000]}"
    )

    if client:
        try:
            ai_cfg_path = SKILL_ROOT / "config" / "user_profile.json"
            model = "gpt-4o-mini"
            if ai_cfg_path.exists():
                with open(ai_cfg_path) as f:
                    p = json.load(f)
                    model = p.get("ai", {}).get("scoring_model", "gpt-4o-mini")

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": EXTRACT_PROMPT}],
                temperature=0,
                max_tokens=800,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            return json.loads(raw)
        except Exception as exc:
            logging.getLogger(__name__).warning("AI extraction failed: %s — using regex fallback", exc)

    # Regex fallback
    result: dict = {
        "name": None, "email": None, "phone": None,
        "linkedin_url": None, "github_url": None, "portfolio_url": None,
        "city": None, "state": None, "country": "United States",
        "job_titles": [], "skills": [], "years_of_experience": None, "domains": [],
        "_note": "AI extraction unavailable — regex fallback used. Set MAMMOTH_API_KEY for full extraction.",
    }
    email_m = _re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", resume_text)
    if email_m:
        result["email"] = email_m.group()
    phone_m = _re.search(r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", resume_text)
    if phone_m:
        result["phone"] = phone_m.group()
    li_m = _re.search(r"linkedin\.com/in/[\w-]+", resume_text, _re.IGNORECASE)
    if li_m:
        result["linkedin_url"] = "https://" + li_m.group()
    gh_m = _re.search(r"github\.com/[\w-]+", resume_text, _re.IGNORECASE)
    if gh_m:
        result["github_url"] = "https://" + gh_m.group()
    return result


def cmd_status(args) -> None:
    stats = db.get_daily_stats()
    print(f"\nJob Search Status — {stats['date']}")
    print(f"{'─' * 40}")
    print(f"  Applied today       : {stats['applied_today']}")
    print(f"  Queued (need review): {stats['queued']}")
    print(f"  Skipped today       : {stats['skipped_today']}")
    print(f"  Unanswered questions: {stats['pending_questions']}")
    print(f"  Total jobs tracked  : {stats['total_jobs_tracked']}")
    print()

    profile = load_profile()
    max_apps = profile.get("automation", {}).get("max_applications_per_day", 20)
    remaining = max(0, max_apps - stats["applied_today"])
    print(f"  Applications remaining today: {remaining}/{max_apps}")
    print()


def cmd_queue(args) -> None:
    rows = db.list_queued_jobs()
    if not rows:
        print("\nNo queued jobs. All caught up!")
        return
    print(f"\nQueued Jobs ({len(rows)} total)")
    print(f"{'─' * 70}")
    for r in rows:
        score = f"{r['relevance_score']:.0%}" if r["relevance_score"] else "N/A"
        print(f"  [{r['id']}]")
        print(f"    {r['title']} @ {r['company']} ({r['platform']}) — score: {score}")
        print(f"    {r['url']}")
        if r["unanswered_questions"]:
            questions = r["unanswered_questions"].split(" | ")
            for q in questions[:2]:
                print(f"    Q: {q[:80]}{'...' if len(q) > 80 else ''}")
        print()


def cmd_skip(args) -> None:
    job_id = args.job_id
    db.mark_job_status(job_id, "skipped", reason="manual skip")
    print(f"  Skipped job {job_id}")


def cmd_apply(args) -> None:
    """Manually trigger Phase 4 dispatcher for a single queued job."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="  %(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    profile = load_profile()
    dry_run = getattr(args, "dry_run", False)

    # Verify the job exists and get its resume path
    rows = db.list_queued_jobs()
    job_row = next((r for r in rows if r["id"] == args.job_id), None)
    if not job_row:
        # Try searching all jobs, not just queued ones
        with db.get_connection() as conn:
            job_row = conn.execute(
                "SELECT id, title, company, url, matched_resume FROM jobs WHERE id = ?",
                (args.job_id,),
            ).fetchone()
    if not job_row:
        print(f"  Job {args.job_id} not found.")
        return

    print(f"\n  Applying to: {job_row['title']} @ {job_row['company']}")
    print(f"  URL: {job_row['url']}")
    if dry_run:
        print("  [DRY RUN] Form will be filled but not submitted.")
    print()

    from scorer import create_ai_client as _create_ai_client
    from dispatcher import apply_single, DispatchDecision

    ai_client = _create_ai_client(profile)
    resume_path = job_row["matched_resume"] or ""

    result = apply_single(
        job_id=args.job_id,
        profile=profile,
        resume_path=resume_path,
        dry_run=dry_run,
        client=ai_client,
        rate_limit_seconds=0,  # no delay for manual apply
    )

    decision_label = {
        DispatchDecision.AUTO:   "Applied" if not dry_run else "Dry-run filled",
        DispatchDecision.QUEUE:  "Queued",
        DispatchDecision.SKIP:   "Skipped",
        DispatchDecision.NOTIFY: "Flagged for manual apply",
    }.get(result.decision, result.decision.value)

    print(f"  Result: {decision_label}")
    print(f"  Reason: {result.reason}")
    if result.missing_fields:
        print("  Missing fields:")
        for f in result.missing_fields[:5]:
            print(f"    - {f}")


def cmd_qa_add(args) -> None:
    print("\nAdd Q&A Entry")
    print("─" * 40)
    question = input("Question (as it appears in forms): ").strip()
    if not question:
        print("Cancelled.")
        return
    print("Answer (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    answer = " ".join(lines).strip()
    if not answer:
        print("Cancelled — no answer provided.")
        return
    with db.get_connection() as conn:
        conn.execute(
            "INSERT INTO qa_store (question_pattern, answer, source) VALUES (lower(?), ?, 'user_provided') "
            "ON CONFLICT(question_pattern) DO UPDATE SET answer=excluded.answer, source=excluded.source",
            (question, answer),
        )
    print(f"\n  Stored: '{question[:60]}...' " if len(question) > 60 else f"\n  Stored: '{question}'")


def cmd_qa_list(args) -> None:
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT id, question_pattern, answer, source, used_count, last_used FROM qa_store ORDER BY used_count DESC"
        ).fetchall()
    if not rows:
        print("\nQ&A store is empty. Run 'python3 main.py setup' to seed defaults.")
        return
    print(f"\nQ&A Store ({len(rows)} entries)")
    print(f"{'─' * 70}")
    for r in rows:
        used = f"used {r['used_count']}x" + (f", last {r['last_used']}" if r["last_used"] else "")
        print(f"  [{r['id']}] ({r['source']}, {used})")
        print(f"    Q: {r['question_pattern'][:80]}")
        print(f"    A: {r['answer'][:100]}{'...' if len(r['answer']) > 100 else ''}")
        print()


def cmd_scheduler(args) -> None:
    """Manage the macOS launchd daily scheduler."""
    import scheduler as _scheduler

    sub = getattr(args, "scheduler_command", None)

    if sub == "install":
        profile = load_profile()
        schedule_hour = profile.get("automation", {}).get("schedule_hour", 9)
        result = _scheduler.install(
            skill_root=SKILL_ROOT,
            schedule_hour=schedule_hour,
        )
        print(f"\n  {'OK' if result['success'] else 'WARN'}: {result['message']}\n")

    elif sub == "uninstall":
        result = _scheduler.uninstall()
        print(f"\n  {result['message']}\n")

    elif sub == "status":
        info = _scheduler.status()
        print(f"\n  Scheduler Status")
        print(f"  {'─' * 40}")
        print(f"  Installed (plist on disk) : {'Yes' if info['installed'] else 'No'}")
        print(f"  Loaded (launchctl)        : {'Yes' if info['loaded'] else 'No'}")
        print(f"  Plist path                : {info['plist_path']}")
        if info["next_run"]:
            print(f"  Schedule                  : {info['next_run']}")
        print()

    elif sub == "qnap":
        _scheduler.print_qnap_guide()

    else:
        print("  Usage: job-search scheduler [install|uninstall|status|qnap]")


def cmd_notify(args) -> None:
    """Manually send the daily digest and poll for replies."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="  %(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    profile = load_profile()

    from notifier import send_digest, poll_replies
    from scorer import create_ai_client as _create_ai_client

    print("\n  Sending daily digest …")
    sent = send_digest(profile=profile)
    if sent:
        print("  Digest email sent.")
    else:
        print("  Digest written to data/logs/ (Gmail not configured or unavailable).")

    if getattr(args, "poll", False) or not getattr(args, "no_poll", False):
        print("  Polling Gmail for replies …")
        ai_client = _create_ai_client(profile)
        reply_stats = poll_replies(profile=profile, client=ai_client)
        print(
            f"  Replies found: {reply_stats['replies_found']}, "
            f"answers stored: {reply_stats['answers_stored']}, "
            f"jobs re-queued: {reply_stats['jobs_requeued']}"
        )
    print()


def cmd_run(args) -> None:
    """
    Full pipeline runner with file logging and per-phase error resilience.
    """
    import json as _json

    logger = logging.getLogger(__name__)
    dry_run = getattr(args, "dry_run", False)
    verbose = getattr(args, "verbose", False)

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="  %(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # File log — always write regardless of dry_run
    log_dir = SKILL_ROOT / "data" / "logs"
    _setup_file_logging(log_dir, verbose=verbose)
    logger.info("Run started (dry_run=%s)", dry_run)

    profile = load_profile()
    resume_map = load_resume_map()

    print(f"\nJob Search Run — {date.today().isoformat()}")
    if dry_run:
        print("  [DRY RUN] No applications will be submitted.")
    print(f"{'─' * 50}")

    # Validate config
    _validate_config(profile, resume_map)

    # Ensure DB exists
    db.init_db()
    stats = db.get_daily_stats()

    max_apps = profile.get("automation", {}).get("max_applications_per_day", 20)
    applied_today = stats["applied_today"]
    if applied_today >= max_apps:
        print(f"  Daily limit reached ({applied_today}/{max_apps}). No more applications today.")
        return

    print(f"  Applications today:  {applied_today}/{max_apps}")
    print(f"  Queued jobs:         {stats['queued']}")
    print(f"  Unanswered Qs:       {stats['pending_questions']}")
    print(f"  Total tracked:       {stats['total_jobs_tracked']}")
    print()

    # ------------------------------------------------------------------
    # Phase 2 — Job Discovery
    # ------------------------------------------------------------------
    platform_cfg_path = SKILL_ROOT / "config" / "platform_config.json"
    if not platform_cfg_path.exists():
        print("  ERROR: config/platform_config.json not found. Run setup first.")
        return

    with open(platform_cfg_path) as f:
        platform_cfg = _json.load(f)

    print("  [Phase 2] Starting job discovery across platforms …")
    try:
        from search import run_discovery

        def _do_discovery():
            return run_discovery(
                profile=profile,
                platform_cfg=platform_cfg,
                fetch_descriptions=not dry_run,
            )

        discovery_results = retry_with_backoff(_do_discovery, max_retries=3, backoff_base=5)

        total_found = sum(r.found for r in discovery_results.values())
        total_new = sum(r.new for r in discovery_results.values())
        print(f"  Discovery complete — found {total_found} listings, {total_new} new jobs stored.")
        logger.info("Discovery complete: found=%d new=%d", total_found, total_new)
        if discovery_results:
            for pid, res in discovery_results.items():
                if res.found or res.errors:
                    err_str = f", {len(res.errors)} error(s)" if res.errors else ""
                    print(f"    {pid:22s} found={res.found}, new={res.new}{err_str}")
                    if res.errors:
                        for e in res.errors:
                            logger.warning("  Platform %s error: %s", pid, e)
    except Exception as exc:
        logger.error("Phase 2 (discovery) failed: %s", exc)
        print(f"  WARNING: Job discovery failed ({exc}). Continuing with existing jobs.")
    print()

    # ------------------------------------------------------------------
    # Phase 2 — Scoring
    # ------------------------------------------------------------------
    print("  [Phase 2] Scoring new jobs for relevance …")
    try:
        from scorer import run_scoring

        def _do_scoring():
            return run_scoring(profile=profile, resume_map=resume_map)

        score_stats = retry_with_backoff(_do_scoring, max_retries=3, backoff_base=5)
        if score_stats.get("skipped_no_ai", 0) == -1:
            print("  Scoring skipped — AI API key not configured.")
        else:
            print(
                f"  Scoring complete — scored: {score_stats['scored']}, "
                f"skipped (low): {score_stats['skipped_low']}, "
                f"errors: {score_stats['errors']}"
            )
            logger.info(
                "Scoring complete: scored=%d skipped_low=%d errors=%d",
                score_stats["scored"], score_stats["skipped_low"], score_stats["errors"],
            )
    except Exception as exc:
        logger.error("Phase 2 (scoring) failed: %s", exc)
        print(f"  WARNING: Scoring failed ({exc}). Continuing with unscored jobs.")
    print()

    # ------------------------------------------------------------------
    # Phase 3 — Resume Router & Tailoring
    # ------------------------------------------------------------------
    print("  [Phase 3] Routing resumes and tailoring for scored jobs …")
    try:
        from resume_router import route as router_route
        from scorer import create_ai_client as _create_ai_client

        ai_cfg = profile.get("ai", {})
        scoring_model = ai_cfg.get("scoring_model", "gpt-4o-mini")
        tailoring_model = ai_cfg.get("tailoring_model", "gpt-4o")
        ai_client = _create_ai_client(profile)

        with db.get_connection() as conn:
            scored_jobs = conn.execute(
                "SELECT id, title, company, description, matched_resume "
                "FROM jobs WHERE status = 'scoring' "
                "ORDER BY relevance_score DESC"
            ).fetchall()

        p3_stats = {"routed": 0, "tailored": 0, "fallback": 0, "skipped": 0}

        for job in scored_jobs:
            job_id = job["id"]
            company = job["company"]
            description = job["description"] or ""

            if not description.strip():
                p3_stats["skipped"] += 1
                continue

            try:
                original_path, tailored_path = retry_with_backoff(
                    lambda: router_route(
                        job_description=description,
                        company=company,
                        resume_map=resume_map,
                        client=ai_client,
                        scoring_model=scoring_model,
                        tailoring_model=tailoring_model,
                        skill_root=SKILL_ROOT,
                    ),
                    max_retries=3,
                    backoff_base=5,
                )
            except Exception as exc:
                logger.error("Phase 3 routing failed for job %s: %s", job_id, exc)
                p3_stats["skipped"] += 1
                continue

            if not original_path:
                p3_stats["skipped"] += 1
                continue

            use_resume = tailored_path or original_path
            with db.get_connection() as conn:
                conn.execute(
                    "UPDATE jobs SET matched_resume = ?, updated_at = datetime('now') WHERE id = ?",
                    (use_resume, job_id),
                )
                if tailored_path and not dry_run:
                    conn.execute(
                        "INSERT OR IGNORE INTO applications "
                        "(job_id, resume_path, tailored_path, status) "
                        "VALUES (?, ?, ?, 'pending_submit')",
                        (job_id, original_path, tailored_path),
                    )

            p3_stats["routed"] += 1
            if tailored_path:
                p3_stats["tailored"] += 1
                logger.info("Tailored resume saved for %s → %s", company, Path(tailored_path).name)
            else:
                p3_stats["fallback"] += 1

        print(
            f"  Resume routing complete — routed: {p3_stats['routed']}, "
            f"tailored: {p3_stats['tailored']}, "
            f"fallback (original): {p3_stats['fallback']}, "
            f"skipped: {p3_stats['skipped']}"
        )
        logger.info("Phase 3 complete: %s", p3_stats)
    except Exception as exc:
        logger.error("Phase 3 (resume routing) failed: %s", exc)
        print(f"  WARNING: Resume routing failed ({exc}). Continuing with existing resume assignments.")
    print()

    # ------------------------------------------------------------------
    # Phase 4 — Application Engine
    # ------------------------------------------------------------------
    print("  [Phase 4] Dispatching applications …")
    try:
        from dispatcher import run_dispatch
        from scorer import create_ai_client as _create_ai_client

        ai_client = _create_ai_client(profile)
        p4_stats = run_dispatch(
            profile=profile,
            dry_run=dry_run,
            client=ai_client,
            rate_limit_seconds=0 if dry_run else 150,
        )
        print(
            f"  Dispatch complete — auto: {p4_stats['auto']}, "
            f"queued: {p4_stats['queued']}, "
            f"skipped: {p4_stats['skipped']}, "
            f"notify: {p4_stats['notify']}, "
            f"errors: {p4_stats['errors']}"
        )
        logger.info("Phase 4 complete: %s", p4_stats)
    except Exception as exc:
        logger.error("Phase 4 (dispatch) failed: %s", exc)
        print(f"  WARNING: Application dispatch failed ({exc}). Jobs remain in their current state.")
        ai_client = None
    print()

    # ------------------------------------------------------------------
    # Phase 5 — Notifications
    # ------------------------------------------------------------------
    print("  [Phase 5] Sending daily digest and polling replies …")
    try:
        from notifier import send_digest, poll_replies

        sent = send_digest(profile=profile)
        if sent:
            print("  Digest email sent.")
            logger.info("Digest email sent.")
        else:
            print("  Digest written to data/logs/ (Gmail not configured or unavailable).")
            logger.info("Digest fallback: written to logs.")

        if not dry_run:
            if ai_client is None:
                from scorer import create_ai_client as _create_ai_client
                ai_client = _create_ai_client(profile)
            reply_stats = poll_replies(profile=profile, client=ai_client)
            print(
                f"  Reply poll — replies: {reply_stats['replies_found']}, "
                f"answers stored: {reply_stats['answers_stored']}, "
                f"jobs re-queued: {reply_stats['jobs_requeued']}"
            )
            logger.info("Phase 5 reply poll: %s", reply_stats)
    except Exception as exc:
        logger.error("Phase 5 (notifications) failed: %s", exc)
        print(f"  WARNING: Notification step failed ({exc}).")
    print()

    updated = db.get_daily_stats()
    logger.info("Run finished. Total jobs tracked: %d", updated["total_jobs_tracked"])
    print(f"  Run complete. Total jobs tracked: {updated['total_jobs_tracked']}")
    print(f"  Run 'python3 main.py status' for today's summary.")


def _validate_config(profile: dict, resume_map: dict) -> None:
    """Validate required config fields and warn about missing resume files."""
    personal = profile.get("personal", {})
    if not personal.get("name"):
        print("  WARNING: name not configured. Run 'python3 main.py setup'")
    if not personal.get("email"):
        print("  WARNING: email not configured. Run 'python3 main.py setup'")

    prefs = profile.get("job_preferences", {})
    if not prefs.get("titles"):
        print("  WARNING: no job titles configured. Run 'python3 main.py setup'")

    missing_resumes = []
    domains = resume_map.get("domains", {})
    for key, val in domains.items():
        path = val.get("resume", "")
        if path and not Path(path).exists():
            missing_resumes.append(f"{val['label']}: {path}")
    default = resume_map.get("default", "")
    if default and not Path(default).exists():
        missing_resumes.append(f"default: {default}")

    if missing_resumes:
        print("  WARNING: The following resume files were not found:")
        for m in missing_resumes:
            print(f"    {m}")
        print(f"  Place resume files in: {SKILL_ROOT / 'data' / 'resumes'}/")

    ai_cfg = profile.get("ai", {})
    key_env = ai_cfg.get("mammoth_api_key_env", "MAMMOTH_API_KEY")
    if not os.environ.get(key_env):
        print(f"  WARNING: {key_env} not set. AI features will not work.")
        print("  Set it in your shell or re-run setup to store it.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-search",
        description="Job Search Automation System",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    p_setup = sub.add_parser("setup", help="First-run configuration wizard")
    p_setup.add_argument("--reconfigure-gmail", action="store_true",
                         help="Re-run only the Gmail OAuth setup")
    p_setup.add_argument("--from-json", metavar="PATH",
                         help="Non-interactive setup: write config from a pre-built JSON file "
                              "(used by chatbot workflows)")
    p_setup.set_defaults(func=cmd_setup)

    # parse-resume
    p_parse = sub.add_parser(
        "parse-resume",
        help="Extract structured profile info from a resume PDF/DOCX/TXT (for chatbot setup)"
    )
    p_parse.add_argument("resume_path", help="Path to resume file (PDF, DOCX, or TXT)")
    p_parse.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    p_parse.set_defaults(func=cmd_parse_resume)

    # run
    p_run = sub.add_parser("run", help="Run the full job search pipeline")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Go through the full flow but do not submit applications")
    p_run.add_argument("--verbose", action="store_true",
                       help="Enable debug-level logging")
    p_run.set_defaults(func=cmd_run)

    # status
    p_status = sub.add_parser("status", help="Show today's activity summary")
    p_status.set_defaults(func=cmd_status)

    # queue
    p_queue = sub.add_parser("queue", help="List all queued jobs needing review")
    p_queue.set_defaults(func=cmd_queue)

    # apply
    p_apply = sub.add_parser("apply", help="Manually trigger apply for a specific job")
    p_apply.add_argument("job_id", help="Job ID from the queue list")
    p_apply.add_argument("--dry-run", action="store_true",
                         help="Fill form but do not submit")
    p_apply.set_defaults(func=cmd_apply)

    # skip
    p_skip = sub.add_parser("skip", help="Mark a job as skipped")
    p_skip.add_argument("job_id", help="Job ID to skip")
    p_skip.set_defaults(func=cmd_skip)

    # notify
    p_notify = sub.add_parser("notify", help="Send the daily digest email and poll for replies")
    p_notify.add_argument("--no-poll", action="store_true",
                          help="Send digest only; skip polling Gmail for replies")
    p_notify.set_defaults(func=cmd_notify)

    # qa
    p_qa = sub.add_parser("qa", help="Manage the Q&A store")
    qa_sub = p_qa.add_subparsers(dest="qa_command", required=True)
    p_qa_add = qa_sub.add_parser("add", help="Add a new Q&A entry")
    p_qa_add.set_defaults(func=cmd_qa_add)
    p_qa_list = qa_sub.add_parser("list", help="List all Q&A entries")
    p_qa_list.set_defaults(func=cmd_qa_list)

    # scheduler
    p_sched = sub.add_parser("scheduler", help="Manage the macOS launchd daily scheduler")
    sched_sub = p_sched.add_subparsers(dest="scheduler_command")
    sched_sub.add_parser("install", help="Install and load the launchd plist")
    sched_sub.add_parser("uninstall", help="Unload and remove the launchd plist")
    sched_sub.add_parser("status", help="Show scheduler install/load status")
    sched_sub.add_parser("qnap", help="Print QNAP NAS SSH cron setup guide")
    p_sched.set_defaults(func=cmd_scheduler)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
