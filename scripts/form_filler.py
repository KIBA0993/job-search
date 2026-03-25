"""
Phase 4: bb-browser form-filling orchestrator.

Fills online job application forms by:
  1. Opening the application URL in bb-browser
  2. Detecting ATS iframes (Greenhouse, Workday, Lever, iCIMS, Taleo)
  3. Snapshotting interactive elements
  4. Classifying each field and resolving its answer from profile / Q&A store
  5. Applying bot_filter before inserting each text answer
  6. Optionally submitting (or returning DRY_RUN)

Returns a FillResult describing the outcome.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

SKILL_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class FillStatus(Enum):
    SUCCESS = "success"
    DRY_RUN = "dry_run"
    INCOMPLETE = "incomplete"
    ERROR = "error"
    CAPTCHA = "captcha"


@dataclass
class FillResult:
    status: FillStatus
    fields_filled: int = 0
    fields_total: int = 0
    missing_fields: list = field(default_factory=list)
    submitted: bool = False
    error_message: str = ""


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

# Maps field category → list of label keywords that identify it as a profile field
_PROFILE_KEYWORDS: dict[str, list[str]] = {
    "name":       ["full name", "your name", "first and last name", "applicant name"],
    "first_name": ["first name", "given name"],
    "last_name":  ["last name", "family name", "surname"],
    "email":      ["email", "e-mail", "email address"],
    "phone":      ["phone", "telephone", "mobile", "cell phone", "phone number"],
    "linkedin":   ["linkedin", "linkedin url", "linkedin profile"],
    "website":    ["website", "portfolio", "personal website", "portfolio url"],
    "location":   ["city", "location", "current location", "where are you located"],
    "resume":     ["resume", "cv", "curriculum vitae", "upload resume", "attach resume", "upload cv"],
    "cover_letter": ["cover letter", "upload cover letter", "attach cover letter"],
}

# Maps Q&A category → list of label keywords that identify it as a Q&A field
_QA_KEYWORDS: dict[str, list[str]] = {
    "salary":           ["salary", "compensation", "expected salary", "desired salary", "pay expectation"],
    "work_authorization": ["authorized to work", "work authorization", "eligible to work", "right to work"],
    "sponsorship":      ["sponsorship", "visa sponsorship", "require sponsorship", "work visa"],
    "notice_period":    ["notice period", "start date", "available to start", "when can you start"],
    "referral":         ["referred by", "referral", "how did you hear", "source of application"],
    "relocation":       ["relocation", "willing to relocate", "open to relocate"],
    "remote":           ["remote work", "work remotely", "prefer remote", "remote preference"],
    "experience_years": ["years of experience", "how many years", "experience level"],
}

_ESSAY_INDICATORS = [
    "why do you want", "tell us about", "describe your", "what makes you",
    "cover letter", "additional information", "anything else", "motivat",
    "why are you interested", "explain",
]


def classify_field(label: str) -> tuple[str, str]:
    """
    Classify a field label into (category, source).

    source is one of: 'profile', 'resume_upload', 'essay', 'qa_store'
    """
    low = label.lower().strip()

    for category, keywords in _PROFILE_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            if category == "resume":
                return ("resume", "resume_upload")
            return (category, "profile")

    for category, keywords in _QA_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return (category, "qa_store")

    # Heuristic for essay / long-form fields
    if any(ind in low for ind in _ESSAY_INDICATORS) or len(label) >= 150:
        return ("essay", "essay")

    return ("unknown", "qa_store")


def _split_name(name: str, part: int) -> str:
    """Split 'First Last' → 'First' (0) or 'Last' (1)."""
    parts = name.strip().split(None, 1)
    return parts[part] if len(parts) > part else name


def resolve_field_answer(
    category: str,
    source: str,
    label: str,
    profile: dict,
    qa_matcher: Optional[Callable] = None,
) -> Optional[str]:
    """
    Return the best available answer for a form field, or None if unknown.

    qa_matcher(label) → Optional[dict] — result from question_handler.find_best_match
    """
    personal = profile.get("personal", profile)

    if source == "profile":
        mapping = {
            "name":         personal.get("name", ""),
            "first_name":   _split_name(personal.get("name", ""), 0),
            "last_name":    _split_name(personal.get("name", ""), 1),
            "email":        personal.get("email", ""),
            "phone":        personal.get("phone", ""),
            "linkedin":     personal.get("linkedin_url", ""),
            "website":      personal.get("portfolio_url", ""),
            "location":     personal.get("location", ""),
            "cover_letter": "",   # upload handled separately
        }
        value = mapping.get(category, "")
        return value if value else None

    if source == "resume_upload":
        return None   # handled as file-upload, not text

    if source == "essay":
        return None   # always routes to QUEUE

    # source == "qa_store"
    if qa_matcher:
        match = qa_matcher(label)
        if match:
            return match.get("answer")

    return None


# ---------------------------------------------------------------------------
# ATS iframe detection
# ---------------------------------------------------------------------------

_ATS_SIGNALS: dict[str, list[str]] = {
    "greenhouse":  ["greenhouse.io", "boards.greenhouse.io", "grnhse_app"],
    "workday":     ["myworkdayjobs.com", "workday.com", "workday-widget"],
    "lever":       ["jobs.lever.co", "lever-apply"],
    "icims":       ["icims.com"],
    "taleo":       ["taleo.net"],
    "smartrecruiters": ["smartrecruiters.com"],
    "jobvite":     ["jobvite.com"],
}


def detect_ats(snapshot: str) -> Optional[str]:
    """
    Return the ATS name detected from a page snapshot, or None.
    """
    low = snapshot.lower()
    for ats, signals in _ATS_SIGNALS.items():
        if any(sig in low for sig in signals):
            return ats
    return None


# ---------------------------------------------------------------------------
# Snapshot parsing
# ---------------------------------------------------------------------------

def parse_interactive_elements(snapshot: str) -> list[dict]:
    """
    Parse `bb-browser snapshot -i` output into a list of element dicts.

    Expected line format (from bb-browser):
      [@N] input[type=text] "Label text" placeholder="..."
      [@N] textarea "Label text"
      [@N] select "Label text"
      [@N] input[type=file]
      [@N] button "Submit"
    """
    elements = []
    for line in snapshot.splitlines():
        line = line.strip()
        if not line.startswith("[@"):
            continue
        elem = _parse_element_line(line)
        if elem:
            elements.append(elem)
    return elements


def _parse_element_line(line: str) -> Optional[dict]:
    """Parse a single snapshot element line into a dict."""
    ref_m = re.match(r"\[@(\d+)\]", line)
    if not ref_m:
        return None
    ref = f"@{ref_m.group(1)}"

    low = line.lower()
    if "input[type=file]" in low or ("file" in low and "input" in low):
        elem_type = "file"
    elif "textarea" in low:
        elem_type = "textarea"
    elif "select" in low:
        elem_type = "select"
    elif "checkbox" in low or "input[type=checkbox]" in low:
        elem_type = "checkbox"
    elif "radio" in low or "input[type=radio]" in low:
        elem_type = "radio"
    elif ("button" in low and "submit" in low) or "type=submit" in low:
        elem_type = "submit"
    elif "input[type=email]" in low or ("email" in low and "input" in low):
        elem_type = "email"
    elif "input[type=tel]" in low or ("tel" in low and "input" in low):
        elem_type = "tel"
    elif "input[type=number]" in low or ("number" in low and "input" in low):
        elem_type = "number"
    else:
        elem_type = "text"

    label_m = re.search(r'"([^"]+)"', line)
    label = label_m.group(1) if label_m else ""

    ph_m = re.search(r'placeholder="([^"]*)"', line, re.IGNORECASE)
    placeholder = ph_m.group(1) if ph_m else ""

    return {
        "ref": ref,
        "type": elem_type,
        "label": label or placeholder,
        "placeholder": placeholder,
    }


def _check_submission_success(snapshot: str) -> bool:
    """Heuristic: does the page look like a confirmation / thank-you page?"""
    if not snapshot:
        return False
    indicators = [
        "thank you",
        "application submitted",
        "application received",
        "successfully applied",
        "we've received your application",
        "your application has been",
        "confirmation number",
        "application complete",
    ]
    low = snapshot.lower()
    return any(ind in low for ind in indicators)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fill_application(
    job: dict,
    resume_path: str,
    profile: dict,
    dry_run: bool = False,
    bb=None,
    client=None,
    bot_filter_fn: Optional[Callable] = None,
) -> FillResult:
    """
    Fill a job application form using bb-browser.

    Args:
        job:           dict with keys id, url, title, company
        resume_path:   path to the PDF/DOCX resume to upload
        profile:       user_profile dict
        dry_run:       fill all fields but do not click Submit
        bb:            BrowserCmd instance; a fresh one is created if None
        client:        Mammoth AI client for Q&A matching
        bot_filter_fn: callable(text) → str applied to every text answer

    Returns:
        FillResult
    """
    from search import BrowserCmd
    import bot_filter as _bot_filter
    import question_handler as qh

    if bb is None:
        bb = BrowserCmd(timeout=60)

    if bot_filter_fn is None:
        bot_filter_fn = lambda t: _bot_filter.scan_and_strip(t, llm_client=client)

    job_url = job.get("url", "")
    company = job.get("company", "unknown")
    logger.info("Filling application: %s @ %s", job.get("title", "?"), company)

    bb.open(job_url)
    bb.wait(3000)

    # Get an initial snapshot to detect ATS iframes
    snapshot_raw = bb.run("snapshot", "--quiet", ignore_errors=True)
    if not snapshot_raw:
        return FillResult(status=FillStatus.ERROR, error_message="Could not snapshot page")

    ats = detect_ats(snapshot_raw)
    if ats:
        logger.info("ATS detected: %s — switching to iframe context", ats)
        bb.run("frame", "iframe", ignore_errors=True)
        bb.wait(1500)
        snapshot_raw = bb.run("snapshot", "--quiet", ignore_errors=True) or snapshot_raw

    # CAPTCHA check
    if any(kw in snapshot_raw.lower() for kw in ["captcha", "i am not a robot", "recaptcha", "hcaptcha"]):
        logger.warning("CAPTCHA detected on %s", job_url)
        bb.run("close", ignore_errors=True)
        return FillResult(status=FillStatus.CAPTCHA, error_message="CAPTCHA detected")

    # Get interactive elements
    snapshot_i = bb.run("snapshot", "-i", ignore_errors=True)
    if not snapshot_i:
        bb.run("close", ignore_errors=True)
        return FillResult(status=FillStatus.ERROR, error_message="No interactive snapshot")

    elements = parse_interactive_elements(snapshot_i)
    if not elements:
        bb.run("close", ignore_errors=True)
        return FillResult(status=FillStatus.ERROR, error_message="No interactive elements found")

    def qa_matcher(label: str):
        return qh.find_best_match(label, client=client)

    missing: list[str] = []
    filled = 0

    for elem in elements:
        elem_type = elem["type"]
        label = elem["label"]
        ref = elem["ref"]

        if elem_type == "submit":
            continue

        category, source = classify_field(label)

        # --- Resume / file upload ---
        if source == "resume_upload":
            if resume_path and Path(resume_path).exists():
                bb.run("upload", ref, resume_path, ignore_errors=True)
                filled += 1
            else:
                missing.append(label or "resume upload")
            continue

        # --- Essay → always route to queue ---
        if source == "essay":
            missing.append(label[:80] if label else "essay field")
            continue

        answer = resolve_field_answer(
            category=category,
            source=source,
            label=label,
            profile=profile,
            qa_matcher=qa_matcher if source == "qa_store" else None,
        )
        if answer is None:
            missing.append(label[:80] or f"field_{ref}")
            continue

        # Apply bot filter to text answers
        answer = bot_filter_fn(answer)
        if not answer:
            missing.append(label[:80] or f"field_{ref}")
            continue

        # Fill based on element type
        if elem_type in ("text", "textarea", "email", "tel", "number"):
            bb.run("fill", ref, answer, ignore_errors=True)
            filled += 1
        elif elem_type == "select":
            bb.run("select", ref, answer, ignore_errors=True)
            filled += 1
        elif elem_type == "checkbox":
            if answer.lower() in ("yes", "true", "1"):
                bb.run("check", ref, ignore_errors=True)
                filled += 1
        elif elem_type == "radio":
            bb.run("click", ref, ignore_errors=True)
            filled += 1

    result = FillResult(
        status=FillStatus.SUCCESS,   # updated below if incomplete / dry-run / error
        fields_filled=filled,
        fields_total=len(elements),
        missing_fields=missing,
    )

    if missing:
        result.status = FillStatus.INCOMPLETE
        bb.run("close", ignore_errors=True)
        return result

    if dry_run:
        result.status = FillStatus.DRY_RUN
        bb.run("close", ignore_errors=True)
        return result

    # Submit
    bb.run("click", "@submit", ignore_errors=True)
    bb.wait(3000)

    after_snapshot = bb.run("snapshot", "--quiet", ignore_errors=True)
    submitted = _check_submission_success(after_snapshot)
    bb.run("close", ignore_errors=True)

    if submitted:
        result.status = FillStatus.SUCCESS
        result.submitted = True
    else:
        result.status = FillStatus.ERROR
        result.error_message = "Submission confirmation not detected"

    return result
