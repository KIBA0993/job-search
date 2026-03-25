"""
Phase 3: Resume domain classification and AI-powered tailoring.

Given a job description:
  1. classify_domain()  — ask Mammoth AI which domain best matches the JD;
                          fall back to 'default' when confidence is below threshold.
  2. get_resume_for_job() — pick the configured resume for that domain.
  3. tailor_resume()    — ask Mammoth AI to reorder / keyword-enhance the resume
                          without fabricating experience, without dashes.
  4. save_tailored_pdf() — render the tailored text to a PDF and save it under
                           data/tailored/{CompanyName}_{YYYY-MM-DD}.pdf
  5. route()            — convenience function that ties steps 1–4 together and
                          returns (original_resume_path, tailored_pdf_path_or_None).

Fallback contract
-----------------
If the AI client is unavailable, classification returns 'default'.
If tailoring fails (timeout, bad response, empty text), the original matched
resume is used and no PDF is written; the caller receives tailored_path=None.
"""

import logging
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

# Word-boundary pattern cache: kw → compiled regex
_KW_PATTERNS: dict[str, re.Pattern] = {}

SKILL_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = (
    "You are a recruitment domain classifier.\n"
    "Given the following job description, choose the single best-fitting domain "
    "from this list: {domains}.\n"
    "Rules:\n"
    "  - Return ONLY the domain key, exactly as listed (e.g. 'ml_engineering').\n"
    "  - If none of the domains match well, return the word 'default'.\n"
    "  - Do NOT explain your answer. Just the key.\n\n"
    "Job description:\n{jd}"
)


def classify_domain(
    job_description: str,
    resume_map: dict,
    client=None,
    model: str = "gpt-4o-mini",
    min_confidence: float = 0.0,  # unused directly; kept for API compatibility
) -> str:
    """
    Return the best-matching domain key from resume_map['domains'], or 'default'.

    Strategy:
      1. Keyword count: tally keyword hits per domain.
      2. If any domain has >= 2 keyword hits (clear winner), return it without
         an AI call — this saves tokens and is fast.
      3. Otherwise call the AI to disambiguate.
      4. If the AI returns an unknown domain key or fails, fall back to 'default'.
    """
    domains: dict = resume_map.get("domains", {})
    if not domains:
        return "default"

    jd_lower = job_description.lower()

    def _kw_matches(kw: str, text: str) -> bool:
        """Case-insensitive whole-word match to avoid 'ui' matching 'required'."""
        kw_lower = kw.lower()
        if kw_lower not in _KW_PATTERNS:
            escaped = re.escape(kw_lower)
            _KW_PATTERNS[kw_lower] = re.compile(r"\b" + escaped + r"\b")
        return bool(_KW_PATTERNS[kw_lower].search(text))

    # --- keyword scoring ---
    counts: dict[str, int] = {}
    for domain_key, domain_val in domains.items():
        keywords = domain_val.get("keywords", [])
        score = sum(1 for kw in keywords if _kw_matches(kw, jd_lower))
        counts[domain_key] = score

    best_key = max(counts, key=lambda k: counts[k])
    best_count = counts[best_key]

    # Check for a clear keyword winner
    sorted_counts = sorted(counts.values(), reverse=True)
    runner_up = sorted_counts[1] if len(sorted_counts) > 1 else 0
    if best_count >= 2 and best_count > runner_up:
        logger.debug(
            "classify_domain: keyword winner '%s' (hits=%d)", best_key, best_count
        )
        return best_key

    # --- AI classification ---
    if client is None:
        # No AI client; use keyword fallback or default
        if best_count >= 1:
            logger.debug(
                "classify_domain: no AI, keyword fallback '%s'", best_key
            )
            return best_key
        return "default"

    domain_keys = list(domains.keys())
    jd_trunc = job_description[:3000]
    prompt = _CLASSIFY_PROMPT.format(domains=domain_keys, jd=jd_trunc)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip().lower()
        # Strip punctuation
        raw = re.sub(r"[^a-z0-9_]", "", raw)
        if raw in domains:
            logger.debug("classify_domain: AI selected '%s'", raw)
            return raw
        if raw == "default":
            return "default"
        logger.debug(
            "classify_domain: AI returned unknown key '%s', falling back", raw
        )
        return "default"
    except Exception as exc:
        logger.warning("classify_domain: AI call failed: %s — using default", exc)
        return "default"


# ---------------------------------------------------------------------------
# Resume selection
# ---------------------------------------------------------------------------

def get_resume_for_domain(domain: str, resume_map: dict) -> str:
    """
    Return the configured resume file path for the given domain.
    Falls back to resume_map['default'] if the domain path is empty or missing.
    Returns '' if nothing is configured.
    """
    domains = resume_map.get("domains", {})
    if domain in domains:
        path = domains[domain].get("resume", "")
        if path:
            return path

    return resume_map.get("default", "")


# ---------------------------------------------------------------------------
# Resume tailoring
# ---------------------------------------------------------------------------

_TAILOR_PROMPT = (
    "You are an expert resume writer. Tailor the resume below for the job "
    "description provided. Follow these strict rules:\n"
    "  1. Reorder bullet points within each role to highlight the most relevant "
    "     experience first.\n"
    "  2. Where the candidate's background supports it, incorporate keywords and "
    "     technologies mentioned in the job description.\n"
    "  3. Do NOT invent, fabricate, or exaggerate any experience, skills, or "
    "     projects that are not already in the original resume.\n"
    "  4. Use only commas, periods, and colons as punctuation in bullet points — "
    "     no dashes, em-dashes, asterisks, or bullet characters.\n"
    "  5. Return ONLY the full tailored resume text. No commentary, no markdown "
    "     fences, no explanations.\n\n"
    "JOB DESCRIPTION:\n{jd}\n\n"
    "ORIGINAL RESUME:\n{resume}"
)


def tailor_resume(
    resume_text: str,
    job_description: str,
    company: str,
    client=None,
    model: str = "gpt-4o",
) -> Optional[str]:
    """
    Call Mammoth AI to produce a tailored version of resume_text for the given JD.

    Returns:
        Tailored resume text (str) on success.
        None if tailoring fails or the AI client is unavailable.
    """
    if not client:
        logger.debug("tailor_resume: no AI client — skipping tailoring for %s", company)
        return None

    if not resume_text.strip() or not job_description.strip():
        logger.debug("tailor_resume: empty resume or JD — skipping for %s", company)
        return None

    resume_trunc = resume_text[:6000]
    jd_trunc = job_description[:3000]
    prompt = _TAILOR_PROMPT.format(jd=jd_trunc, resume=resume_trunc)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        tailored = response.choices[0].message.content.strip()
        if len(tailored) < 50:
            logger.warning(
                "tailor_resume: suspiciously short response (%d chars) for %s",
                len(tailored), company,
            )
            return None
        return tailored
    except Exception as exc:
        logger.warning("tailor_resume: AI call failed for %s: %s", company, exc)
        return None


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _sanitize_company_name(name: str) -> str:
    """Strip characters unsafe for filenames."""
    safe = re.sub(r"[^\w\s-]", "", name).strip()
    safe = re.sub(r"\s+", "_", safe)
    return safe or "Company"


def save_tailored_pdf(
    tailored_text: str,
    company: str,
    skill_root: Path = SKILL_ROOT,
    run_date: Optional[date] = None,
) -> Optional[str]:
    """
    Render tailored_text to a PDF and save it.

    File path: data/tailored/{CompanyName}_{YYYY-MM-DD}.pdf

    Returns the absolute path string on success, None on failure.
    Requires either 'reportlab' or 'weasyprint' to be installed.
    """
    if not tailored_text or not tailored_text.strip():
        return None

    if run_date is None:
        run_date = date.today()

    company_safe = _sanitize_company_name(company)
    date_str = run_date.strftime("%Y-%m-%d")
    filename = f"{company_safe}_{date_str}.pdf"

    tailored_dir = skill_root / "data" / "tailored"
    tailored_dir.mkdir(parents=True, exist_ok=True)
    out_path = tailored_dir / filename

    # Try reportlab first (lighter dependency)
    if _save_with_reportlab(tailored_text, out_path):
        logger.info("save_tailored_pdf: saved %s", out_path)
        return str(out_path)

    # Try weasyprint as fallback
    if _save_with_weasyprint(tailored_text, out_path):
        logger.info("save_tailored_pdf: saved %s (weasyprint)", out_path)
        return str(out_path)

    logger.warning(
        "save_tailored_pdf: neither reportlab nor weasyprint available. "
        "Install one: pip install reportlab  or  pip install weasyprint"
    )
    return None


def _save_with_reportlab(text: str, out_path: Path) -> bool:
    try:
        from reportlab.lib.pagesizes import letter  # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  # type: ignore
        from reportlab.lib.units import inch  # type: ignore

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=letter,
            leftMargin=1 * inch,
            rightMargin=1 * inch,
            topMargin=1 * inch,
            bottomMargin=1 * inch,
        )
        styles = getSampleStyleSheet()
        body_style = styles["BodyText"]
        body_style.fontSize = 10
        body_style.leading = 14

        story = []
        for line in text.split("\n"):
            if line.strip():
                story.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), body_style))
            else:
                story.append(Spacer(1, 6))

        doc.build(story)
        return True
    except ImportError:
        return False
    except Exception as exc:
        logger.warning("_save_with_reportlab failed: %s", exc)
        return False


def _save_with_weasyprint(text: str, out_path: Path) -> bool:
    try:
        from weasyprint import HTML  # type: ignore

        html_lines = []
        for line in text.split("\n"):
            if line.strip():
                escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html_lines.append(f"<p>{escaped}</p>")
            else:
                html_lines.append("<br/>")

        html_body = "\n".join(html_lines)
        html = (
            "<html><body style='font-family:sans-serif;font-size:11pt;"
            "margin:2cm;line-height:1.5'>"
            f"{html_body}</body></html>"
        )
        HTML(string=html).write_pdf(str(out_path))
        return True
    except ImportError:
        return False
    except Exception as exc:
        logger.warning("_save_with_weasyprint failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# High-level router
# ---------------------------------------------------------------------------

def route(
    job_description: str,
    company: str,
    resume_map: dict,
    client=None,
    scoring_model: str = "gpt-4o-mini",
    tailoring_model: str = "gpt-4o",
    skill_root: Path = SKILL_ROOT,
    run_date: Optional[date] = None,
) -> tuple[str, Optional[str]]:
    """
    Full routing pipeline for one job.

    Returns:
        (original_resume_path, tailored_pdf_path)

    tailored_pdf_path is None when:
      - AI is unavailable
      - Tailoring produces empty / suspicious output
      - PDF generation fails (reportlab / weasyprint not installed)

    Callers should use tailored_pdf_path when it is not None; otherwise fall
    back to original_resume_path.
    """
    # Step 1 — classify domain
    domain = classify_domain(
        job_description, resume_map, client=client, model=scoring_model
    )

    # Step 2 — get base resume
    original_path = get_resume_for_domain(domain, resume_map)
    if not original_path:
        logger.warning(
            "route: no resume configured for domain '%s' and no default set", domain
        )
        return ("", None)

    # Step 3 — read resume text for tailoring
    sys.path.insert(0, str(SKILL_ROOT / "scripts"))
    from scorer import read_resume_text  # local import; scorer already loaded in Phase 2

    resume_text = read_resume_text(original_path)
    if not resume_text.strip():
        logger.warning(
            "route: could not read resume text from %s — skipping tailoring", original_path
        )
        return (original_path, None)

    # Step 4 — tailor
    tailored_text = tailor_resume(
        resume_text=resume_text,
        job_description=job_description,
        company=company,
        client=client,
        model=tailoring_model,
    )
    if not tailored_text:
        logger.info("route: tailoring unavailable for %s — using original", company)
        return (original_path, None)

    # Step 5 — save PDF
    tailored_path = save_tailored_pdf(
        tailored_text=tailored_text,
        company=company,
        skill_root=skill_root,
        run_date=run_date,
    )

    return (original_path, tailored_path)
