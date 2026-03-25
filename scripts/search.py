"""
Phase 2: Job discovery via bb-browser platform adapters.

For each enabled platform:
  1. Build search URL from user profile criteria
  2. Open with bb-browser
  3. Extract job listings via JavaScript eval (CSS selectors from platform_config.json)
  4. Deduplicate using URL hash + title hash
  5. Store new jobs in SQLite with status 'new'

bb-browser must be installed and accessible in PATH.
"""

import hashlib
import json
import logging
import re
import subprocess
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SKILL_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RawJob:
    platform: str
    title: str
    company: str
    url: str
    description: str = ""
    posted_date: Optional[str] = None


@dataclass
class DiscoveryResult:
    platform: str
    found: int = 0
    new: int = 0
    errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hash / normalization utilities
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Strip tracking parameters and normalize URL for deduplication."""
    url = url.strip()
    try:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        _TRACKING = {
            "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
            "trackingId", "refId", "ref", "src", "trk", "trkCampaign",
            "originalSubdomain", "viewGenericContent",
        }
        clean_params = {k: v for k, v in params.items() if k not in _TRACKING}
        clean_query = urllib.parse.urlencode(clean_params, doseq=True)
        normalized = urllib.parse.urlunparse(
            parsed._replace(query=clean_query, fragment="", scheme=parsed.scheme.lower())
        )
        return normalized.lower()
    except Exception:
        return url.lower()


def compute_url_hash(url: str) -> str:
    """SHA-256 of normalized URL."""
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()


def normalize_title(company: str, title: str) -> str:
    """Normalize company + title for cross-platform deduplication."""
    company = re.sub(r"\s+", " ", company.strip().lower())
    t = title.lower()
    # Strip abbreviated seniority tokens like "Sr." / "Jr." (with optional period + space)
    t = re.sub(r"\b(sr|jr)\.?\s*", "", t, flags=re.IGNORECASE)
    # Strip word-form seniority modifiers
    t = re.sub(r"\b(senior|junior|lead|principal|staff|associate)\b\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t.strip())
    return f"{company}:{t}"


def compute_title_hash(company: str, title: str) -> str:
    """SHA-256 of normalized company:title string."""
    return hashlib.sha256(normalize_title(company, title).encode()).hexdigest()


# ---------------------------------------------------------------------------
# bb-browser subprocess wrapper
# ---------------------------------------------------------------------------

class BrowserCmd:
    """Thin subprocess wrapper around the bb-browser CLI."""

    def __init__(self, timeout: int = 45):
        self.timeout = timeout

    def run(self, *args, ignore_errors: bool = False, timeout: int = None) -> str:
        """Run a bb-browser command and return stdout. Returns '' on failure."""
        cmd = ["bb-browser"] + [str(a) for a in args]
        _timeout = timeout if timeout is not None else self.timeout
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_timeout,
            )
            if result.returncode != 0 and not ignore_errors:
                logger.debug(
                    "bb-browser %s → exit %d: %s",
                    " ".join(args[:2]), result.returncode, result.stderr[:200],
                )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            if not ignore_errors:
                logger.debug("bb-browser %s timed out after %ds", " ".join(args[:2]), _timeout)
            return ""
        except FileNotFoundError:
            logger.error("bb-browser not found in PATH. Install it first.")
            return ""

    def open(self, url: str) -> str:
        return self.run("open", url)

    def close(self) -> None:
        self.run("close", ignore_errors=True)

    def wait(self, ms: int = 2000) -> None:
        self.run("wait", str(ms), ignore_errors=True, timeout=max(10, ms // 1000 + 5))

    def scroll(self, direction: str = "down", pixels: int = 800) -> None:
        """Scroll the page — silently ignored if the page doesn't respond."""
        self.run("scroll", direction, str(pixels), ignore_errors=True, timeout=8)

    def eval(self, js: str) -> str:
        return self.run("eval", js)

    def eval_json(self, js: str) -> Optional[list]:
        """Execute JS that returns JSON.stringify output; parse and return or None."""
        raw = self.eval(js)
        if not raw:
            return None
        # bb-browser may wrap output in quotes or add extra whitespace
        raw = raw.strip()
        # If it looks like a JSON string (starts/ends with "), unescape it
        if raw.startswith('"') and raw.endswith('"'):
            try:
                raw = json.loads(raw)  # unescape JSON string → get inner JSON
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("eval_json: could not parse response: %s", raw[:300])
            return None


# ---------------------------------------------------------------------------
# JavaScript extractor builder
# ---------------------------------------------------------------------------

def _esc_js_str(s: str) -> str:
    """Escape a string for safe embedding inside a JS single-quoted string literal."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def build_extractor_js(platform_cfg: dict) -> str:
    """
    Return a JavaScript IIFE that extracts job listings from the current page.
    Uses CSS selectors from platform_config.json. Returns JSON.stringify output.
    """
    list_sel = _esc_js_str(platform_cfg.get("job_list_selector", ""))
    title_sel = _esc_js_str(platform_cfg.get("job_title_selector", ""))
    company_sel = _esc_js_str(platform_cfg.get("job_company_selector", ""))
    url_sel = _esc_js_str(platform_cfg.get("job_url_selector", ""))
    date_sel = _esc_js_str(platform_cfg.get("job_date_selector") or "")

    date_expr = (
        f"(el.querySelector('{date_sel}')?.innerText || '').trim()"
        if date_sel else "''"
    )

    return (
        "(function(){"
        "try{"
        f"var els=Array.from(document.querySelectorAll('{list_sel}')).slice(0,60);"
        "return JSON.stringify(els.map(function(el){"
        f"var uEl=el.querySelector('{url_sel}');"
        "var href=uEl?(uEl.href||uEl.getAttribute('href')||''):'';"
        f"return{{title:(el.querySelector('{title_sel}')?.innerText||'').trim(),"
        f"company:(el.querySelector('{company_sel}')?.innerText||'').trim(),"
        f"url:href,date:{date_expr}}};"
        "}).filter(function(j){return j.title&&j.company&&j.url;}));"
        "}catch(e){return '[]';}"
        "})()"
    )


# ---------------------------------------------------------------------------
# Search URL builder
# ---------------------------------------------------------------------------

def build_search_url(template: str, criteria: dict) -> str:
    """Fill URL placeholders from criteria dict.

    Supported placeholders:
      {title}         — URL-encoded job title
      {location}      — URL-encoded location
      {title_slug}    — kebab-case job title
      {location_slug} — kebab-case location
      {remote}        — "true" if remote, else ""
      {salary_min}    — integer minimum salary (0 if not set)
    """
    title = criteria.get("title", "software engineer")
    location = criteria.get("location", "")
    remote = criteria.get("remote", "flexible")
    salary_min = int(criteria.get("salary_min") or 0)
    replacements = {
        "{title}": urllib.parse.quote_plus(title),
        "{location}": urllib.parse.quote_plus(location),
        "{title_slug}": re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"),
        "{location_slug}": re.sub(r"[^a-z0-9]+", "-", location.lower()).strip("-"),
        "{remote}": "true" if remote == "remote" else "",
        "{salary_min}": str(salary_min),
    }
    url = template
    for placeholder, value in replacements.items():
        url = url.replace(placeholder, value)
    return url


# ---------------------------------------------------------------------------
# Platform adapter
# ---------------------------------------------------------------------------

class PlatformAdapter:
    """
    Generic adapter for a single job platform, driven entirely by
    CSS selectors and URL patterns from platform_config.json.
    """

    def __init__(self, platform_id: str, config: dict):
        self.platform_id = platform_id
        self.config = config
        self.rate_limit_s = config.get("rate_limit_ms", 2000) / 1000.0

    # Phrases that indicate a login wall is blocking the page
    _LOGIN_WALL_PHRASES = (
        "sign in", "log in", "create an account", "join now",
        "register to", "please login", "please log in",
    )

    def _is_login_wall(self, browser: BrowserCmd) -> bool:
        """Return True if the page has redirected to a login/sign-up wall.

        Only blocks on confirmed redirects (login keywords in page title or URL).
        Does NOT block on missing login-indicator elements — those selectors go
        stale as sites redesign, and would create false positives when the user
        IS logged in.
        """
        page_title = browser.eval("(document.title || '').trim()").strip().strip('"')
        current_url = browser.eval("window.location.href").strip().strip('"')
        lower_title = page_title.lower()
        lower_url = current_url.lower()

        for phrase in self._LOGIN_WALL_PHRASES:
            if phrase in lower_title:
                logger.warning(
                    "[%s] Redirected to login page (title: %r). "
                    "Open %s in Chrome and log in, then re-run.",
                    self.platform_id, page_title,
                    self.config.get("base_url", "this site"),
                )
                return True

        # Only flag URL-based redirects when the path explicitly contains a login token
        # AND we've left the site's expected domain.
        login_url_tokens = ("login", "signin", "sign-in", "/auth/", "/register",
                            "accounts.google", "account.linkedin")
        for token in login_url_tokens:
            if token in lower_url:
                logger.warning(
                    "[%s] Redirected to auth URL (%s). "
                    "Open %s in Chrome and log in, then re-run.",
                    self.platform_id, current_url[:100],
                    self.config.get("base_url", "this site"),
                )
                return True

        return False

    def search(self, criteria: dict, max_pages: int = None) -> tuple[list, list]:
        """
        Open platform search for the given criteria, extract all job listings.
        Returns (list[RawJob], list[error_str]).
        """
        jobs: list[RawJob] = []
        errors: list[str] = []

        max_pages = max_pages if max_pages is not None else self.config.get("max_pages", 3)
        search_url = build_search_url(self.config["search_url"], criteria)
        # Honour per-platform page-load wait (SPA sites need more time)
        page_load_wait = self.config.get("page_load_wait_ms", 3000)
        browser = BrowserCmd(timeout=60)

        try:
            logger.info("[%s] Searching: %s", self.platform_id, search_url)
            browser.open(search_url)
            browser.wait(page_load_wait)

            if self._is_login_wall(browser):
                errors.append(f"{self.platform_id}: login wall — not logged in")
                return jobs, errors

            js = build_extractor_js(self.config)
            seen_urls: set[str] = set()

            for page_num in range(1, max_pages + 1):
                # Scroll to trigger lazy-loaded job cards before extracting
                browser.scroll("down", 600)
                browser.wait(1200)
                browser.scroll("down", 600)
                browser.wait(800)

                raw = browser.eval_json(js)

                if not raw:
                    page_title = browser.eval("document.title").strip().strip('"')
                    _login_words = ("sign in", "log in", "login", "sign up",
                                    "register", "create account", "accounts")
                    if any(w in page_title.lower() for w in _login_words):
                        logger.warning(
                            "[%s] Page %d redirected to login page ('%s'). "
                            "Log into this site in Chrome and re-run.",
                            self.platform_id, page_num, page_title,
                        )
                    else:
                        logger.warning(
                            "[%s] Page %d: 0 jobs extracted. Page title: %r. "
                            "CSS selector may be outdated — check platform_config.json.",
                            self.platform_id, page_num, page_title,
                        )
                    break

                new_on_page = 0
                for item in raw:
                    url = (item.get("url") or "").strip()
                    title = (item.get("title") or "").strip()
                    company = (item.get("company") or "").strip()
                    if not url or not title or not company:
                        continue
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    jobs.append(RawJob(
                        platform=self.platform_id,
                        title=title,
                        company=company,
                        url=url,
                        posted_date=(item.get("date") or "").strip() or None,
                    ))
                    new_on_page += 1

                logger.debug("[%s] Page %d: %d jobs", self.platform_id, page_num, new_on_page)

                if page_num >= max_pages:
                    break

                # Try to advance to next page
                next_sel = self.config.get("next_page_selector")
                if not next_sel:
                    break

                next_js = (
                    f"(function(){{var b=document.querySelector('{_esc_js_str(next_sel)}');"
                    f"if(b&&!b.disabled){{b.click();return true;}}return false;}})()"
                )
                clicked = browser.eval(next_js)
                if clicked.strip() not in ("true", "True"):
                    break

                time.sleep(self.rate_limit_s)
                browser.wait(2500)

        except Exception as exc:
            msg = f"{self.platform_id}: {exc}"
            errors.append(msg)
            logger.error("[%s] Search error: %s", self.platform_id, exc)
        finally:
            browser.close()

        return jobs, errors


# ---------------------------------------------------------------------------
# Job description fetcher
# ---------------------------------------------------------------------------

_DESCRIPTION_SELECTORS = [
    ".jobs-description__content",      # LinkedIn
    ".jobDescriptionText",              # Indeed
    "#job-description",
    "[data-testid='job-description']",
    ".job-description",
    ".description__text",
    ".jobsearch-jobDescriptionText",
    "article",
]


def fetch_job_description(url: str, browser: Optional[BrowserCmd] = None) -> str:
    """Navigate to a job URL and extract the description text (up to 5 000 chars)."""
    own_browser = browser is None
    if own_browser:
        browser = BrowserCmd(timeout=30)
    try:
        browser.open(url)
        browser.wait(2000)
        for sel in _DESCRIPTION_SELECTORS:
            js = f"(document.querySelector('{_esc_js_str(sel)}')?.innerText||'').trim()"
            text = browser.eval(js)
            if text and len(text) > 150:
                return text[:5000]
        # Fallback: grab main/article content
        fallback_js = (
            "(document.querySelector('main,article,[role=\"main\"]')"
            "?.innerText||'').trim().substring(0,5000)"
        )
        return browser.eval(fallback_js) or ""
    except Exception as exc:
        logger.warning("fetch_job_description(%s): %s", url[:60], exc)
        return ""
    finally:
        if own_browser:
            browser.close()


# ---------------------------------------------------------------------------
# Discovery runner
# ---------------------------------------------------------------------------

def run_discovery(
    profile: dict,
    platform_cfg: dict,
    db_path=None,
    fetch_descriptions: bool = True,
    max_pages_override: int = None,
) -> dict:
    """
    Run job search on all enabled platforms using the user's profile criteria.
    Inserts new jobs into the database.

    Returns:
        {platform_id: DiscoveryResult}
    """
    import db as _db  # local import to avoid circular deps in tests

    if db_path is None:
        db_path = _db.DB_PATH

    prefs = profile.get("job_preferences", {})
    titles = prefs.get("titles", [])
    locations = prefs.get("locations", [""])
    remote = prefs.get("remote", "flexible")
    salary_min = int(prefs.get("salary_min") or 0)
    enabled_platforms = profile.get("platforms", list(platform_cfg["platforms"].keys()))

    if not titles:
        logger.warning("No job titles configured — skipping discovery.")
        return {}

    # Use at most one primary location per run (extend later for multi-location)
    primary_location = locations[0] if locations else ""

    results: dict[str, DiscoveryResult] = {}
    desc_browser = BrowserCmd(timeout=30)

    for platform_id in enabled_platforms:
        pcfg = platform_cfg.get("platforms", {}).get(platform_id)
        if not pcfg or not pcfg.get("enabled", True):
            logger.debug("Skipping disabled platform: %s", platform_id)
            continue

        p_result = DiscoveryResult(platform=platform_id)
        adapter = PlatformAdapter(platform_id=platform_id, config=pcfg)

        # Cap at 3 titles per platform per run to keep runtime reasonable
        for title in titles[:3]:
            criteria = {
                "title": title,
                "location": primary_location,
                "remote": remote,
                "salary_min": salary_min,
            }
            raw_jobs, errors = adapter.search(
                criteria, max_pages=max_pages_override
            )
            p_result.errors.extend(errors)

            for raw_job in raw_jobs:
                p_result.found += 1

                if not raw_job.title or not raw_job.company or not raw_job.url:
                    continue

                url_hash = compute_url_hash(raw_job.url)
                title_hash = compute_title_hash(raw_job.company, raw_job.title)

                if _db.job_exists(url_hash, title_hash, db_path):
                    continue

                description = raw_job.description
                if fetch_descriptions and not description and raw_job.url:
                    description = fetch_job_description(raw_job.url, desc_browser)
                    time.sleep(0.3)

                job_id = f"{platform_id}:{uuid.uuid4().hex[:12]}"
                job_record = {
                    "id": job_id,
                    "platform": platform_id,
                    "title": raw_job.title,
                    "company": raw_job.company,
                    "url": raw_job.url,
                    "description": description,
                    "posted_date": raw_job.posted_date,
                    "url_hash": url_hash,
                    "title_hash": title_hash,
                }

                if _db.insert_job(job_record, db_path):
                    p_result.new += 1
                    logger.info(
                        "[%s] + %s @ %s", platform_id, raw_job.title, raw_job.company
                    )

            time.sleep(pcfg.get("rate_limit_ms", 2000) / 1000.0)

        results[platform_id] = p_result
        logger.info(
            "[%s] done — found %d, new %d, errors %d",
            platform_id, p_result.found, p_result.new, len(p_result.errors),
        )

    return results
