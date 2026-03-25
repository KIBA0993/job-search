"""
First-run onboarding wizard.
Collects user profile, resume paths, Gmail config, and AI API key.
Writes config/user_profile.json and config/resume_map.json.
Seeds the Q&A store with standard answers.
"""

import json
import os
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).parent.parent
CONFIG_DIR = SKILL_ROOT / "config"
RESUMES_DIR = SKILL_ROOT / "data" / "resumes"

# Import db after setting up path
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
import db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = "", required: bool = False) -> str:
    """Prompt user for input with optional default."""
    display = f"{prompt}"
    if default:
        display += f" [{default}]"
    display += ": "
    while True:
        val = input(display).strip()
        if not val and default:
            return default
        if not val and required:
            print("  This field is required.")
            continue
        return val or default


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no answer."""
    hint = "Y/n" if default else "y/N"
    val = input(f"{prompt} [{hint}]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes")


def ask_int(prompt: str, default: int = 0, min_val: int = 0) -> int:
    """Prompt for integer input."""
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            n = int(val.replace(",", ""))
            if n < min_val:
                print(f"  Must be at least {min_val}.")
                continue
            return n
        except ValueError:
            print("  Please enter a number.")


def ask_list(prompt: str, example: str = "") -> list:
    """Prompt for comma-separated list."""
    hint = f" (e.g. {example})" if example else ""
    val = input(f"{prompt}{hint}: ").strip()
    if not val:
        return []
    return [item.strip() for item in val.split(",") if item.strip()]


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Section collectors
# ---------------------------------------------------------------------------

def collect_personal(existing: dict) -> dict:
    section("PERSONAL INFORMATION")
    p = existing.get("personal", {})
    return {
        "name": ask("Full name", p.get("name", ""), required=True),
        "email": ask("Email address", p.get("email", ""), required=True),
        "phone": ask("Phone number", p.get("phone", "")),
        "linkedin_url": ask("LinkedIn URL", p.get("linkedin_url", "")),
        "portfolio_url": ask("Portfolio / website URL", p.get("portfolio_url", "")),
        "github_url": ask("GitHub URL", p.get("github_url", "")),
        "city": ask("City", p.get("city", "")),
        "state": ask("State", p.get("state", "")),
        "country": ask("Country", p.get("country", "United States")),
    }


def collect_job_preferences(existing: dict) -> dict:
    section("JOB PREFERENCES")
    j = existing.get("job_preferences", {})
    titles = ask_list(
        "Job titles to search for",
        "Software Engineer, Senior Engineer",
    ) or j.get("titles", [])
    locations = ask_list(
        "Preferred locations",
        "San Francisco CA, Remote",
    ) or j.get("locations", [])
    remote_opts = {"1": "remote", "2": "hybrid", "3": "onsite", "4": "flexible"}
    print("  Remote preference: 1=Remote  2=Hybrid  3=Onsite  4=Flexible")
    remote_input = ask("  Choose", "4")
    remote = remote_opts.get(remote_input, "flexible")

    salary_min = ask_int("Minimum salary (USD, 0 = skip)", j.get("salary_min", 0))
    salary_pref = ask_int("Preferred salary (USD, 0 = skip)", j.get("salary_preferred", 0))

    return {
        "titles": titles,
        "locations": locations,
        "remote": remote,
        "salary_min": salary_min,
        "salary_preferred": salary_pref,
        "salary_currency": "USD",
        "industry": [],
        "company_size": [],
        "company_type": [],
    }


def collect_application_defaults(existing: dict) -> dict:
    section("APPLICATION DEFAULTS")
    a = existing.get("application_defaults", {})
    work_auth = ask_yes_no(
        "Are you legally authorized to work in the United States?",
        a.get("work_authorized", True),
    )
    sponsorship = ask_yes_no(
        "Will you now or in the future require visa sponsorship (e.g. H-1B)?",
        a.get("visa_sponsorship_required", False),
    )
    notice = ask_int("Notice period (days)", a.get("notice_period_days", 14))
    years_exp = ask_int("Years of total work experience", a.get("years_of_experience", 0))
    return {
        "notice_period_days": notice,
        "work_authorized": work_auth,
        "visa_sponsorship_required": sponsorship,
        "referral": False,
        "years_of_experience": years_exp,
    }


def collect_resume_map(existing_map: dict) -> dict:
    section("RESUME CONFIGURATION")
    print("  Place your resume files in:")
    print(f"    {RESUMES_DIR}")
    print()

    domains = existing_map.get("domains", {})
    print("  Available domain slots:")
    for i, (key, val) in enumerate(domains.items(), 1):
        current = val.get("resume", "")
        print(f"    {i}. {val['label']}: {current or '(not set)'}")
    print()

    print("  Enter resume file paths for each domain.")
    print("  Press Enter to skip a domain. Use absolute paths or filenames in data/resumes/")
    print()

    for key, val in domains.items():
        current = val.get("resume", "")
        path_input = ask(f"  Resume for '{val['label']}'", current)
        if path_input:
            if not os.path.isabs(path_input):
                candidate = RESUMES_DIR / path_input
                path_input = str(candidate) if candidate.exists() else path_input
            domains[key]["resume"] = path_input

    default_resume = existing_map.get("default", "")
    default_input = ask("Default resume (used when domain doesn't match)", default_resume)
    if default_input and not os.path.isabs(default_input):
        candidate = RESUMES_DIR / default_input
        default_input = str(candidate) if candidate.exists() else default_input

    existing_map["domains"] = domains
    existing_map["default"] = default_input
    return existing_map


def collect_automation_settings(existing: dict) -> dict:
    section("AUTOMATION SETTINGS")
    a = existing.get("automation", {})
    max_apps = ask_int("Max applications per day (recommended: 15–20)", a.get("max_applications_per_day", 20))
    threshold = float(ask("Relevance threshold to attempt applying (0.0–1.0)", str(a.get("relevance_threshold", 0.40))))
    schedule_hour = ask_int("Hour to run daily (24h format)", a.get("schedule_hour", 9))
    return {
        **a,
        "max_applications_per_day": max_apps,
        "relevance_threshold": threshold,
        "schedule_hour": schedule_hour,
    }


def collect_notifications(existing: dict) -> dict:
    section("DAILY DIGEST (GMAIL)")
    n = existing.get("notifications", {})
    print("  The system sends you a daily email with: applied jobs, queue, unanswered questions.")
    print("  You'll need to set up Gmail API credentials (free) — see README.md for instructions.")
    print()
    gmail = ask("Gmail address for daily digest", n.get("gmail_digest_address", ""))
    return {
        **n,
        "gmail_digest_address": gmail,
        "gmail_credentials_path": "config/.gmail_credentials.json",
        "gmail_token_path": "config/.gmail_token.json",
    }


def collect_ai_config(existing: dict) -> dict:
    section("AI CONFIGURATION (MAMMOTH AI)")
    ai = existing.get("ai", {})
    print("  The system uses Mammoth AI for job scoring, resume tailoring, and Q&A matching.")
    print("  Your API key can be set here or via the MAMMOTH_API_KEY environment variable.")
    print()
    current_key = os.environ.get("MAMMOTH_API_KEY", "")
    if current_key:
        print(f"  MAMMOTH_API_KEY already set in environment (length: {len(current_key)}). Press Enter to keep.")

    key_input = ask("Mammoth AI API key (leave blank to use env var)", "")
    if key_input:
        env_file = SKILL_ROOT / "config" / ".env"
        with open(env_file, "w") as f:
            f.write(f"MAMMOTH_API_KEY={key_input}\n")
        print(f"  API key written to {env_file} (keep this file private)")
        ai["mammoth_api_key_env"] = "MAMMOTH_API_KEY"

    api_base = ask("Mammoth AI API base URL", ai.get("mammoth_api_base", "https://api.mammoth.ai/v1"))
    return {
        **ai,
        "mammoth_api_base": api_base,
    }


def collect_platforms(existing: dict) -> list:
    section("PLATFORMS TO SEARCH")
    all_platforms = [
        ("linkedin", "LinkedIn"),
        ("indeed", "Indeed"),
        ("glassdoor", "Glassdoor"),
        ("ziprecruiter", "ZipRecruiter"),
        ("wellfound", "Wellfound (AngelList)"),
        ("trueup", "TrueUp"),
        ("otta", "Otta"),
        ("workatstartup", "Work at a Startup"),
        ("underdog", "Underdog.io"),
        ("welcometothejungle", "Welcome to the Jungle"),
    ]
    current = set(existing.get("platforms", [p[0] for p in all_platforms]))
    print("  Toggle platforms on/off (Enter platform numbers separated by commas to disable).")
    print()
    for i, (key, label) in enumerate(all_platforms, 1):
        status = "ON " if key in current else "OFF"
        print(f"    {i:2}. [{status}] {label}")
    print()
    disable_input = ask("  Enter numbers to DISABLE (e.g. 3,7) or press Enter to keep all", "")
    if disable_input:
        try:
            to_disable = {int(x.strip()) - 1 for x in disable_input.split(",") if x.strip()}
            for i, (key, _) in enumerate(all_platforms):
                if i in to_disable:
                    current.discard(key)
        except ValueError:
            print("  Invalid input, keeping current selection.")
    return list(current)


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------

def run(reconfigure_gmail: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  JOB SEARCH AUTOMATION — FIRST-RUN SETUP")
    print("=" * 60)
    print("\n  This wizard configures the automation system.")
    print("  Press Ctrl+C at any time to quit without saving.\n")

    profile_path = CONFIG_DIR / "user_profile.json"
    resume_map_path = CONFIG_DIR / "resume_map.json"

    existing_profile = load_config(profile_path) if profile_path.exists() else {}
    existing_map = load_config(resume_map_path) if resume_map_path.exists() else {"domains": {}, "default": ""}

    try:
        # Collect all sections
        profile = dict(existing_profile)
        profile["personal"] = collect_personal(existing_profile)
        profile["job_preferences"] = collect_job_preferences(existing_profile)
        profile["application_defaults"] = collect_application_defaults(existing_profile)
        profile["automation"] = collect_automation_settings(existing_profile)
        profile["platforms"] = collect_platforms(existing_profile)
        profile["notifications"] = collect_notifications(existing_profile)
        profile["ai"] = collect_ai_config(existing_profile)

        resume_map = collect_resume_map(existing_map)

        # Remove comment keys before saving
        profile.pop("_comment", None)
        resume_map.pop("_comment", None)

        section("REVIEW & SAVE")
        print(f"  Name    : {profile['personal']['name']}")
        print(f"  Email   : {profile['personal']['email']}")
        print(f"  Titles  : {', '.join(profile['job_preferences']['titles'])}")
        print(f"  Salary  : ${profile['job_preferences']['salary_preferred']:,} preferred")
        print(f"  Platforms: {len(profile['platforms'])} enabled")
        print()
        if ask_yes_no("Save configuration?", True):
            save_config(profile_path, profile)
            save_config(resume_map_path, resume_map)

            # Initialize database and seed Q&A
            db.init_db()
            db.seed_qa_store(profile)

            print()
            print("  Setup complete!")
            print(f"  Run the system: python3 {SKILL_ROOT}/scripts/main.py run --dry-run")
        else:
            print("  Cancelled. No changes saved.")

    except KeyboardInterrupt:
        print("\n\n  Setup cancelled.")
        sys.exit(0)


def run_from_json(json_path: str) -> None:
    """
    Non-interactive setup driven by a pre-built JSON config file.
    Used by chatbot workflows. Writes user_profile.json, resume_map.json,
    initialises DB, and seeds Q&A store.

    The JSON file must have the same shape as user_profile.json + a top-level
    "resume_map" key. Unknown keys are ignored. Missing keys fall back to
    their defaults.

    Example minimal JSON:
    {
      "personal": {"name": "Simon Chen", "email": "simon@example.com", ...},
      "job_preferences": {"titles": ["Software Engineer"], "locations": ["Remote"], ...},
      "application_defaults": {"work_authorized": true, "visa_sponsorship_required": false, ...},
      "automation": {"max_applications_per_day": 20, "relevance_threshold": 0.4, "schedule_hour": 9},
      "platforms": ["linkedin", "indeed", ...],
      "notifications": {"gmail_digest_address": "simon@gmail.com"},
      "ai": {"mammoth_api_base": "https://api.mammoth.ai/v1"},
      "resume_map": {
        "domains": { ... },
        "default": "/path/to/default.pdf"
      }
    }
    """
    import json as _json

    path = Path(json_path).expanduser()
    if not path.exists():
        print(f"  Error: config file not found: {path}")
        import sys; sys.exit(1)

    with open(path) as f:
        data = _json.load(f)

    profile_path = CONFIG_DIR / "user_profile.json"
    resume_map_path = CONFIG_DIR / "resume_map.json"

    # Load existing configs as base (so partial JSONs only override what's provided)
    existing_profile = load_config(profile_path) if profile_path.exists() else {}
    existing_map = load_config(resume_map_path) if resume_map_path.exists() else {"domains": {}, "default": ""}

    # Deep-merge: top-level section keys from data override existing
    profile = dict(existing_profile)
    for section_key in ("personal", "job_preferences", "application_defaults",
                        "automation", "platforms", "notifications", "ai"):
        if section_key in data:
            if isinstance(data[section_key], dict):
                profile[section_key] = {**(profile.get(section_key) or {}), **data[section_key]}
            else:
                profile[section_key] = data[section_key]

    resume_map = dict(existing_map)
    if "resume_map" in data:
        rm = data["resume_map"]
        if "domains" in rm:
            resume_map["domains"] = {**(resume_map.get("domains") or {}), **rm["domains"]}
        if "default" in rm:
            resume_map["default"] = rm["default"]

    profile.pop("_comment", None)
    resume_map.pop("_comment", None)

    save_config(profile_path, profile)
    save_config(resume_map_path, resume_map)

    db.init_db()
    db.seed_qa_store(profile)

    print()
    print("  Setup complete (non-interactive)!")
    print(f"  Run: python3 {SKILL_ROOT}/scripts/main.py run --dry-run")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconfigure-gmail", action="store_true")
    parser.add_argument("--from-json", help="Non-interactive setup from a JSON config file")
    args = parser.parse_args()
    if args.from_json:
        run_from_json(args.from_json)
    else:
        run(reconfigure_gmail=args.reconfigure_gmail)
