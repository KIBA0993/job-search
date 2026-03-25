# Job Search Automation System

Automates job discovery and application across 10 platforms. Zero additional cost — runs on your existing Mac + bb-browser + Mammoth AI setup.

## Prerequisites

1. **bb-browser** installed (you already have it)
2. **Python 3.9+** (already available)
3. **Chrome open** and logged into job platforms you want to use
4. **Mammoth AI API key** — export it or add to config
5. **Gmail API credentials** — set up during first-run wizard
6. **Python packages** (install once):

```bash
pip3 install pypdf reportlab google-api-python-client google-auth-oauthlib numpy
```

> `pypdf` is required to read your resume PDF files. Without it the system cannot extract resume text for scoring or tailoring.

## First-Time Setup

```bash
python3 ~/.agents/skills/job-search/scripts/main.py setup
```

The wizard will ask for:
- Your name, email, phone, LinkedIn URL
- Salary expectations (min / preferred)
- Notice period (days)
- Work authorization (yes/no)
- Visa sponsorship needed (yes/no)
- Preferred job titles and locations
- Resume file paths and domain labels
- Gmail address for daily digest
- Mammoth AI API key

## Daily Usage

The system runs automatically via macOS launchd (set up in Phase 6). To run manually:

```bash
# Full run
python3 ~/.agents/skills/job-search/scripts/main.py run

# Test without submitting anything
python3 ~/.agents/skills/job-search/scripts/main.py run --dry-run

# Check what happened today
python3 ~/.agents/skills/job-search/scripts/main.py status
```

## Managing Your Q&A Store

The Q&A store is how the system answers application questions automatically. It's pre-seeded with your standard answers (salary, work auth, etc.) during setup. You can add more:

```bash
# Add an answer manually
python3 ~/.agents/skills/job-search/scripts/main.py qa add

# See all stored answers
python3 ~/.agents/skills/job-search/scripts/main.py qa list
```

When a question can't be answered automatically, it appears in your daily digest email. Reply with:
```
ANSWER: {question_id} | Your answer text here
```
The system will store it and retry the application automatically.

## Resume Setup

Place your resume files in `data/resumes/`:
```
data/resumes/
  ml_engineering.pdf
  backend.pdf
  product.pdf
  general.pdf
```

Then configure `config/resume_map.json` to map domains to files and keywords. During setup the wizard guides you through this interactively.

## Platform Coverage

| Platform | URL | Notes |
|----------|-----|-------|
| LinkedIn | linkedin.com/jobs | Includes Easy Apply |
| Indeed | indeed.com | |
| Glassdoor | glassdoor.com/Job | |
| ZipRecruiter | ziprecruiter.com | |
| Wellfound | wellfound.com | Formerly AngelList |
| TrueUp | trueup.io | |
| Otta | otta.com | |
| Work at a Startup | workatastartup.com | YC companies |
| Underdog.io | underdog.io | |
| welcometothejungle | welcometothejungle.com | |

## Safety Limits

- Max 20 applications per day (configurable, not recommended to raise above 30)
- 2–3 minute random delay between submissions
- Bot-detection honeypot questions are automatically stripped
- CAPTCHA pages are skipped and saved for manual apply
- Resumes are tailored (keywords added) but experience is never fabricated

## Files and Data

All data stays on your Mac. Nothing is sent to external servers except:
- Job descriptions and resume text → Mammoth AI (for scoring/tailoring)
- Daily digest → your Gmail

```
~/.agents/skills/job-search/
  config/           Configuration (edit after setup)
  scripts/          Python source files
  data/
    jobs.db         All jobs, applications, Q&A (SQLite)
    resumes/        Your resume files (you add these)
    tailored/       AI-tailored resumes per company
    logs/           Daily run logs
  tests/            Automated tests
```

## Chatbot Setup (non-interactive)

If you want an AI assistant to drive the setup, it can use two commands instead of the interactive wizard:

**1. Extract info from resume:**
```bash
python3 ~/.agents/skills/job-search/scripts/main.py parse-resume /path/to/resume.pdf --pretty
```
Outputs JSON: name, email, phone, LinkedIn, skills, job titles, domains, years of experience.

**2. Write config non-interactively:**

Build a JSON file like this and pass it to `setup --from-json`:

```json
{
  "personal": {
    "name": "Simon Chen",
    "email": "simon@example.com",
    "phone": "+1-415-555-0100",
    "linkedin_url": "https://linkedin.com/in/simonchen",
    "github_url": "https://github.com/simonchen",
    "portfolio_url": "",
    "city": "San Francisco",
    "state": "CA",
    "country": "United States"
  },
  "job_preferences": {
    "titles": ["Software Engineer", "Senior Software Engineer"],
    "locations": ["San Francisco CA", "Remote"],
    "remote": "flexible",
    "salary_min": 150000,
    "salary_preferred": 180000,
    "salary_currency": "USD"
  },
  "application_defaults": {
    "work_authorized": true,
    "visa_sponsorship_required": false,
    "notice_period_days": 14,
    "years_of_experience": 5,
    "referral": false
  },
  "automation": {
    "max_applications_per_day": 20,
    "relevance_threshold": 0.40,
    "schedule_hour": 9
  },
  "platforms": [
    "linkedin","indeed","glassdoor","ziprecruiter","wellfound",
    "trueup","otta","workatstartup","underdog","welcometothejungle"
  ],
  "notifications": {
    "gmail_digest_address": "simon@gmail.com"
  },
  "ai": {
    "mammoth_api_base": "https://api.mammoth.ai/v1"
  },
  "resume_map": {
    "domains": {
      "ml_engineering": {
        "keywords": ["llm","pytorch","model training"],
        "resume": "/Users/kiba/Documents/Simon_Chen_Resume_MASTER.pdf"
      }
    },
    "default": "/Users/kiba/Documents/Simon_Chen_Resume_MASTER.pdf"
  }
}
```

```bash
python3 ~/.agents/skills/job-search/scripts/main.py setup --from-json /tmp/job_search_setup.json
```

## Troubleshooting

**Platform not returning results**: Make sure Chrome is open and you're logged in to that platform. Run `bb-browser open https://linkedin.com/jobs` and check manually.

**Application failed**: Check `data/logs/run_YYYY-MM-DD.log` for the error. Most failures are CAPTCHA or session expiry — just log back in to that platform in Chrome.

**Gmail not sending**: Re-run `python3 scripts/main.py setup --reconfigure-gmail` to refresh OAuth tokens.
