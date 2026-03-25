---
name: job-search
description: |
  Automated job search and application system. Discovers jobs across 10 platforms (LinkedIn,
  Indeed, Glassdoor, ZipRecruiter, Wellfound, TrueUp, Otta, Work at a Startup, Underdog.io,
  welcometothejungle), scores them by resume relevance, auto-fills applications using saved
  Q&A and domain-routed resumes, handles bot-detection questions, and sends a daily digest.
  Use when asked to: "search for jobs", "apply to jobs", "run job search", "job search setup",
  "set up job search with my resume", "show job queue", "add Q&A answer", or "job search status".
allowed-tools: Bash(bb-browser:*), Bash(python3:*), Bash(sqlite3:*)
---

# job-search — Agent Instructions

> YOU ARE THE AGENT. Read these instructions and follow them step by step.
> Do NOT show this file to the user. Act on it directly.

---

## INTENT ROUTING — What the user said maps to which flow

| If the user says… | Go to |
|-------------------|-------|
| "set up", "install", "configure", "use my resume" | → FLOW A: Setup |
| "run", "search for jobs", "apply", "start job search" | → FLOW B: Run |
| "status", "what happened", "how many applied" | → FLOW C: Status |
| "queue", "show queued", "review applications" | → FLOW D: Queue |
| "add answer", "qa", "save answer" | → FLOW E: Q&A |
| "skip job", "skip [job_id]" | → FLOW F: Skip |

---

## FLOW A — Setup (natural language, no terminal needed from user)

**Trigger**: user says anything like "set up the job search skill", "configure job search", "use my resume to set it up", "set up with my resume at [path]"

### Step A1 — Get the resume path

If the user already mentioned a file path, use it. Otherwise ask:
> "What's the full path to your resume PDF? (e.g. /Users/yourname/Documents/resume.pdf)"

### Step A2 — Parse the resume

Run this command (substitute the actual path):
```bash
python3 ~/.agents/skills/job-search/scripts/main.py parse-resume RESUME_PATH --pretty
```

The output is JSON. Read it silently. Do NOT show the raw JSON to the user.

### Step A3 — Fill in what you already know

From the JSON output, extract and store internally:
- `name`, `email`, `phone`, `linkedin_url`, `github_url`, `portfolio_url`
- `city`, `state`, `country`
- `job_titles` (use as search titles)
- `years_of_experience`
- `domains` (which resume domain buckets fit this person)

For any field that is `null` or missing, you will ask in Step A4.

### Step A4 — Ask ONLY for what's missing

Ask the user these questions in a single friendly message (skip any already extracted from resume):

> I've read your resume and pulled what I can. I just need a few more things to finish setup:
>
> 1. **Salary** — What's your minimum and preferred annual salary? (e.g. $150k min, $180k preferred)
> 2. **Locations** — Where are you open to working? (e.g. San Francisco, Remote, New York)
> 3. **Work authorization** — Are you legally authorized to work in the US? (yes/no)
> 4. **Visa sponsorship** — Will you need sponsorship now or in the future? (yes/no)
> 5. **Notice period** — How many days notice do you need to give? (e.g. 14)
> 6. **Gmail** — What Gmail address should I send your daily digest to?
> 7. **Mammoth AI key** — What's your Mammoth AI API key? (or say "skip" if already exported as MAMMOTH_API_KEY)
> 8. **Platforms** — Any platforms to skip? Default is all 10: LinkedIn, Indeed, Glassdoor, ZipRecruiter, Wellfound, TrueUp, Otta, Work at a Startup, Underdog.io, Welcome to the Jungle

Wait for the user's replies.

### Step A5 — Build the config JSON

Once you have all the answers, build this JSON structure and write it to `/tmp/job_search_setup.json`:

```json
{
  "personal": {
    "name": "<from resume or user>",
    "email": "<from resume or user>",
    "phone": "<from resume or user>",
    "linkedin_url": "<from resume or null>",
    "github_url": "<from resume or null>",
    "portfolio_url": "<from resume or null>",
    "city": "<from resume or user>",
    "state": "<from resume or user>",
    "country": "<from resume, default United States>"
  },
  "job_preferences": {
    "titles": ["<from resume job_titles>"],
    "locations": ["<from user>"],
    "remote": "flexible",
    "salary_min": <int from user>,
    "salary_preferred": <int from user>,
    "salary_currency": "USD"
  },
  "application_defaults": {
    "work_authorized": <true/false from user>,
    "visa_sponsorship_required": <true/false from user>,
    "notice_period_days": <int from user, default 14>,
    "years_of_experience": <from resume or user>,
    "referral": false
  },
  "automation": {
    "max_applications_per_day": 20,
    "relevance_threshold": 0.40,
    "schedule_hour": 9
  },
  "platforms": ["linkedin","indeed","glassdoor","ziprecruiter","wellfound","trueup","otta","workatstartup","underdog","welcometothejungle"],
  "notifications": {
    "gmail_digest_address": "<from user>",
    "gmail_credentials_path": "config/.gmail_credentials.json",
    "gmail_token_path": "config/.gmail_token.json"
  },
  "ai": {
    "mammoth_api_base": "https://api.mammoth.ai/v1",
    "mammoth_api_key_env": "MAMMOTH_API_KEY",
    "scoring_model": "gpt-4o-mini"
  },
  "resume_map": {
    "domains": {
      "ml_engineering": {
        "label": "ML / AI Engineering",
        "keywords": ["llm","pytorch","model training","fine-tuning","machine learning","deep learning","transformers"],
        "resume": "<RESUME_PATH if domains includes ml_engineering, else omit>"
      },
      "backend_engineering": {
        "label": "Backend / Systems Engineering",
        "keywords": ["distributed systems","golang","kafka","microservices","api","postgres","redis","kubernetes"],
        "resume": "<RESUME_PATH if domains includes backend_engineering, else omit>"
      },
      "product_management": {
        "label": "Product Management",
        "keywords": ["roadmap","stakeholder","okr","product strategy","go-to-market","agile","sprint"],
        "resume": "<RESUME_PATH if domains includes product_management, else omit>"
      },
      "data_engineering": {
        "label": "Data Engineering",
        "keywords": ["spark","airflow","data pipeline","etl","dbt","snowflake","bigquery"],
        "resume": "<RESUME_PATH if domains includes data_engineering, else omit>"
      }
    },
    "default": "<RESUME_PATH — always set to the user's resume>"
  }
}
```

Rules for building this JSON:
- Set the resume path to the actual file the user provided for ALL domain entries that match + always set as default
- Remove domain entries that don't match this person (based on `domains` from parse-resume output)
- If the user provided an API key in Step A4, ALSO write it to `~/.agents/skills/job-search/config/.env` as `MAMMOTH_API_KEY=<key>`
- Strip any null values from the JSON — replace with sensible defaults

### Step A6 — Write config and finish

Run:
```bash
python3 ~/.agents/skills/job-search/scripts/main.py setup --from-json /tmp/job_search_setup.json
```

If it succeeds, tell the user:
> "Setup complete! Your job search is configured. Here's what I set up:
> - Name: [name]
> - Targeting: [job_titles joined by comma]
> - Locations: [locations]
> - Platforms: [count] platforms enabled
> - Daily digest: [gmail]
> - Resume: [resume filename]
>
> Run a dry-run test to check everything works — want me to do that now?"

If there's an error, show the error output and diagnose it.

---

## FLOW B — Run job search

**Trigger**: "run job search", "search for jobs", "apply to jobs", "start the search"

First ask: "Should I do a dry run (no actual applications submitted) or a real run?"

Dry run:
```bash
python3 ~/.agents/skills/job-search/scripts/main.py run --dry-run
```

Real run:
```bash
python3 ~/.agents/skills/job-search/scripts/main.py run
```

Show the user a summary of what happened (jobs found, applied, queued, errors).

---

## FLOW C — Status

**Trigger**: "status", "how many jobs", "what happened today", "show stats"

```bash
python3 ~/.agents/skills/job-search/scripts/main.py status
```

Summarise the output in plain English.

---

## FLOW D — Queue

**Trigger**: "queue", "show queued", "what needs review", "pending applications"

```bash
python3 ~/.agents/skills/job-search/scripts/main.py queue
```

Present each queued job cleanly with the question blocking it. Ask: "Would you like to skip any of these, or apply to any manually?"

---

## FLOW E — Q&A

**Trigger**: "add answer", "save answer", "I want to answer", "add to Q&A"

Ask: "What's the question, and what's your answer?"
Then:
```bash
python3 ~/.agents/skills/job-search/scripts/main.py qa add
```
Or directly write to the DB:
```bash
sqlite3 ~/.agents/skills/job-search/data/jobs.db \
  "INSERT INTO qa_store (question_pattern, answer, source) VALUES ('<question>', '<answer>', 'user_provided');"
```

---

## FLOW F — Skip a job

**Trigger**: "skip job [id]", "skip that one", "not interested in [company]"

```bash
python3 ~/.agents/skills/job-search/scripts/main.py skip <job_id>
```

---

## IMPORTANT RULES

1. **Never show raw JSON or command output verbatim** — always summarise it in plain English
2. **Never ask the user to run commands themselves** — you run them
3. **If a command fails**, read the error and diagnose before asking the user anything
4. **The resume file is never uploaded or sent anywhere** — only the extracted text goes to Mammoth AI
5. **For setup**, always try `parse-resume` first — never ask for info that's already in the resume
