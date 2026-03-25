#!/usr/bin/env python3
"""
scheduler.py — macOS launchd job installer and QNAP NAS cron guide.

Manages the com.jobsearch.daily.plist LaunchAgent so the pipeline runs
automatically at a configurable hour each day.

Public API
----------
install(skill_root, schedule_hour, python_executable)
    Write the plist to ~/Library/LaunchAgents/ and load it via launchctl.

uninstall()
    Unload and remove the plist from ~/Library/LaunchAgents/.

status() -> dict
    Return {'installed': bool, 'loaded': bool, 'plist_path': str}.

get_plist_content(skill_root, schedule_hour, python_executable) -> str
    Return the plist XML as a string (without writing to disk).
"""

import logging
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PLIST_LABEL = "com.jobsearch.daily"
PLIST_FILENAME = f"{PLIST_LABEL}.plist"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


# ---------------------------------------------------------------------------
# Plist generation
# ---------------------------------------------------------------------------

def get_plist_content(
    skill_root: Path,
    schedule_hour: int = 9,
    python_executable: Optional[str] = None,
) -> str:
    """Return the launchd plist XML string for the daily job-search run."""
    if python_executable is None:
        import sys
        python_executable = sys.executable

    main_py = str(skill_root / "scripts" / "main.py")
    log_dir = str(skill_root / "data" / "logs")

    # Ensure the log dir path uses forward slashes (macOS standard)
    stdout_path = f"{log_dir}/launchd_stdout.log"
    stderr_path = f"{log_dir}/launchd_stderr.log"

    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{PLIST_LABEL}</string>

            <key>ProgramArguments</key>
            <array>
                <string>{python_executable}</string>
                <string>{main_py}</string>
                <string>run</string>
            </array>

            <key>StartCalendarInterval</key>
            <dict>
                <key>Hour</key>
                <integer>{schedule_hour}</integer>
                <key>Minute</key>
                <integer>0</integer>
            </dict>

            <key>StandardOutPath</key>
            <string>{stdout_path}</string>

            <key>StandardErrorPath</key>
            <string>{stderr_path}</string>

            <key>RunAtLoad</key>
            <false/>

            <key>EnvironmentVariables</key>
            <dict>
                <key>PATH</key>
                <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
            </dict>
        </dict>
        </plist>
        """)


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

def install(
    skill_root: Path,
    schedule_hour: int = 9,
    python_executable: Optional[str] = None,
) -> dict:
    """
    Write the plist and load it with launchctl.

    Returns
    -------
    {'success': bool, 'plist_path': str, 'message': str}
    """
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    plist_path = LAUNCH_AGENTS_DIR / PLIST_FILENAME
    content = get_plist_content(skill_root, schedule_hour, python_executable)

    # Ensure log dir exists
    log_dir = skill_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write plist
    plist_path.write_text(content)
    logger.info("Plist written to %s", plist_path)

    # Unload first if already loaded (idempotent re-install)
    _launchctl_unload(plist_path, ignore_errors=True)

    # Load
    result = _launchctl_load(plist_path)
    if result["success"]:
        msg = (
            f"Scheduler installed and loaded.\n"
            f"  Plist: {plist_path}\n"
            f"  Runs daily at {schedule_hour:02d}:00\n"
            f"  Logs: {skill_root / 'data' / 'logs'}"
        )
        logger.info(msg)
    else:
        msg = f"Plist written but launchctl load failed: {result['error']}"
        logger.warning(msg)

    return {
        "success": result["success"],
        "plist_path": str(plist_path),
        "message": msg,
    }


def uninstall() -> dict:
    """
    Unload and remove the plist.

    Returns
    -------
    {'success': bool, 'message': str}
    """
    plist_path = LAUNCH_AGENTS_DIR / PLIST_FILENAME

    if not plist_path.exists():
        return {
            "success": True,
            "message": "Scheduler was not installed (plist not found).",
        }

    unload_result = _launchctl_unload(plist_path, ignore_errors=True)
    plist_path.unlink()

    if unload_result["success"]:
        msg = f"Scheduler unloaded and plist removed: {plist_path}"
    else:
        msg = f"Plist removed but unload warning: {unload_result.get('error', '')}"

    logger.info(msg)
    return {"success": True, "message": msg}


def status() -> dict:
    """
    Return the current status of the launchd job.

    Returns
    -------
    {
        'installed': bool,      # plist file exists on disk
        'loaded': bool,         # launchctl list shows the label
        'plist_path': str,
        'next_run': str | None, # e.g. "daily at 09:00" parsed from plist
    }
    """
    plist_path = LAUNCH_AGENTS_DIR / PLIST_FILENAME
    installed = plist_path.exists()
    loaded = False
    next_run = None

    if installed:
        # Try to detect schedule_hour from the plist
        try:
            content = plist_path.read_text()
            # Simple text scan for <integer>N</integer> after <key>Hour</key>
            import re
            m = re.search(r"<key>Hour</key>\s*<integer>(\d+)</integer>", content)
            if m:
                next_run = f"daily at {int(m.group(1)):02d}:00"
        except Exception:
            pass

    # Check if the label appears in `launchctl list`
    try:
        result = subprocess.run(
            ["launchctl", "list", PLIST_LABEL],
            capture_output=True,
            text=True,
            timeout=10,
        )
        loaded = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        loaded = False

    return {
        "installed": installed,
        "loaded": loaded,
        "plist_path": str(plist_path),
        "next_run": next_run,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _launchctl_load(plist_path: Path) -> dict:
    try:
        result = subprocess.run(
            ["launchctl", "load", str(plist_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return {"success": True}
        return {"success": False, "error": result.stderr.strip() or result.stdout.strip()}
    except FileNotFoundError:
        return {"success": False, "error": "launchctl not found (not macOS?)"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "launchctl load timed out"}


def _launchctl_unload(plist_path: Path, ignore_errors: bool = False) -> dict:
    try:
        result = subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 or ignore_errors:
            return {"success": True}
        return {"success": False, "error": result.stderr.strip()}
    except FileNotFoundError:
        if ignore_errors:
            return {"success": True}
        return {"success": False, "error": "launchctl not found"}
    except subprocess.TimeoutExpired:
        if ignore_errors:
            return {"success": True}
        return {"success": False, "error": "launchctl unload timed out"}


# ---------------------------------------------------------------------------
# QNAP NAS cron guide (returned as a string for printing / docs)
# ---------------------------------------------------------------------------

QNAP_CRON_GUIDE = """
QNAP NAS — SSH Cron Fallback Configuration
===========================================

Use this if you want the job search to run on your QNAP NAS instead of (or
in addition to) your MacBook.  The QNAP approach uses a standard cron job
over SSH.

Prerequisites
-------------
1.  Enable SSH on the NAS: Control Panel → Network Services → SSH → Enable SSH.
2.  Install Python 3 on the NAS via QNAP App Center (search "Python 3") or
    Entware: `opkg install python3`.
3.  Clone or copy the job-search skill to the NAS:
      scp -r ~/.agents/skills/job-search/ admin@<NAS_IP>:/share/homes/admin/job-search/
4.  Install Python dependencies on the NAS:
      ssh admin@<NAS_IP>
      cd /share/homes/admin/job-search
      pip3 install -r requirements.txt   # or pip3 install openai google-api-python-client ...

Add the Cron Job
----------------
On the NAS, edit the crontab:

    ssh admin@<NAS_IP>
    crontab -e

Add one of the following lines:

  # Run daily at 09:00 NAS local time
  0 9 * * * /opt/bin/python3 /share/homes/admin/job-search/scripts/main.py run >> /share/homes/admin/job-search/data/logs/cron.log 2>&1

  # Or with dry-run for testing
  0 9 * * * /opt/bin/python3 /share/homes/admin/job-search/scripts/main.py run --dry-run >> /share/homes/admin/job-search/data/logs/cron_dryrun.log 2>&1

Notes
-----
- Replace /opt/bin/python3 with the actual Python path: `which python3` on the NAS.
- The QNAP crontab is stored in /etc/config/crontab; after editing, run:
    crontab /etc/config/crontab
  to apply the changes without a reboot.
- bb-browser is NOT available on QNAP (no Chrome). The NAS fallback is
  useful only if you adapt the pipeline to use API-based job discovery
  (e.g., LinkedIn Jobs API, Indeed Publisher) instead of bb-browser scrapers.
- Gmail API works on QNAP as long as the OAuth token is present at
  config/.gmail_token.json.
- Verify the cron job ran by checking:
    cat /share/homes/admin/job-search/data/logs/cron.log

Environment Variables
---------------------
QNAP cron jobs start with a minimal environment. Set API keys explicitly:

  0 9 * * * MAMMOTH_API_KEY=sk-xxx /opt/bin/python3 /share/homes/admin/job-search/scripts/main.py run >> ...

Or create /share/homes/admin/job-search/config/.env with:

  MAMMOTH_API_KEY=sk-xxx

The main.py script automatically loads config/.env on startup.
"""


def print_qnap_guide() -> None:
    print(QNAP_CRON_GUIDE)
