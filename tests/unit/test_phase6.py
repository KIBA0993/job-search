"""
Phase 6 tests: Scheduler & Hardening.

Covers:
  - scheduler.get_plist_content   — valid XML, correct fields, schedule_hour
  - scheduler.install             — writes plist, calls launchctl load
  - scheduler.uninstall           — calls launchctl unload, removes plist
  - scheduler.status              — parses installed / loaded states
  - scheduler.QNAP_CRON_GUIDE    — guide contains key sections
  - main.retry_with_backoff       — success on first try, retry on failure,
                                    exhausts retries and re-raises
  - main._setup_file_logging      — creates log file, returns FileHandler
  - dry-run flag propagation      — cmd_run respects --dry-run throughout
  - per-phase error isolation     — phase failure does not abort later phases

Run:
  python3 -m pytest tests/unit/test_phase6.py -v
  python3 tests/unit/test_phase6.py
"""

import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

SKILL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import scheduler
import main as main_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_skill_root(tmp_dir: str) -> Path:
    """Return a temp directory tree that mirrors the skill root layout."""
    root = Path(tmp_dir)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "data" / "logs").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    return root


# ===========================================================================
# scheduler.get_plist_content
# ===========================================================================

class TestGetPlistContent(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.skill_root = _tmp_skill_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_xml_string(self):
        content = scheduler.get_plist_content(self.skill_root)
        self.assertIsInstance(content, str)
        self.assertIn("<?xml", content)

    def test_contains_plist_label(self):
        content = scheduler.get_plist_content(self.skill_root)
        self.assertIn("com.jobsearch.daily", content)

    def test_default_schedule_hour_9(self):
        content = scheduler.get_plist_content(self.skill_root, schedule_hour=9)
        self.assertIn("<integer>9</integer>", content)

    def test_custom_schedule_hour(self):
        content = scheduler.get_plist_content(self.skill_root, schedule_hour=7)
        self.assertIn("<integer>7</integer>", content)

    def test_contains_main_py_path(self):
        content = scheduler.get_plist_content(self.skill_root)
        self.assertIn("main.py", content)
        self.assertIn("run", content)

    def test_contains_log_paths(self):
        content = scheduler.get_plist_content(self.skill_root)
        self.assertIn("launchd_stdout.log", content)
        self.assertIn("launchd_stderr.log", content)

    def test_custom_python_executable(self):
        content = scheduler.get_plist_content(
            self.skill_root,
            python_executable="/usr/local/bin/python3.12",
        )
        self.assertIn("/usr/local/bin/python3.12", content)

    def test_run_at_load_false(self):
        content = scheduler.get_plist_content(self.skill_root)
        self.assertIn("<false/>", content)

    def test_minute_is_zero(self):
        content = scheduler.get_plist_content(self.skill_root)
        self.assertIn("<key>Minute</key>", content)
        # The minute entry should be 0
        import re
        minutes = re.findall(r"<key>Minute</key>\s*<integer>(\d+)</integer>", content)
        self.assertEqual(minutes, ["0"])


# ===========================================================================
# scheduler.install
# ===========================================================================

class TestSchedulerInstall(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.skill_root = _tmp_skill_root(self.tmp.name)
        self.fake_agents_dir = Path(self.tmp.name) / "LaunchAgents"
        self.fake_agents_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _patch_dirs(self):
        """Patch LAUNCH_AGENTS_DIR to point to temp dir."""
        return patch.object(scheduler, "LAUNCH_AGENTS_DIR", self.fake_agents_dir)

    def test_plist_written_to_disk(self):
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
                result = scheduler.install(self.skill_root, schedule_hour=9)
        plist_path = self.fake_agents_dir / scheduler.PLIST_FILENAME
        self.assertTrue(plist_path.exists())
        self.assertIn("com.jobsearch.daily", plist_path.read_text())

    def test_returns_success_dict(self):
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
                result = scheduler.install(self.skill_root, schedule_hour=9)
        self.assertTrue(result["success"])
        self.assertIn("plist_path", result)
        self.assertIn("message", result)

    def test_launchctl_load_called(self):
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
                scheduler.install(self.skill_root, schedule_hour=9)
        # At least one call should include "load"
        load_calls = [c for c in mock_run.call_args_list if "load" in c.args[0]]
        self.assertTrue(len(load_calls) >= 1)

    def test_launchctl_failure_still_writes_plist(self):
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="error msg", stdout="")
                result = scheduler.install(self.skill_root, schedule_hour=9)
        plist_path = self.fake_agents_dir / scheduler.PLIST_FILENAME
        self.assertTrue(plist_path.exists())
        # success=False because launchctl failed
        self.assertFalse(result["success"])

    def test_log_dir_created(self):
        log_dir = self.skill_root / "data" / "logs"
        import shutil
        shutil.rmtree(log_dir, ignore_errors=True)
        self.assertFalse(log_dir.exists())
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
                scheduler.install(self.skill_root, schedule_hour=9)
        self.assertTrue(log_dir.exists())


# ===========================================================================
# scheduler.uninstall
# ===========================================================================

class TestSchedulerUninstall(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake_agents_dir = Path(self.tmp.name) / "LaunchAgents"
        self.fake_agents_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _patch_dirs(self):
        return patch.object(scheduler, "LAUNCH_AGENTS_DIR", self.fake_agents_dir)

    def test_no_plist_returns_success(self):
        with self._patch_dirs():
            result = scheduler.uninstall()
        self.assertTrue(result["success"])
        self.assertIn("not installed", result["message"])

    def test_removes_plist_file(self):
        plist_path = self.fake_agents_dir / scheduler.PLIST_FILENAME
        plist_path.write_text("<plist/>")
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
                result = scheduler.uninstall()
        self.assertFalse(plist_path.exists())
        self.assertTrue(result["success"])

    def test_launchctl_unload_called(self):
        plist_path = self.fake_agents_dir / scheduler.PLIST_FILENAME
        plist_path.write_text("<plist/>")
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
                scheduler.uninstall()
        unload_calls = [c for c in mock_run.call_args_list if "unload" in c.args[0]]
        self.assertTrue(len(unload_calls) >= 1)


# ===========================================================================
# scheduler.status
# ===========================================================================

class TestSchedulerStatus(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake_agents_dir = Path(self.tmp.name) / "LaunchAgents"
        self.fake_agents_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _patch_dirs(self):
        return patch.object(scheduler, "LAUNCH_AGENTS_DIR", self.fake_agents_dir)

    def test_not_installed(self):
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="", stdout="")
                info = scheduler.status()
        self.assertFalse(info["installed"])
        self.assertFalse(info["loaded"])
        self.assertIsNone(info["next_run"])

    def test_installed_not_loaded(self):
        plist_path = self.fake_agents_dir / scheduler.PLIST_FILENAME
        skill_root = _tmp_skill_root(self.tmp.name)
        plist_path.write_text(scheduler.get_plist_content(skill_root, schedule_hour=8))
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stderr="", stdout="")
                info = scheduler.status()
        self.assertTrue(info["installed"])
        self.assertFalse(info["loaded"])
        self.assertEqual(info["next_run"], "daily at 08:00")

    def test_installed_and_loaded(self):
        plist_path = self.fake_agents_dir / scheduler.PLIST_FILENAME
        skill_root = _tmp_skill_root(self.tmp.name)
        plist_path.write_text(scheduler.get_plist_content(skill_root, schedule_hour=9))
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="PID = 1234")
                info = scheduler.status()
        self.assertTrue(info["installed"])
        self.assertTrue(info["loaded"])
        self.assertEqual(info["next_run"], "daily at 09:00")

    def test_plist_path_in_result(self):
        with self._patch_dirs():
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                info = scheduler.status()
        self.assertIn(scheduler.PLIST_FILENAME, info["plist_path"])


# ===========================================================================
# scheduler.QNAP_CRON_GUIDE
# ===========================================================================

class TestQnapGuide(unittest.TestCase):

    def test_guide_is_string(self):
        self.assertIsInstance(scheduler.QNAP_CRON_GUIDE, str)

    def test_guide_contains_crontab(self):
        self.assertIn("crontab", scheduler.QNAP_CRON_GUIDE)

    def test_guide_contains_ssh(self):
        self.assertIn("SSH", scheduler.QNAP_CRON_GUIDE)

    def test_guide_contains_python(self):
        self.assertIn("python3", scheduler.QNAP_CRON_GUIDE)

    def test_guide_contains_env_vars(self):
        self.assertIn("MAMMOTH_API_KEY", scheduler.QNAP_CRON_GUIDE)

    def test_guide_mentions_dry_run(self):
        self.assertIn("dry-run", scheduler.QNAP_CRON_GUIDE)

    def test_print_qnap_guide_runs(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            scheduler.print_qnap_guide()
        self.assertIn("QNAP", buf.getvalue())


# ===========================================================================
# main.retry_with_backoff
# ===========================================================================

class TestRetryWithBackoff(unittest.TestCase):

    def test_success_on_first_try(self):
        calls = []
        def fn():
            calls.append(1)
            return "ok"
        result = main_mod.retry_with_backoff(fn, max_retries=3, backoff_base=0.001)
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)

    def test_success_on_second_try(self):
        calls = []
        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("transient")
            return "ok"
        result = main_mod.retry_with_backoff(fn, max_retries=3, backoff_base=0.001)
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 2)

    def test_exhausts_retries_and_raises(self):
        calls = []
        def fn():
            calls.append(1)
            raise ValueError("persistent failure")
        with self.assertRaises(ValueError):
            main_mod.retry_with_backoff(fn, max_retries=3, backoff_base=0.001)
        self.assertEqual(len(calls), 3)

    def test_re_raises_last_exception(self):
        def fn():
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError) as ctx:
            main_mod.retry_with_backoff(fn, max_retries=2, backoff_base=0.001)
        self.assertIn("boom", str(ctx.exception))

    def test_respects_exception_filter(self):
        """Only retries on specified exception types; others bubble immediately."""
        calls = []
        def fn():
            calls.append(1)
            raise KeyError("not retried")
        with self.assertRaises(KeyError):
            main_mod.retry_with_backoff(
                fn, max_retries=3, backoff_base=0.001, exceptions=(ValueError,)
            )
        # Should have raised on first attempt without retrying
        self.assertEqual(len(calls), 1)

    def test_single_retry(self):
        calls = []
        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise IOError("once")
            return "done"
        result = main_mod.retry_with_backoff(fn, max_retries=2, backoff_base=0.001)
        self.assertEqual(result, "done")
        self.assertEqual(len(calls), 2)


# ===========================================================================
# main._setup_file_logging
# ===========================================================================

class TestSetupFileLogging(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp.name) / "logs"

    def tearDown(self):
        # Remove any FileHandlers we added during tests
        root_logger = logging.getLogger()
        root_logger.handlers = [
            h for h in root_logger.handlers
            if not isinstance(h, logging.FileHandler)
            or "test_" not in (getattr(h, "baseFilename", "") or "")
        ]
        self.tmp.cleanup()

    def _clean_handlers(self):
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                h.close()
                root.removeHandler(h)

    def test_creates_log_directory(self):
        self._clean_handlers()
        self.assertFalse(self.log_dir.exists())
        main_mod._setup_file_logging(self.log_dir)
        self.assertTrue(self.log_dir.exists())
        self._clean_handlers()

    def test_creates_log_file(self):
        self._clean_handlers()
        main_mod._setup_file_logging(self.log_dir)
        from datetime import date
        expected = self.log_dir / f"run_{date.today().isoformat()}.log"
        self.assertTrue(expected.exists())
        self._clean_handlers()

    def test_returns_file_handler(self):
        self._clean_handlers()
        handler = main_mod._setup_file_logging(self.log_dir)
        self.assertIsInstance(handler, logging.FileHandler)
        handler.close()
        logging.getLogger().removeHandler(handler)

    def test_log_file_receives_messages(self):
        self._clean_handlers()
        handler = main_mod._setup_file_logging(self.log_dir)
        logger = logging.getLogger("test_phase6_file_log")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info("Test log message from phase6")
        handler.flush()
        handler.close()
        logging.getLogger().removeHandler(handler)
        logger.removeHandler(handler)

        from datetime import date
        log_path = self.log_dir / f"run_{date.today().isoformat()}.log"
        content = log_path.read_text()
        self.assertIn("Test log message from phase6", content)

    def test_invalid_dir_returns_none(self):
        """If the log dir cannot be created (permissions), returns None gracefully."""
        self._clean_handlers()
        # Use a path that can't be a directory (a file path)
        bogus = Path(self.tmp.name) / "this_is_a_file"
        bogus.write_text("block")
        bad_dir = bogus / "subdir"  # can't mkdir under a file
        handler = main_mod._setup_file_logging(bad_dir)
        self.assertIsNone(handler)


# ===========================================================================
# Per-phase error isolation in cmd_run
# ===========================================================================

class TestCmdRunErrorIsolation(unittest.TestCase):
    """
    Verify that a crash in one pipeline phase does not prevent later phases
    from executing.  We mock each module import and inject failures selectively.
    """

    def _make_profile(self) -> dict:
        return {
            "personal": {"name": "Test User", "email": "test@example.com"},
            "job_preferences": {"titles": ["Engineer"]},
            "automation": {"max_applications_per_day": 20, "schedule_hour": 9},
            "ai": {"mammoth_api_key_env": "MAMMOTH_API_KEY"},
        }

    def _make_args(self, dry_run=True, verbose=False):
        args = MagicMock()
        args.dry_run = dry_run
        args.verbose = verbose
        return args

    def _base_patches(self):
        """Return a dict of patches that make the pipeline succeed by default."""
        import types

        # Fake DiscoveryResult
        fake_result = MagicMock()
        fake_result.found = 3
        fake_result.new = 2
        fake_result.errors = []

        mock_search = MagicMock()
        mock_search.run_discovery.return_value = {"linkedin": fake_result}

        mock_scorer = MagicMock()
        mock_scorer.run_scoring.return_value = {
            "scored": 2, "skipped_low": 0, "errors": 0, "skipped_no_ai": 0
        }
        mock_scorer.create_ai_client.return_value = MagicMock()

        mock_router = MagicMock()
        mock_router.route.return_value = ("/path/resume.pdf", None)

        mock_dispatcher = MagicMock()
        mock_dispatcher.run_dispatch.return_value = {
            "auto": 1, "queued": 0, "skipped": 0, "notify": 0, "errors": 0
        }

        mock_notifier = MagicMock()
        mock_notifier.send_digest.return_value = True
        mock_notifier.poll_replies.return_value = {
            "replies_found": 0, "answers_stored": 0, "jobs_requeued": 0
        }

        return {
            "search": mock_search,
            "scorer": mock_scorer,
            "resume_router": mock_router,
            "dispatcher": mock_dispatcher,
            "notifier": mock_notifier,
        }

    def _run_with_mocks(self, mock_modules: dict, tmp_dir: str):
        """Patch sys.modules and run cmd_run in dry-run mode."""
        skill_root_patch = patch.object(main_mod, "SKILL_ROOT", Path(tmp_dir))

        profile = self._make_profile()
        platform_cfg = {"platforms": {}}

        (Path(tmp_dir) / "config").mkdir(parents=True, exist_ok=True)
        import json
        (Path(tmp_dir) / "config" / "user_profile.json").write_text(json.dumps(profile))
        (Path(tmp_dir) / "config" / "resume_map.json").write_text(
            json.dumps({"domains": {}, "default": ""})
        )
        (Path(tmp_dir) / "config" / "platform_config.json").write_text(
            json.dumps({"platforms": {}})
        )
        (Path(tmp_dir) / "data" / "logs").mkdir(parents=True, exist_ok=True)

        import db
        db_path = Path(tmp_dir) / "data" / "jobs.db"
        db.init_db(db_path=db_path)

        with skill_root_patch:
            with patch.dict(sys.modules, mock_modules):
                with patch.object(db, "get_connection") as mock_conn:
                    mock_conn.return_value.__enter__ = MagicMock(
                        return_value=MagicMock(
                            execute=MagicMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))
                        )
                    )
                    mock_conn.return_value.__exit__ = MagicMock(return_value=False)
                    with patch.object(db, "get_daily_stats") as mock_stats:
                        mock_stats.return_value = {
                            "date": "2026-03-23",
                            "applied_today": 0,
                            "queued": 0,
                            "skipped_today": 0,
                            "pending_questions": 0,
                            "total_jobs_tracked": 5,
                        }
                        with patch.object(db, "init_db"):
                            args = self._make_args(dry_run=True)
                            main_mod.cmd_run(args)

    def test_discovery_failure_does_not_abort(self):
        """Phase 2 discovery crash → pipeline continues to Phase 5."""
        with tempfile.TemporaryDirectory() as tmp:
            mocks = self._base_patches()
            mocks["search"].run_discovery.side_effect = RuntimeError("platform down")
            notifier_called = []
            mocks["notifier"].send_digest.side_effect = lambda **kw: notifier_called.append(1) or True

            try:
                self._run_with_mocks(mocks, tmp)
            except SystemExit:
                pass
            # Notifier should still have been called
            self.assertTrue(len(notifier_called) >= 1)

    def test_scoring_failure_does_not_abort(self):
        """Phase 2 scoring crash → pipeline continues."""
        with tempfile.TemporaryDirectory() as tmp:
            mocks = self._base_patches()
            mocks["scorer"].run_scoring.side_effect = RuntimeError("AI down")
            notifier_called = []
            mocks["notifier"].send_digest.side_effect = lambda **kw: notifier_called.append(1) or True

            try:
                self._run_with_mocks(mocks, tmp)
            except SystemExit:
                pass
            self.assertTrue(len(notifier_called) >= 1)

    def test_dispatch_failure_does_not_abort_notifications(self):
        """Phase 4 crash → Phase 5 notifications still run."""
        with tempfile.TemporaryDirectory() as tmp:
            mocks = self._base_patches()
            mocks["dispatcher"].run_dispatch.side_effect = RuntimeError("dispatcher error")
            notifier_called = []
            mocks["notifier"].send_digest.side_effect = lambda **kw: notifier_called.append(1) or True

            try:
                self._run_with_mocks(mocks, tmp)
            except SystemExit:
                pass
            self.assertTrue(len(notifier_called) >= 1)


# ===========================================================================
# Dry-run propagation test
# ===========================================================================

class TestDryRunFlag(unittest.TestCase):

    def test_dry_run_arg_parsed(self):
        parser = main_mod.build_parser()
        args = parser.parse_args(["run", "--dry-run"])
        self.assertTrue(args.dry_run)

    def test_no_dry_run_is_false(self):
        parser = main_mod.build_parser()
        args = parser.parse_args(["run"])
        self.assertFalse(args.dry_run)

    def test_verbose_arg_parsed(self):
        parser = main_mod.build_parser()
        args = parser.parse_args(["run", "--verbose"])
        self.assertTrue(args.verbose)


# ===========================================================================
# Scheduler CLI integration in argument parser
# ===========================================================================

class TestSchedulerCLI(unittest.TestCase):

    def test_scheduler_install_parses(self):
        parser = main_mod.build_parser()
        args = parser.parse_args(["scheduler", "install"])
        self.assertEqual(args.command, "scheduler")
        self.assertEqual(args.scheduler_command, "install")

    def test_scheduler_uninstall_parses(self):
        parser = main_mod.build_parser()
        args = parser.parse_args(["scheduler", "uninstall"])
        self.assertEqual(args.scheduler_command, "uninstall")

    def test_scheduler_status_parses(self):
        parser = main_mod.build_parser()
        args = parser.parse_args(["scheduler", "status"])
        self.assertEqual(args.scheduler_command, "status")

    def test_scheduler_qnap_parses(self):
        parser = main_mod.build_parser()
        args = parser.parse_args(["scheduler", "qnap"])
        self.assertEqual(args.scheduler_command, "qnap")

    def test_scheduler_func_is_cmd_scheduler(self):
        parser = main_mod.build_parser()
        args = parser.parse_args(["scheduler", "status"])
        self.assertEqual(args.func, main_mod.cmd_scheduler)


# ===========================================================================
# Full run log written test
# ===========================================================================

class TestRunLogWritten(unittest.TestCase):

    def test_log_file_exists_after_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            main_mod._setup_file_logging(log_dir)
            from datetime import date
            log_path = log_dir / f"run_{date.today().isoformat()}.log"
            self.assertTrue(log_path.exists())
            # Cleanup handler
            root = logging.getLogger()
            for h in list(root.handlers):
                if isinstance(h, logging.FileHandler) and tmp in getattr(h, "baseFilename", ""):
                    h.close()
                    root.removeHandler(h)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
