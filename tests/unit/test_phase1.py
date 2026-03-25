"""
Phase 1 tests: SQLite schema, config validation, Q&A seeding, deduplication.

Run:
  python3 -m pytest tests/unit/test_phase1.py -v
  # or directly:
  python3 tests/unit/test_phase1.py
"""

import json
import os
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import db


# ---------------------------------------------------------------------------
# Helper: in-memory / temp DB for isolated tests
# ---------------------------------------------------------------------------

def make_temp_db() -> Path:
    """Create a temporary DB path (deleted after test)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)
    db.init_db(path)
    return path


def cleanup(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Test: Schema initialization
# ---------------------------------------------------------------------------

class TestDatabaseSchema(unittest.TestCase):

    def setUp(self):
        self.db_path = make_temp_db()

    def tearDown(self):
        cleanup(self.db_path)

    def test_all_tables_created(self):
        """All four tables must exist after init_db."""
        conn = db.get_connection(self.db_path)
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        self.assertIn("jobs", tables)
        self.assertIn("applications", tables)
        self.assertIn("qa_store", tables)
        self.assertIn("pending_questions", tables)

    def test_jobs_columns(self):
        """jobs table must have all required columns."""
        conn = db.get_connection(self.db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        conn.close()
        required = {"id", "platform", "title", "company", "url", "description",
                    "relevance_score", "matched_resume", "posted_date",
                    "discovered_date", "status", "skip_reason",
                    "url_hash", "title_hash", "updated_at"}
        self.assertTrue(required.issubset(cols), f"Missing columns: {required - cols}")

    def test_qa_store_columns(self):
        """qa_store table must have embedding and all required columns."""
        conn = db.get_connection(self.db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(qa_store)").fetchall()}
        conn.close()
        required = {"id", "question_pattern", "answer", "source",
                    "embedding", "used_count", "last_used", "created_at"}
        self.assertTrue(required.issubset(cols), f"Missing columns: {required - cols}")

    def test_indexes_created(self):
        """All six indexes must be created."""
        conn = db.get_connection(self.db_path)
        indexes = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        expected = {"idx_jobs_status", "idx_jobs_url_hash", "idx_jobs_title_hash",
                    "idx_jobs_discovered", "idx_applications_job", "idx_pending_answered"}
        self.assertTrue(expected.issubset(indexes), f"Missing indexes: {expected - indexes}")

    def test_idempotent_init(self):
        """Calling init_db twice must not raise or duplicate tables."""
        try:
            db.init_db(self.db_path)
        except Exception as e:
            self.fail(f"Second init_db raised: {e}")


# ---------------------------------------------------------------------------
# Test: Job insertion and deduplication
# ---------------------------------------------------------------------------

class TestJobDedup(unittest.TestCase):

    def setUp(self):
        self.db_path = make_temp_db()

    def tearDown(self):
        cleanup(self.db_path)

    def _make_job(self, suffix="1"):
        return {
            "id": f"linkedin:job{suffix}",
            "platform": "linkedin",
            "title": f"Software Engineer {suffix}",
            "company": "Acme Corp",
            "url": f"https://linkedin.com/jobs/{suffix}",
            "description": "Build cool things.",
            "posted_date": "2026-03-23",
            "url_hash": f"urlhash{suffix}",
            "title_hash": f"titlehash{suffix}",
        }

    def test_insert_new_job(self):
        """Inserting a new job returns True and stores it."""
        job = self._make_job("1")
        inserted = db.insert_job(job, self.db_path)
        self.assertTrue(inserted)
        conn = db.get_connection(self.db_path)
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],)).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["title"], job["title"])

    def test_duplicate_url_hash_rejected(self):
        """Second insert with same url_hash must return False and not duplicate."""
        job1 = self._make_job("1")
        job2 = dict(job1)
        job2["id"] = "indeed:job999"
        job2["url"] = "https://indeed.com/job/999"
        # Same url_hash
        db.insert_job(job1, self.db_path)
        inserted = db.insert_job(job2, self.db_path)
        self.assertFalse(inserted)
        conn = db.get_connection(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_duplicate_title_hash_rejected(self):
        """Same company+title on different platform must be treated as duplicate."""
        job1 = self._make_job("1")
        job2 = self._make_job("1")
        job2["id"] = "glassdoor:job1"
        job2["url"] = "https://glassdoor.com/job/1"
        job2["url_hash"] = "different_url_hash"
        # Same title_hash simulates same company+title
        db.insert_job(job1, self.db_path)
        inserted = db.insert_job(job2, self.db_path)
        self.assertFalse(inserted)

    def test_different_jobs_both_inserted(self):
        """Two genuinely different jobs must both be inserted."""
        db.insert_job(self._make_job("1"), self.db_path)
        db.insert_job(self._make_job("2"), self.db_path)
        conn = db.get_connection(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.close()
        self.assertEqual(count, 2)

    def test_default_status_is_new(self):
        """Newly inserted job must have status='new'."""
        db.insert_job(self._make_job("1"), self.db_path)
        conn = db.get_connection(self.db_path)
        row = conn.execute("SELECT status FROM jobs WHERE id = 'linkedin:job1'").fetchone()
        conn.close()
        self.assertEqual(row["status"], "new")


# ---------------------------------------------------------------------------
# Test: Q&A seeding
# ---------------------------------------------------------------------------

class TestQASeeding(unittest.TestCase):

    def setUp(self):
        self.db_path = make_temp_db()

    def tearDown(self):
        cleanup(self.db_path)

    def _sample_profile(self, **overrides):
        profile = {
            "job_preferences": {
                "salary_min": 150000,
                "salary_preferred": 180000,
                "salary_currency": "USD",
            },
            "application_defaults": {
                "work_authorized": True,
                "visa_sponsorship_required": False,
                "notice_period_days": 14,
                "referral": False,
            },
        }
        profile["application_defaults"].update(overrides)
        return profile

    def test_seeding_inserts_rows(self):
        """Seeding must insert at least 5 Q&A rows."""
        db.seed_qa_store(self._sample_profile(), self.db_path)
        conn = db.get_connection(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM qa_store").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(count, 5)

    def test_work_auth_yes_stored(self):
        """Work authorization 'yes' must be stored correctly."""
        db.seed_qa_store(self._sample_profile(work_authorized=True), self.db_path)
        conn = db.get_connection(self.db_path)
        row = conn.execute(
            "SELECT answer FROM qa_store WHERE question_pattern LIKE '%authorized to work%'"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertIn("Yes", row["answer"])

    def test_sponsorship_no_stored(self):
        """Sponsorship=False must produce 'No' answer."""
        db.seed_qa_store(self._sample_profile(visa_sponsorship_required=False), self.db_path)
        conn = db.get_connection(self.db_path)
        row = conn.execute(
            "SELECT answer FROM qa_store WHERE question_pattern LIKE '%sponsorship%'"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertIn("No", row["answer"])

    def test_salary_answer_contains_amount(self):
        """Salary Q&A answer must contain the preferred salary amount."""
        db.seed_qa_store(self._sample_profile(), self.db_path)
        conn = db.get_connection(self.db_path)
        row = conn.execute(
            "SELECT answer FROM qa_store WHERE question_pattern LIKE '%salary%' LIMIT 1"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertIn("180,000", row["answer"])

    def test_seeding_is_idempotent(self):
        """Seeding twice must produce the same row count as seeding once."""
        profile = self._sample_profile()
        db.seed_qa_store(profile, self.db_path)
        conn = db.get_connection(self.db_path)
        count_after_first = conn.execute("SELECT COUNT(*) FROM qa_store").fetchone()[0]
        conn.close()

        db.seed_qa_store(profile, self.db_path)
        conn = db.get_connection(self.db_path)
        count_after_second = conn.execute("SELECT COUNT(*) FROM qa_store").fetchone()[0]
        conn.close()

        self.assertEqual(count_after_first, count_after_second,
                         "Seeding twice must not add duplicate rows")


# ---------------------------------------------------------------------------
# Test: Status and stats
# ---------------------------------------------------------------------------

class TestDailyStats(unittest.TestCase):

    def setUp(self):
        self.db_path = make_temp_db()

    def tearDown(self):
        cleanup(self.db_path)

    def test_empty_stats(self):
        """Fresh DB must return zero counts for all stats."""
        stats = db.get_daily_stats(self.db_path)
        self.assertEqual(stats["applied_today"], 0)
        self.assertEqual(stats["queued"], 0)
        self.assertEqual(stats["pending_questions"], 0)
        self.assertEqual(stats["total_jobs_tracked"], 0)

    def test_stats_after_insert(self):
        """Inserted job shows up in total_jobs_tracked."""
        db.insert_job({
            "id": "linkedin:123",
            "platform": "linkedin",
            "title": "Engineer",
            "company": "Corp",
            "url": "https://linkedin.com/jobs/123",
            "description": "test",
            "posted_date": "2026-03-23",
            "url_hash": "abc",
            "title_hash": "def",
        }, self.db_path)
        stats = db.get_daily_stats(self.db_path)
        self.assertEqual(stats["total_jobs_tracked"], 1)


# ---------------------------------------------------------------------------
# Test: Config files exist and are valid JSON
# ---------------------------------------------------------------------------

class TestConfigFiles(unittest.TestCase):

    CONFIG_DIR = SKILL_ROOT / "config"

    def _load_json(self, filename):
        path = self.CONFIG_DIR / filename
        self.assertTrue(path.exists(), f"{filename} not found at {path}")
        with open(path) as f:
            return json.load(f)

    def test_user_profile_valid_json(self):
        data = self._load_json("user_profile.json")
        self.assertIn("personal", data)
        self.assertIn("job_preferences", data)
        self.assertIn("application_defaults", data)
        self.assertIn("automation", data)
        self.assertIn("platforms", data)
        self.assertIn("notifications", data)
        self.assertIn("ai", data)

    def test_resume_map_valid_json(self):
        data = self._load_json("resume_map.json")
        self.assertIn("domains", data)
        self.assertIn("default", data)
        self.assertIsInstance(data["domains"], dict)

    def test_platform_config_valid_json(self):
        data = self._load_json("platform_config.json")
        self.assertIn("platforms", data)
        platforms = data["platforms"]
        expected_platforms = [
            "linkedin", "indeed", "glassdoor", "ziprecruiter",
            "wellfound", "trueup", "otta", "workatstartup",
            "underdog", "welcometothejungle"
        ]
        for p in expected_platforms:
            self.assertIn(p, platforms, f"Platform '{p}' missing from platform_config.json")

    def test_all_11_platforms_in_user_profile(self):
        data = self._load_json("user_profile.json")
        platforms = data.get("platforms", [])
        self.assertEqual(len(platforms), 10,
                         f"Expected 10 platforms, got {len(platforms)}: {platforms}")

    def test_automation_defaults_sensible(self):
        data = self._load_json("user_profile.json")
        auto = data["automation"]
        self.assertLessEqual(auto["max_applications_per_day"], 30,
                             "max_applications_per_day should not exceed 30 (safety)")
        self.assertGreaterEqual(auto["relevance_threshold"], 0.0)
        self.assertLessEqual(auto["relevance_threshold"], 1.0)

    def test_resume_map_has_domain_keywords(self):
        data = self._load_json("resume_map.json")
        for domain_key, domain_val in data["domains"].items():
            self.assertIn("keywords", domain_val,
                          f"Domain '{domain_key}' missing keywords list")
            self.assertGreater(len(domain_val["keywords"]), 0,
                               f"Domain '{domain_key}' has empty keywords list")


# ---------------------------------------------------------------------------
# Test: Job status transitions
# ---------------------------------------------------------------------------

class TestJobStatusTransitions(unittest.TestCase):

    def setUp(self):
        self.db_path = make_temp_db()
        db.insert_job({
            "id": "linkedin:test1",
            "platform": "linkedin",
            "title": "Engineer",
            "company": "Corp",
            "url": "https://example.com/job/1",
            "description": "test",
            "posted_date": "2026-03-23",
            "url_hash": "hash1",
            "title_hash": "thash1",
        }, self.db_path)

    def tearDown(self):
        cleanup(self.db_path)

    def test_mark_skipped(self):
        db.mark_job_status("linkedin:test1", "skipped", reason="low score", db_path=self.db_path)
        conn = db.get_connection(self.db_path)
        row = conn.execute("SELECT status, skip_reason FROM jobs WHERE id = 'linkedin:test1'").fetchone()
        conn.close()
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(row["skip_reason"], "low score")

    def test_update_score(self):
        db.update_job_score("linkedin:test1", 0.85, "/path/to/resume.pdf", self.db_path)
        conn = db.get_connection(self.db_path)
        row = conn.execute("SELECT relevance_score, matched_resume, status FROM jobs WHERE id = 'linkedin:test1'").fetchone()
        conn.close()
        self.assertAlmostEqual(row["relevance_score"], 0.85)
        self.assertEqual(row["matched_resume"], "/path/to/resume.pdf")
        self.assertEqual(row["status"], "scoring")

    def test_rate_limit_counter(self):
        """count_applications_today must return 0 before any application recorded."""
        count = db.count_applications_today(self.db_path)
        self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestDatabaseSchema,
        TestJobDedup,
        TestQASeeding,
        TestDailyStats,
        TestConfigFiles,
        TestJobStatusTransitions,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
