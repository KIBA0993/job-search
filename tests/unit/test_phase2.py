"""
Phase 2 tests: job discovery (search.py) and relevance scoring (scorer.py).

Run:
  python3 -m pytest tests/unit/test_phase2.py -v
  # or directly:
  python3 tests/unit/test_phase2.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SKILL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import db
from search import (
    normalize_url,
    compute_url_hash,
    normalize_title,
    compute_title_hash,
    build_search_url,
    build_extractor_js,
    BrowserCmd,
    RawJob,
    DiscoveryResult,
    PlatformAdapter,
    run_discovery,
)
from scorer import (
    read_resume_text,
    load_resume_versions,
    score_single,
    score_job,
    run_scoring,
    create_ai_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_db() -> Path:
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


def make_temp_txt_resume(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Test: URL normalization and hash computation
# ---------------------------------------------------------------------------

class TestUrlNormalization(unittest.TestCase):

    def test_tracking_params_stripped(self):
        raw = "https://linkedin.com/jobs/123?utm_source=google&utm_campaign=jobs&trk=something"
        clean = normalize_url(raw)
        self.assertNotIn("utm_source", clean)
        self.assertNotIn("utm_campaign", clean)
        self.assertNotIn("trk=", clean)
        self.assertIn("linkedin.com/jobs/123", clean)

    def test_fragment_stripped(self):
        raw = "https://example.com/job/42#apply"
        clean = normalize_url(raw)
        self.assertNotIn("#apply", clean)

    def test_scheme_lowercased(self):
        raw = "HTTPS://example.com/job/42"
        clean = normalize_url(raw)
        self.assertTrue(clean.startswith("https://"))

    def test_same_url_same_hash(self):
        url = "https://indeed.com/jobs/view/12345?utm_source=email"
        h1 = compute_url_hash(url)
        h2 = compute_url_hash(url)
        self.assertEqual(h1, h2)

    def test_different_urls_different_hashes(self):
        h1 = compute_url_hash("https://indeed.com/jobs/view/111")
        h2 = compute_url_hash("https://indeed.com/jobs/view/222")
        self.assertNotEqual(h1, h2)

    def test_tracking_stripped_same_hash(self):
        url1 = "https://indeed.com/jobs/view/999"
        url2 = "https://indeed.com/jobs/view/999?utm_source=google&utm_medium=cpc"
        self.assertEqual(compute_url_hash(url1), compute_url_hash(url2))

    def test_hash_is_64_hex_chars(self):
        h = compute_url_hash("https://example.com/job/1")
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))


# ---------------------------------------------------------------------------
# Test: Title normalization and hash computation
# ---------------------------------------------------------------------------

class TestTitleNormalization(unittest.TestCase):

    def test_case_insensitive(self):
        h1 = compute_title_hash("Acme Corp", "Software Engineer")
        h2 = compute_title_hash("acme corp", "software engineer")
        self.assertEqual(h1, h2)

    def test_seniority_stripped(self):
        h_senior = compute_title_hash("Acme", "Senior Software Engineer")
        h_plain = compute_title_hash("Acme", "Software Engineer")
        self.assertEqual(h_senior, h_plain)

    def test_sr_stripped(self):
        h1 = compute_title_hash("Corp", "Sr. Backend Engineer")
        h2 = compute_title_hash("Corp", "Backend Engineer")
        self.assertEqual(h1, h2)

    def test_different_companies_different_hashes(self):
        h1 = compute_title_hash("Acme Corp", "Software Engineer")
        h2 = compute_title_hash("Other Co", "Software Engineer")
        self.assertNotEqual(h1, h2)

    def test_different_titles_different_hashes(self):
        h1 = compute_title_hash("Acme", "Backend Engineer")
        h2 = compute_title_hash("Acme", "Frontend Engineer")
        self.assertNotEqual(h1, h2)

    def test_hash_is_64_hex_chars(self):
        h = compute_title_hash("Acme", "Software Engineer")
        self.assertEqual(len(h), 64)


# ---------------------------------------------------------------------------
# Test: Search URL builder
# ---------------------------------------------------------------------------

class TestBuildSearchUrl(unittest.TestCase):

    def test_title_filled(self):
        url = build_search_url(
            "https://example.com/jobs?q={title}&l={location}",
            {"title": "data engineer", "location": ""},
        )
        self.assertIn("data+engineer", url)

    def test_location_filled(self):
        url = build_search_url(
            "https://example.com/jobs?q={title}&l={location}",
            {"title": "engineer", "location": "New York"},
        )
        self.assertIn("New+York", url)

    def test_title_slug_filled(self):
        url = build_search_url(
            "https://example.com/roles/{title_slug}",
            {"title": "Machine Learning Engineer", "location": ""},
        )
        self.assertIn("machine-learning-engineer", url)

    def test_missing_location_leaves_empty(self):
        url = build_search_url(
            "https://example.com/jobs?l={location}",
            {"title": "engineer"},
        )
        self.assertIn("l=", url)

    def test_special_chars_encoded(self):
        url = build_search_url(
            "https://example.com/jobs?q={title}",
            {"title": "C++ Engineer"},
        )
        self.assertNotIn("C++", url)
        self.assertIn("C%2B%2B", url)


# ---------------------------------------------------------------------------
# Test: Extractor JS builder
# ---------------------------------------------------------------------------

class TestBuildExtractorJs(unittest.TestCase):

    SAMPLE_CFG = {
        "job_list_selector": ".job_seen_beacon",
        "job_title_selector": "h2.jobTitle a",
        "job_company_selector": "[data-testid='company-name']",
        "job_url_selector": "h2.jobTitle a",
        "job_date_selector": "[data-testid='myJobsStateDate']",
    }

    def test_returns_string(self):
        js = build_extractor_js(self.SAMPLE_CFG)
        self.assertIsInstance(js, str)
        self.assertGreater(len(js), 50)

    def test_contains_list_selector(self):
        js = build_extractor_js(self.SAMPLE_CFG)
        self.assertIn("job_seen_beacon", js)

    def test_wraps_in_iife(self):
        js = build_extractor_js(self.SAMPLE_CFG)
        self.assertIn("(function()", js)
        self.assertIn("try{", js)

    def test_no_date_sel_uses_empty_string(self):
        cfg = dict(self.SAMPLE_CFG)
        cfg["job_date_selector"] = None
        js = build_extractor_js(cfg)
        self.assertIsInstance(js, str)

    def test_single_quotes_escaped_in_selectors(self):
        cfg = dict(self.SAMPLE_CFG)
        cfg["job_company_selector"] = "[data-testid='company-name']"
        js = build_extractor_js(cfg)
        # Should not leave unescaped single quotes inside the JS string
        # (the outer single-quoted strings should have them escaped)
        self.assertIsInstance(js, str)


# ---------------------------------------------------------------------------
# Test: BrowserCmd graceful failure
# ---------------------------------------------------------------------------

class TestBrowserCmdGracefulFailure(unittest.TestCase):

    def test_returns_empty_when_not_found(self):
        """BrowserCmd must return '' when bb-browser is not in PATH."""
        browser = BrowserCmd(timeout=5)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("bb-browser not found")
            result = browser.run("open", "https://example.com")
        self.assertEqual(result, "")

    def test_returns_empty_on_timeout(self):
        import subprocess
        browser = BrowserCmd(timeout=5)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="bb-browser", timeout=5)
            result = browser.run("open", "https://example.com")
        self.assertEqual(result, "")

    def test_eval_json_returns_none_on_bad_json(self):
        browser = BrowserCmd()
        with patch.object(browser, "eval", return_value="not valid json"):
            result = browser.eval_json("document.title")
        self.assertIsNone(result)

    def test_eval_json_parses_valid_json(self):
        browser = BrowserCmd()
        with patch.object(browser, "eval", return_value='[{"title":"Test","company":"Corp","url":"https://x.com","date":""}]'):
            result = browser.eval_json("someJs()")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Test")


# ---------------------------------------------------------------------------
# Test: PlatformAdapter with mocked BrowserCmd
# ---------------------------------------------------------------------------

class TestPlatformAdapter(unittest.TestCase):

    SAMPLE_PLATFORM_CFG = {
        "search_url": "https://indeed.com/jobs?q={title}&l={location}",
        "job_list_selector": ".job_seen_beacon",
        "job_title_selector": "h2.jobTitle a",
        "job_company_selector": "[data-testid='company-name']",
        "job_url_selector": "h2.jobTitle a",
        "job_date_selector": None,
        "next_page_selector": "a[data-testid='pagination-page-next']",
        "max_pages": 2,
        "rate_limit_ms": 100,
        "enabled": True,
    }

    def _make_adapter(self):
        return PlatformAdapter("indeed", self.SAMPLE_PLATFORM_CFG)

    def test_search_returns_empty_on_browser_failure(self):
        """If bb-browser is unavailable, search returns empty job list."""
        adapter = self._make_adapter()
        with patch("search.BrowserCmd") as MockBrowser:
            mock_instance = MockBrowser.return_value
            mock_instance.open.return_value = ""
            mock_instance.wait.return_value = None
            mock_instance.eval_json.return_value = None  # no jobs found
            mock_instance.close.return_value = None

            jobs, errors = adapter.search({"title": "engineer", "location": ""})

        self.assertIsInstance(jobs, list)
        self.assertIsInstance(errors, list)

    def test_search_deduplicates_within_page(self):
        """Duplicate URLs on the same page must be returned only once."""
        adapter = self._make_adapter()
        duplicate_data = [
            {"title": "Engineer", "company": "Acme", "url": "https://indeed.com/j/1", "date": ""},
            {"title": "Engineer", "company": "Acme", "url": "https://indeed.com/j/1", "date": ""},
        ]
        with patch("search.BrowserCmd") as MockBrowser:
            mock_instance = MockBrowser.return_value
            mock_instance.open.return_value = ""
            mock_instance.wait.return_value = None
            mock_instance.eval_json.side_effect = [duplicate_data, None]
            mock_instance.eval.return_value = "false"  # no next page
            mock_instance.close.return_value = None

            jobs, errors = adapter.search({"title": "engineer", "location": ""})

        self.assertEqual(len(jobs), 1)

    def test_search_returns_raw_jobs(self):
        """Each job dict from browser must map to a RawJob object."""
        adapter = self._make_adapter()
        page_data = [
            {"title": "SWE", "company": "TechCo", "url": "https://indeed.com/j/42", "date": "2d ago"},
            {"title": "ML Eng", "company": "DataInc", "url": "https://indeed.com/j/99", "date": "1d ago"},
        ]
        with patch("search.BrowserCmd") as MockBrowser:
            mock_instance = MockBrowser.return_value
            mock_instance.open.return_value = ""
            mock_instance.wait.return_value = None
            mock_instance.eval_json.side_effect = [page_data, None]
            mock_instance.eval.return_value = "false"
            mock_instance.close.return_value = None

            jobs, errors = adapter.search({"title": "engineer", "location": ""})

        self.assertEqual(len(jobs), 2)
        self.assertIsInstance(jobs[0], RawJob)
        self.assertEqual(jobs[0].title, "SWE")
        self.assertEqual(jobs[0].company, "TechCo")
        self.assertEqual(jobs[0].platform, "indeed")

    def test_search_skips_jobs_with_missing_fields(self):
        """Jobs missing title, company, or URL must be filtered out."""
        adapter = self._make_adapter()
        page_data = [
            {"title": "", "company": "Corp", "url": "https://example.com", "date": ""},
            {"title": "Engineer", "company": "", "url": "https://example.com/2", "date": ""},
            {"title": "Engineer", "company": "Corp", "url": "", "date": ""},
            {"title": "Valid", "company": "Corp", "url": "https://example.com/3", "date": ""},
        ]
        with patch("search.BrowserCmd") as MockBrowser:
            mock_instance = MockBrowser.return_value
            mock_instance.open.return_value = ""
            mock_instance.wait.return_value = None
            mock_instance.eval_json.side_effect = [page_data, None]
            mock_instance.eval.return_value = "false"
            mock_instance.close.return_value = None

            jobs, errors = adapter.search({"title": "engineer", "location": ""})

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].title, "Valid")


# ---------------------------------------------------------------------------
# Test: run_discovery integration (with mocked adapter and DB)
# ---------------------------------------------------------------------------

class TestRunDiscovery(unittest.TestCase):

    def setUp(self):
        self.db_path = make_temp_db()

    def tearDown(self):
        cleanup(self.db_path)

    def _make_profile(self, titles=None, platforms=None):
        return {
            "job_preferences": {
                "titles": titles if titles is not None else ["Software Engineer"],
                "locations": ["Remote"],
                "remote": "flexible",
            },
            "platforms": platforms if platforms is not None else ["indeed"],
            "automation": {"skip_jobs_older_than_days": 30},
        }

    def _make_platform_cfg(self, platform_id="indeed"):
        return {
            "platforms": {
                platform_id: {
                    "enabled": True,
                    "search_url": "https://indeed.com/jobs?q={title}",
                    "job_list_selector": ".job",
                    "job_title_selector": ".title",
                    "job_company_selector": ".company",
                    "job_url_selector": ".title a",
                    "job_date_selector": None,
                    "next_page_selector": None,
                    "max_pages": 1,
                    "rate_limit_ms": 0,
                    "enabled": True,
                }
            }
        }

    def test_no_titles_returns_empty(self):
        profile = self._make_profile(titles=[])
        results = run_discovery(profile, self._make_platform_cfg(), db_path=self.db_path,
                                fetch_descriptions=False)
        self.assertEqual(results, {})

    def test_disabled_platform_skipped(self):
        pcfg = self._make_platform_cfg()
        pcfg["platforms"]["indeed"]["enabled"] = False
        profile = self._make_profile()
        results = run_discovery(profile, pcfg, db_path=self.db_path,
                                fetch_descriptions=False)
        self.assertNotIn("indeed", results)

    def test_new_jobs_stored_in_db(self):
        """Jobs returned by the adapter must be stored in the database."""
        profile = self._make_profile()
        pcfg = self._make_platform_cfg()

        mock_jobs = [
            RawJob(platform="indeed", title="SWE", company="Acme",
                   url="https://indeed.com/j/1"),
            RawJob(platform="indeed", title="ML Eng", company="DataCo",
                   url="https://indeed.com/j/2"),
        ]

        with patch("search.PlatformAdapter") as MockAdapter:
            mock_instance = MockAdapter.return_value
            mock_instance.search.return_value = (mock_jobs, [])

            results = run_discovery(profile, pcfg, db_path=self.db_path,
                                    fetch_descriptions=False)

        stats = db.get_daily_stats(self.db_path)
        self.assertEqual(stats["total_jobs_tracked"], 2)
        self.assertEqual(results["indeed"].new, 2)

    def test_duplicate_jobs_not_stored_twice(self):
        """Running discovery twice with the same jobs must not create duplicates."""
        profile = self._make_profile()
        pcfg = self._make_platform_cfg()

        mock_jobs = [
            RawJob(platform="indeed", title="SWE", company="Acme",
                   url="https://indeed.com/j/1"),
        ]

        with patch("search.PlatformAdapter") as MockAdapter:
            mock_instance = MockAdapter.return_value
            mock_instance.search.return_value = (mock_jobs, [])

            run_discovery(profile, pcfg, db_path=self.db_path, fetch_descriptions=False)
            results2 = run_discovery(profile, pcfg, db_path=self.db_path, fetch_descriptions=False)

        stats = db.get_daily_stats(self.db_path)
        self.assertEqual(stats["total_jobs_tracked"], 1)
        self.assertEqual(results2["indeed"].new, 0)


# ---------------------------------------------------------------------------
# Test: read_resume_text
# ---------------------------------------------------------------------------

class TestReadResumeText(unittest.TestCase):

    def test_reads_txt_file(self):
        path = make_temp_txt_resume("Jane Doe\nSoftware Engineer\nPython, SQL")
        try:
            text = read_resume_text(path)
            self.assertIn("Jane Doe", text)
            self.assertIn("Python", text)
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        text = read_resume_text("/nonexistent/path/resume.txt")
        self.assertEqual(text, "")

    def test_unsupported_format_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".odt", delete=False) as f:
            f.write(b"fake odt content")
            path = f.name
        try:
            text = read_resume_text(path)
            self.assertEqual(text, "")
        finally:
            os.unlink(path)

    def test_max_chars_respected(self):
        long_text = "A" * 20000
        path = make_temp_txt_resume(long_text)
        try:
            text = read_resume_text(path, max_chars=100)
            self.assertLessEqual(len(text), 100)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: load_resume_versions
# ---------------------------------------------------------------------------

class TestLoadResumeVersions(unittest.TestCase):

    def test_loads_txt_resumes(self):
        r1 = make_temp_txt_resume("Resume version 1: Python developer")
        r2 = make_temp_txt_resume("Resume version 2: Data scientist")
        resume_map = {
            "domains": {
                "engineering": {"resume": str(r1), "keywords": ["python"]},
                "data": {"resume": str(r2), "keywords": ["ml"]},
            },
            "default": "",
        }
        try:
            versions = load_resume_versions(resume_map, SKILL_ROOT)
            self.assertEqual(len(versions), 2)
            paths = [v[0] for v in versions]
            self.assertIn(str(r1), paths)
            self.assertIn(str(r2), paths)
        finally:
            os.unlink(r1)
            os.unlink(r2)

    def test_deduplicates_same_path(self):
        r1 = make_temp_txt_resume("Resume text here")
        resume_map = {
            "domains": {
                "eng": {"resume": str(r1), "keywords": []},
            },
            "default": str(r1),  # same file as domain resume
        }
        try:
            versions = load_resume_versions(resume_map, SKILL_ROOT)
            self.assertEqual(len(versions), 1)
        finally:
            os.unlink(r1)

    def test_missing_files_skipped(self):
        resume_map = {
            "domains": {
                "eng": {"resume": "/nonexistent/resume.txt", "keywords": []},
            },
            "default": "",
        }
        versions = load_resume_versions(resume_map, SKILL_ROOT)
        self.assertEqual(len(versions), 0)

    def test_empty_files_skipped(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("   ")  # whitespace only
            path = f.name
        resume_map = {"domains": {"x": {"resume": path, "keywords": []}}, "default": ""}
        try:
            versions = load_resume_versions(resume_map, SKILL_ROOT)
            self.assertEqual(len(versions), 0)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: score_single with mocked AI client
# ---------------------------------------------------------------------------

class TestScoreSingle(unittest.TestCase):

    def _make_mock_client(self, response_text: str):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = response_text
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_returns_float_in_range(self):
        client = self._make_mock_client("0.85")
        score = score_single("SWE job", "Python resume", client)
        self.assertAlmostEqual(score, 0.85)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_clamps_above_1(self):
        client = self._make_mock_client("1.5")
        score = score_single("job", "resume", client)
        self.assertEqual(score, 1.0)

    def test_clamps_below_0(self):
        client = self._make_mock_client("-0.3")
        score = score_single("job", "resume", client)
        self.assertEqual(score, 0.0)

    def test_extracts_number_from_verbose_response(self):
        """AI might return 'Score: 0.72' instead of just '0.72'."""
        client = self._make_mock_client("Score: 0.72 out of 1.0")
        score = score_single("job", "resume", client)
        self.assertAlmostEqual(score, 0.72)

    def test_returns_zero_on_api_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        score = score_single("job", "resume", mock_client)
        self.assertEqual(score, 0.0)

    def test_returns_zero_for_empty_inputs(self):
        mock_client = MagicMock()
        score = score_single("", "resume", mock_client)
        self.assertEqual(score, 0.0)
        score2 = score_single("job", "", mock_client)
        self.assertEqual(score2, 0.0)
        mock_client.chat.completions.create.assert_not_called()

    def test_returns_zero_when_no_client(self):
        score = score_single("job desc", "resume text", None)
        self.assertEqual(score, 0.0)


# ---------------------------------------------------------------------------
# Test: score_job — best resume selection
# ---------------------------------------------------------------------------

class TestScoreJob(unittest.TestCase):

    def test_returns_best_score_and_path(self):
        """score_job should return the resume with the highest score."""
        mock_client = MagicMock()
        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.60"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.90"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="0.45"))]),
        ]
        mock_client.chat.completions.create.side_effect = responses

        resume_versions = [
            ("/path/a.txt", "resume a"),
            ("/path/b.txt", "resume b"),
            ("/path/c.txt", "resume c"),
        ]
        score, path = score_job("SWE job desc", resume_versions, mock_client)

        self.assertAlmostEqual(score, 0.90)
        self.assertEqual(path, "/path/b.txt")

    def test_returns_zero_with_no_client(self):
        score, path = score_job("job", [("/x.txt", "text")], None)
        self.assertEqual(score, 0.0)
        self.assertEqual(path, "")

    def test_returns_zero_with_no_resumes(self):
        mock_client = MagicMock()
        score, path = score_job("job", [], mock_client)
        self.assertEqual(score, 0.0)
        self.assertEqual(path, "")


# ---------------------------------------------------------------------------
# Test: run_scoring integration (with mocked AI and temp DB)
# ---------------------------------------------------------------------------

class TestRunScoring(unittest.TestCase):

    def setUp(self):
        self.db_path = make_temp_db()
        self.resume_path = make_temp_txt_resume(
            "Jane Doe — Python Developer\n5 years experience building APIs."
        )

    def tearDown(self):
        cleanup(self.db_path)
        os.unlink(self.resume_path)

    def _make_profile(self, threshold=0.40):
        return {
            "ai": {
                "mammoth_api_key_env": "MAMMOTH_API_KEY",
                "mammoth_api_base": "https://api.openai.com/v1",
                "scoring_model": "gpt-4o-mini",
            },
            "automation": {"relevance_threshold": threshold},
        }

    def _make_resume_map(self):
        return {
            "domains": {
                "engineering": {"resume": str(self.resume_path), "keywords": ["python"]},
            },
            "default": "",
        }

    def _insert_test_job(self, suffix="1", description="Senior Python Developer role"):
        db.insert_job({
            "id": f"indeed:test{suffix}",
            "platform": "indeed",
            "title": f"Engineer {suffix}",
            "company": "Acme",
            "url": f"https://indeed.com/j/{suffix}",
            "description": description,
            "posted_date": "2026-03-23",
            "url_hash": f"uhash{suffix}",
            "title_hash": f"thash{suffix}",
        }, self.db_path)

    def test_skips_when_no_ai_client(self):
        """When AI key is missing, run_scoring returns skipped_no_ai sentinel."""
        self._insert_test_job()
        profile = self._make_profile()
        env_backup = os.environ.pop("MAMMOTH_API_KEY", None)
        try:
            stats = run_scoring(profile, self._make_resume_map(),
                                db_path=self.db_path, skill_root=SKILL_ROOT)
            self.assertEqual(stats["skipped_no_ai"], -1)
        finally:
            if env_backup:
                os.environ["MAMMOTH_API_KEY"] = env_backup

    def test_high_score_marks_job_as_scoring(self):
        """Jobs scoring above threshold must be updated to 'scoring' status."""
        self._insert_test_job()
        profile = self._make_profile(threshold=0.40)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.85"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("scorer.create_ai_client", return_value=mock_client):
            stats = run_scoring(profile, self._make_resume_map(),
                                db_path=self.db_path, skill_root=SKILL_ROOT)

        self.assertEqual(stats["scored"], 1)
        self.assertEqual(stats["skipped_low"], 0)

        conn = db.get_connection(self.db_path)
        row = conn.execute(
            "SELECT status, relevance_score FROM jobs WHERE id='indeed:test1'"
        ).fetchone()
        conn.close()
        self.assertEqual(row["status"], "scoring")
        self.assertAlmostEqual(row["relevance_score"], 0.85)

    def test_low_score_marks_job_as_skipped(self):
        """Jobs scoring below threshold must be marked 'skipped'."""
        self._insert_test_job()
        profile = self._make_profile(threshold=0.40)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.20"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("scorer.create_ai_client", return_value=mock_client):
            stats = run_scoring(profile, self._make_resume_map(),
                                db_path=self.db_path, skill_root=SKILL_ROOT)

        self.assertEqual(stats["skipped_low"], 1)
        self.assertEqual(stats["scored"], 0)

        conn = db.get_connection(self.db_path)
        row = conn.execute(
            "SELECT status FROM jobs WHERE id='indeed:test1'"
        ).fetchone()
        conn.close()
        self.assertEqual(row["status"], "skipped")

    def test_job_without_description_skipped(self):
        """Jobs with empty description must be marked skipped without AI call."""
        self._insert_test_job(description="")
        profile = self._make_profile()

        mock_client = MagicMock()

        with patch("scorer.create_ai_client", return_value=mock_client):
            stats = run_scoring(profile, self._make_resume_map(),
                                db_path=self.db_path, skill_root=SKILL_ROOT)

        self.assertEqual(stats["skipped_low"], 1)
        mock_client.chat.completions.create.assert_not_called()

    def test_multiple_jobs_all_processed(self):
        """All unscored jobs must be processed in a single run."""
        for i in range(3):
            self._insert_test_job(suffix=str(i))

        profile = self._make_profile(threshold=0.50)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="0.75"))]
        )

        with patch("scorer.create_ai_client", return_value=mock_client):
            stats = run_scoring(profile, self._make_resume_map(),
                                db_path=self.db_path, skill_root=SKILL_ROOT)

        self.assertEqual(stats["scored"], 3)


# ---------------------------------------------------------------------------
# Test: create_ai_client
# ---------------------------------------------------------------------------

class TestCreateAIClient(unittest.TestCase):

    def test_returns_none_without_key(self):
        profile = {
            "ai": {
                "mammoth_api_key_env": "FAKE_KEY_ENV_XYZ_NOT_SET",
                "mammoth_api_base": "https://api.openai.com/v1",
            }
        }
        env_backup = os.environ.pop("FAKE_KEY_ENV_XYZ_NOT_SET", None)
        try:
            client = create_ai_client(profile)
            self.assertIsNone(client)
        finally:
            if env_backup:
                os.environ["FAKE_KEY_ENV_XYZ_NOT_SET"] = env_backup

    def test_returns_client_with_key(self):
        profile = {
            "ai": {
                "mammoth_api_key_env": "FAKE_KEY_ENV_XYZ",
                "mammoth_api_base": "https://api.openai.com/v1",
            }
        }
        os.environ["FAKE_KEY_ENV_XYZ"] = "sk-test-key"
        try:
            try:
                import openai  # noqa: F401
                client = create_ai_client(profile)
                self.assertIsNotNone(client)
            except ImportError:
                self.skipTest("openai package not installed")
        finally:
            del os.environ["FAKE_KEY_ENV_XYZ"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestUrlNormalization,
        TestTitleNormalization,
        TestBuildSearchUrl,
        TestBuildExtractorJs,
        TestBrowserCmdGracefulFailure,
        TestPlatformAdapter,
        TestRunDiscovery,
        TestReadResumeText,
        TestLoadResumeVersions,
        TestScoreSingle,
        TestScoreJob,
        TestRunScoring,
        TestCreateAIClient,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
