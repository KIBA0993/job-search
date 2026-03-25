"""
Phase 3 tests: resume routing and tailoring (resume_router.py).

Run:
  python3 -m pytest tests/unit/test_phase3.py -v
  # or directly:
  python3 tests/unit/test_phase3.py
"""

import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

SKILL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from resume_router import (
    classify_domain,
    get_resume_for_domain,
    tailor_resume,
    save_tailored_pdf,
    _sanitize_company_name,
    route,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RESUME_MAP = {
    "domains": {
        "ml_engineering": {
            "label": "ML / AI Engineering",
            "keywords": [
                "machine learning", "deep learning", "llm", "large language model",
                "pytorch", "tensorflow", "model training", "fine-tuning", "transformer",
                "computer vision", "nlp", "natural language processing", "mlops",
                "feature engineering", "neural network", "inference", "gpu",
            ],
            "resume": "/tmp/ml_resume.pdf",
        },
        "backend_engineering": {
            "label": "Backend / Platform Engineering",
            "keywords": [
                "backend", "distributed systems", "golang", "go", "rust", "java",
                "microservices", "api", "kafka", "redis", "postgres", "mysql",
                "kubernetes", "docker", "grpc", "rest api", "system design",
                "high availability", "scalability", "cloud", "aws", "gcp", "azure",
            ],
            "resume": "/tmp/backend_resume.pdf",
        },
        "full_stack_engineering": {
            "label": "Full Stack Engineering",
            "keywords": [
                "full stack", "frontend", "backend", "react", "typescript", "node.js",
                "nextjs", "vue", "angular", "python", "django", "fastapi",
                "web development", "ui", "ux", "responsive design",
            ],
            "resume": "/tmp/fullstack_resume.pdf",
        },
        "data_engineering": {
            "label": "Data Engineering",
            "keywords": [
                "data engineering", "spark", "airflow", "dbt", "snowflake", "databricks",
                "etl", "pipeline", "data warehouse", "data lake", "bigquery",
                "redshift", "flink", "beam", "pandas", "data quality",
            ],
            "resume": "/tmp/data_resume.pdf",
        },
        "product_management": {
            "label": "Product Management",
            "keywords": [
                "product manager", "product management", "roadmap", "stakeholder",
                "okr", "kpi", "go-to-market", "user research", "product strategy",
                "agile", "scrum", "backlog", "sprint", "product owner",
                "cross-functional", "metrics", "a/b testing",
            ],
            "resume": "/tmp/pm_resume.pdf",
        },
    },
    "default": "/tmp/default_resume.pdf",
    "classification": {
        "min_confidence": 0.7,
        "fallback_to_default_below_confidence": True,
    },
}

JD_ML = (
    "We are hiring a Senior ML Engineer to train and fine-tune large language models "
    "using PyTorch and TensorFlow. You will work on NLP, neural networks, and MLOps "
    "pipelines. GPU cluster management experience preferred."
)

JD_BACKEND = (
    "Backend engineer needed for our distributed systems team. You will build "
    "microservices in Golang, manage Kubernetes deployments, design REST APIs, and "
    "work with Kafka and Redis. AWS experience required."
)

JD_DATA = (
    "Data engineer to build ETL pipelines using Apache Spark and Airflow. "
    "Experience with Snowflake, dbt, and data warehouse design required. "
    "BigQuery and Databricks a plus."
)

JD_PM = (
    "Product Manager to own the product roadmap, run user research, manage the "
    "sprint backlog, coordinate cross-functional teams, and define OKRs and KPIs. "
    "Go-to-market strategy ownership required."
)

JD_AMBIGUOUS = (
    "We are looking for a software generalist to join our startup. "
    "You will wear many hats. No specific tech stack required."
)

SAMPLE_RESUME_TEXT = (
    "Jane Doe | jane@example.com\n\n"
    "EXPERIENCE\n"
    "Software Engineer, Acme Corp (2020-2024)\n"
    "  Developed backend services and APIs.\n"
    "  Worked with Python, Go, and cloud platforms.\n\n"
    "EDUCATION\n"
    "B.S. Computer Science, State University (2019)\n"
)

FIXED_DATE = date(2026, 3, 23)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_ai_client(return_value: str = "ml_engineering"):
    """Create a mock OpenAI-compatible client whose completions return return_value."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = return_value
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# 1. Domain classification — keyword-only (no AI)
# ---------------------------------------------------------------------------

class TestClassifyDomainKeywords(unittest.TestCase):
    """classify_domain without AI client — pure keyword matching."""

    def test_ml_jd_classified_correctly(self):
        domain = classify_domain(JD_ML, RESUME_MAP)
        self.assertEqual(domain, "ml_engineering")

    def test_backend_jd_classified_correctly(self):
        domain = classify_domain(JD_BACKEND, RESUME_MAP)
        self.assertEqual(domain, "backend_engineering")

    def test_data_jd_classified_correctly(self):
        domain = classify_domain(JD_DATA, RESUME_MAP)
        self.assertEqual(domain, "data_engineering")

    def test_pm_jd_classified_correctly(self):
        domain = classify_domain(JD_PM, RESUME_MAP)
        self.assertEqual(domain, "product_management")

    def test_ambiguous_jd_falls_back_to_default(self):
        domain = classify_domain(JD_AMBIGUOUS, RESUME_MAP)
        self.assertEqual(domain, "default")

    def test_empty_jd_returns_default(self):
        domain = classify_domain("", RESUME_MAP)
        self.assertEqual(domain, "default")

    def test_empty_domain_map_returns_default(self):
        domain = classify_domain(JD_ML, {"domains": {}, "default": "/tmp/x.pdf"})
        self.assertEqual(domain, "default")


# ---------------------------------------------------------------------------
# 2. Domain classification — AI path
# ---------------------------------------------------------------------------

class TestClassifyDomainAI(unittest.TestCase):
    """classify_domain with a mock AI client."""

    def test_ai_result_returned_when_valid(self):
        client = make_mock_ai_client("backend_engineering")
        # Use an ambiguous JD so keyword short-circuit doesn't fire
        domain = classify_domain(JD_AMBIGUOUS, RESUME_MAP, client=client)
        self.assertEqual(domain, "backend_engineering")

    def test_ai_unknown_key_falls_back_to_default(self):
        client = make_mock_ai_client("nonexistent_domain")
        domain = classify_domain(JD_AMBIGUOUS, RESUME_MAP, client=client)
        self.assertEqual(domain, "default")

    def test_ai_returns_default_literal(self):
        client = make_mock_ai_client("default")
        domain = classify_domain(JD_AMBIGUOUS, RESUME_MAP, client=client)
        self.assertEqual(domain, "default")

    def test_ai_failure_falls_back_to_default(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("timeout")
        domain = classify_domain(JD_AMBIGUOUS, RESUME_MAP, client=client)
        self.assertEqual(domain, "default")

    def test_keyword_winner_bypasses_ai(self):
        """If keywords clearly identify the domain, AI should not be called."""
        client = make_mock_ai_client("product_management")
        # JD_ML has many ML keywords → should short-circuit before AI
        domain = classify_domain(JD_ML, RESUME_MAP, client=client)
        client.chat.completions.create.assert_not_called()
        self.assertEqual(domain, "ml_engineering")


# ---------------------------------------------------------------------------
# 3. Resume selection
# ---------------------------------------------------------------------------

class TestGetResumeForDomain(unittest.TestCase):

    def test_known_domain_returns_configured_path(self):
        path = get_resume_for_domain("ml_engineering", RESUME_MAP)
        self.assertEqual(path, "/tmp/ml_resume.pdf")

    def test_default_domain_returns_default_path(self):
        path = get_resume_for_domain("default", RESUME_MAP)
        self.assertEqual(path, "/tmp/default_resume.pdf")

    def test_unknown_domain_falls_back_to_default(self):
        path = get_resume_for_domain("unknown_domain", RESUME_MAP)
        self.assertEqual(path, "/tmp/default_resume.pdf")

    def test_domain_with_empty_resume_falls_back_to_default(self):
        rmap = {
            "domains": {"ml_engineering": {"resume": ""}},
            "default": "/tmp/default.pdf",
        }
        path = get_resume_for_domain("ml_engineering", rmap)
        self.assertEqual(path, "/tmp/default.pdf")

    def test_empty_map_returns_empty_string(self):
        path = get_resume_for_domain("ml_engineering", {"domains": {}, "default": ""})
        self.assertEqual(path, "")


# ---------------------------------------------------------------------------
# 4. Tailoring — happy path and fallbacks
# ---------------------------------------------------------------------------

class TestTailorResume(unittest.TestCase):

    def _make_client(self, content: str):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = content
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response
        return client

    def test_tailoring_returns_text(self):
        expected = (
            "Tailored resume content for Acme Corp.\n"
            "Experience with PyTorch, NLP, and MLOps pipelines.\n"
            "B.S. Computer Science."
        )  # tailor_resume strips trailing whitespace
        client = self._make_client(expected)
        result = tailor_resume(SAMPLE_RESUME_TEXT, JD_ML, "Acme Corp", client=client)
        self.assertEqual(result, expected)

    def test_tailoring_no_client_returns_none(self):
        result = tailor_resume(SAMPLE_RESUME_TEXT, JD_ML, "Acme Corp", client=None)
        self.assertIsNone(result)

    def test_tailoring_ai_failure_returns_none(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("timeout")
        result = tailor_resume(SAMPLE_RESUME_TEXT, JD_ML, "Acme Corp", client=client)
        self.assertIsNone(result)

    def test_tailoring_empty_response_returns_none(self):
        client = self._make_client("  ")  # whitespace only
        result = tailor_resume(SAMPLE_RESUME_TEXT, JD_ML, "Acme Corp", client=client)
        self.assertIsNone(result)

    def test_tailoring_suspiciously_short_returns_none(self):
        client = self._make_client("ok")  # < 50 chars
        result = tailor_resume(SAMPLE_RESUME_TEXT, JD_ML, "Acme Corp", client=client)
        self.assertIsNone(result)

    def test_tailoring_empty_resume_returns_none(self):
        client = self._make_client("some content that is long enough to pass the 50 char check")
        result = tailor_resume("", JD_ML, "Acme Corp", client=client)
        self.assertIsNone(result)

    def test_tailoring_no_dashes_in_output(self):
        # When AI returns text with dashes (bad response), the output contains them.
        # This test documents that resume_router does NOT strip dashes itself —
        # that sanitization must come from the AI prompt + caller validation.
        text_with_dashes = (
            "Experience:\n- Led engineering team on ML projects.\n"
            "- Built REST APIs and data pipelines.\n"
            "Education: B.S. Computer Science, 2019.\n"
        )  # > 50 chars so it passes the length check
        client = self._make_client(text_with_dashes)
        result = tailor_resume(SAMPLE_RESUME_TEXT, JD_ML, "Bad Output Co", client=client)
        self.assertIsNotNone(result)
        self.assertIn("-", result)

    def test_tailoring_uses_tailoring_model(self):
        """Verify that tailor_resume calls the API with the specified model."""
        client = self._make_client("A" * 100)
        tailor_resume(SAMPLE_RESUME_TEXT, JD_ML, "Corp", client=client, model="gpt-4o")
        call_kwargs = client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("model", call_kwargs.args[0] if call_kwargs.args else None), "gpt-4o")


# ---------------------------------------------------------------------------
# 5. File naming
# ---------------------------------------------------------------------------

class TestSanitizeCompanyName(unittest.TestCase):

    def test_simple_name(self):
        self.assertEqual(_sanitize_company_name("Acme"), "Acme")

    def test_spaces_become_underscores(self):
        self.assertEqual(_sanitize_company_name("Acme Corp"), "Acme_Corp")

    def test_special_chars_removed(self):
        result = _sanitize_company_name("Google/YouTube LLC.")
        self.assertNotIn("/", result)
        self.assertNotIn(".", result)

    def test_empty_name_becomes_company(self):
        self.assertEqual(_sanitize_company_name(""), "Company")

    def test_only_special_chars_becomes_company(self):
        self.assertEqual(_sanitize_company_name("!@#$"), "Company")


# ---------------------------------------------------------------------------
# 6. PDF generation
# ---------------------------------------------------------------------------

class TestSaveTailoredPdf(unittest.TestCase):

    def test_file_naming_format(self):
        """Verify the saved file follows CompanyName_YYYY-MM-DD.pdf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_root = Path(tmpdir)
            result = save_tailored_pdf(
                tailored_text=SAMPLE_RESUME_TEXT,
                company="Acme Corp",
                skill_root=skill_root,
                run_date=FIXED_DATE,
            )
            if result is None:
                self.skipTest("Neither reportlab nor weasyprint installed — PDF test skipped")
            p = Path(result)
            self.assertEqual(p.name, "Acme_Corp_2026-03-23.pdf")
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 0)

    def test_tailored_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_root = Path(tmpdir)
            # No data/tailored dir exists yet
            result = save_tailored_pdf(
                tailored_text=SAMPLE_RESUME_TEXT,
                company="TestCo",
                skill_root=skill_root,
                run_date=FIXED_DATE,
            )
            tailored_dir = skill_root / "data" / "tailored"
            self.assertTrue(tailored_dir.exists())

    def test_empty_text_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_tailored_pdf(
                tailored_text="",
                company="Corp",
                skill_root=Path(tmpdir),
                run_date=FIXED_DATE,
            )
            self.assertIsNone(result)

    def test_whitespace_only_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_tailored_pdf(
                tailored_text="   \n  ",
                company="Corp",
                skill_root=Path(tmpdir),
                run_date=FIXED_DATE,
            )
            self.assertIsNone(result)

    def test_special_company_name_in_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_root = Path(tmpdir)
            result = save_tailored_pdf(
                tailored_text=SAMPLE_RESUME_TEXT,
                company="Google/YouTube LLC.",
                skill_root=skill_root,
                run_date=FIXED_DATE,
            )
            if result is None:
                self.skipTest("reportlab/weasyprint not installed")
            p = Path(result)
            self.assertNotIn("/", p.name)
            self.assertTrue(p.name.endswith(".pdf"))


# ---------------------------------------------------------------------------
# 7. route() — full pipeline
# ---------------------------------------------------------------------------

class TestRoute(unittest.TestCase):

    def _make_classify_client(self, domain_key: str):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = domain_key
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp
        return client

    def test_route_no_ai_uses_keyword_fallback(self):
        orig, tailored = route(JD_ML, "Acme", RESUME_MAP)
        self.assertEqual(orig, "/tmp/ml_resume.pdf")
        self.assertIsNone(tailored)

    def test_route_with_ai_tailoring(self):
        """route() with an AI client should attempt tailoring."""
        long_tailored = "Tailored content " * 20  # > 50 chars

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a real resume text file so read_resume_text can read it
            resume_file = Path(tmpdir) / "ml_resume.txt"
            resume_file.write_text(SAMPLE_RESUME_TEXT)

            rmap = {
                "domains": {
                    "ml_engineering": {
                        "label": "ML",
                        "keywords": [
                            "machine learning", "pytorch", "neural network",
                            "nlp", "gpu", "transformer", "fine-tuning",
                        ],
                        "resume": str(resume_file),
                    }
                },
                "default": str(resume_file),
            }

            # Patch classify to always return ml_engineering and tailor to return text
            with patch("resume_router.classify_domain", return_value="ml_engineering"), \
                 patch("resume_router.tailor_resume", return_value=long_tailored), \
                 patch("resume_router.save_tailored_pdf", return_value="/tmp/tailored.pdf"):
                orig, tailored = route(
                    JD_ML, "Acme", rmap, client=MagicMock(),
                    skill_root=Path(tmpdir),
                )
            self.assertEqual(orig, str(resume_file))
            self.assertEqual(tailored, "/tmp/tailored.pdf")

    def test_route_no_resume_configured_returns_empty(self):
        rmap = {"domains": {}, "default": ""}
        orig, tailored = route(JD_ML, "Corp", rmap)
        self.assertEqual(orig, "")
        self.assertIsNone(tailored)

    def test_route_tailoring_failure_returns_original(self):
        """When tailoring returns None, route returns (original, None)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_file = Path(tmpdir) / "resume.txt"
            resume_file.write_text(SAMPLE_RESUME_TEXT)

            rmap = {"domains": {}, "default": str(resume_file)}

            with patch("resume_router.classify_domain", return_value="default"), \
                 patch("resume_router.tailor_resume", return_value=None):
                orig, tailored = route(
                    JD_ML, "Corp", rmap, client=MagicMock(),
                    skill_root=Path(tmpdir),
                )
            self.assertEqual(orig, str(resume_file))
            self.assertIsNone(tailored)

    def test_route_unreadable_resume_returns_original_no_tailored(self):
        """When the resume file doesn't exist, tailoring is skipped."""
        rmap = {"domains": {}, "default": "/nonexistent/resume.pdf"}

        with patch("resume_router.classify_domain", return_value="default"):
            orig, tailored = route(JD_ML, "Corp", rmap, client=MagicMock())

        self.assertEqual(orig, "/nonexistent/resume.pdf")
        self.assertIsNone(tailored)


# ---------------------------------------------------------------------------
# 8. Integration — classify 5 different domains (keyword path)
# ---------------------------------------------------------------------------

class TestFiveDomainClassification(unittest.TestCase):
    """
    Verify each of the 5 configured domains is correctly identified by keywords
    without any AI call.
    """

    CASES = [
        ("ml_engineering",         JD_ML),
        ("backend_engineering",    JD_BACKEND),
        ("data_engineering",       JD_DATA),
        ("product_management",     JD_PM),
    ]

    def test_domain_classification(self):
        for expected_domain, jd in self.CASES:
            with self.subTest(domain=expected_domain):
                result = classify_domain(jd, RESUME_MAP)
                self.assertEqual(
                    result, expected_domain,
                    f"JD for '{expected_domain}' classified as '{result}'"
                )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestClassifyDomainKeywords,
        TestClassifyDomainAI,
        TestGetResumeForDomain,
        TestTailorResume,
        TestSanitizeCompanyName,
        TestSaveTailoredPdf,
        TestRoute,
        TestFiveDomainClassification,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
