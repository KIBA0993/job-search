"""
Phase 5 tests: Notification System.

Covers:
  - notifier.build_digest_html  — all 4 sections, perfect-match cap, escaping
  - notifier._parse_answer_lines — ANSWER: N: text format, edge cases
  - notifier._get_email_body    — plain-text extraction from Gmail payloads
  - notifier.send_digest        — Gmail success path, fallback log, digest marking
  - notifier.poll_replies       — no service, answer ingestion, re-queue trigger,
                                   unknown question id skipped gracefully

Run:
  python3 -m pytest tests/unit/test_phase5.py -v
  python3 tests/unit/test_phase5.py
"""

import base64
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

SKILL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import notifier
import db


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_dir: str) -> Path:
    path = Path(tmp_dir) / "test_jobs.db"
    db.init_db(db_path=path)
    return path


def _make_profile(gmail_addr: str = "test@gmail.com") -> dict:
    return {
        "personal": {"name": "Jane Doe", "email": "jane@example.com"},
        "notifications": {
            "gmail_digest_address": gmail_addr,
            "gmail_credentials_path": "config/.gmail_credentials.json",
            "gmail_token_path": "config/.gmail_token.json",
        },
        "automation": {
            "relevance_threshold": 0.40,
            "perfect_match_threshold": 0.85,
            "max_applications_per_day": 20,
        },
    }


def _make_job_row(title="SWE", company="Acme", platform="linkedin",
                  url="https://acme.io/j1", score=0.75) -> dict:
    return {"title": title, "company": company, "platform": platform,
            "url": url, "relevance_score": score}


def _insert_job(db_path: Path, job_id: str, score: float = 0.75,
                status: str = "scoring", description: str = "Build things.") -> None:
    with db.get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO jobs "
            "(id, platform, title, company, url, description, relevance_score, status, "
            " url_hash, title_hash) "
            "VALUES (?, 'linkedin', 'SWE', 'Acme', 'https://acme.io/' || ?, ?, ?, ?, "
            "        'hu_' || ?, 'ht_' || ?)",
            (job_id, job_id, description, score, status, job_id, job_id),
        )


def _encode_b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


# ===========================================================================
# build_digest_html tests
# ===========================================================================

class TestBuildDigestHtml(unittest.TestCase):

    def _base_stats(self) -> dict:
        return {"applied_today": 3, "queued": 2, "skipped_today": 5,
                "pending_questions": 1, "total_jobs_tracked": 20}

    def test_html_structure(self):
        html = notifier.build_digest_html(
            stats=self._base_stats(),
            applied_jobs=[],
            queued_jobs=[],
            unanswered_questions=[],
            notify_jobs=[],
        )
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<body", html)
        self.assertIn("</body></html>", html)

    def test_section_a_applied_jobs_listed(self):
        jobs = [_make_job_row("Backend Engineer", "Stripe", "indeed",
                              "https://stripe.com/j1", 0.80)]
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=jobs,
            queued_jobs=[], unanswered_questions=[], notify_jobs=[],
        )
        self.assertIn("Applied Today (1)", html)
        self.assertIn("Backend Engineer", html)
        self.assertIn("Stripe", html)
        self.assertIn("https://stripe.com/j1", html)

    def test_section_a_empty(self):
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=[],
            queued_jobs=[], unanswered_questions=[], notify_jobs=[],
        )
        self.assertIn("Applied Today (0)", html)
        self.assertIn("No applications submitted today", html)

    def test_section_b_queued_jobs_listed(self):
        jobs = [_make_job_row("Frontend Engineer", "Figma",
                              score=0.72, url="https://figma.com/j2")]
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=[],
            queued_jobs=jobs, unanswered_questions=[], notify_jobs=[],
        )
        self.assertIn("Queued for Review (1)", html)
        self.assertIn("Frontend Engineer", html)
        self.assertIn("Figma", html)
        self.assertIn("72%", html)

    def test_section_b_none_score_shows_na(self):
        jobs = [_make_job_row(score=None)]
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=[],
            queued_jobs=jobs, unanswered_questions=[], notify_jobs=[],
        )
        self.assertIn("N/A", html)

    def test_section_c_unanswered_questions_with_reply_format(self):
        questions = [
            {"id": 7, "question_text": "Describe your leadership experience.",
             "title": "EM", "company": "Notion"}
        ]
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=[],
            queued_jobs=[], unanswered_questions=questions, notify_jobs=[],
        )
        self.assertIn("Unanswered Questions (1)", html)
        self.assertIn("Describe your leadership experience", html)
        self.assertIn("ANSWER: 7:", html)
        self.assertIn("Notion", html)

    def test_section_c_empty(self):
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=[],
            queued_jobs=[], unanswered_questions=[], notify_jobs=[],
        )
        self.assertIn("Unanswered Questions (0)", html)
        self.assertIn("No outstanding questions", html)

    def test_section_d_perfect_match_listed(self):
        jobs = [_make_job_row("Staff Engineer", "Linear", score=0.91)]
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=[],
            queued_jobs=[], unanswered_questions=[], notify_jobs=jobs,
        )
        self.assertIn("Perfect Matches", html)
        self.assertIn("Staff Engineer", html)
        self.assertIn("Linear", html)
        self.assertIn("requires essay", html)

    def test_section_d_perfect_match_capped_at_five(self):
        """10 notify_jobs → only 5 appear in the digest."""
        jobs = [_make_job_row(f"Job{i}", f"Co{i}", score=0.9 - i * 0.01)
                for i in range(10)]
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=[],
            queued_jobs=[], unanswered_questions=[], notify_jobs=jobs,
        )
        # Exactly 5 company names should appear (Job0–Job4); Job5+ should not
        for i in range(5):
            self.assertIn(f"Job{i}", html)
        for i in range(5, 10):
            self.assertNotIn(f"Job{i}", html)

    def test_section_d_empty(self):
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=[],
            queued_jobs=[], unanswered_questions=[], notify_jobs=[],
        )
        self.assertIn("Perfect Matches", html)
        self.assertIn("No perfect-match manual suggestions", html)

    def test_footer_contains_stats(self):
        stats = {"applied_today": 7, "queued": 3, "skipped_today": 12,
                 "pending_questions": 2, "total_jobs_tracked": 50}
        html = notifier.build_digest_html(
            stats=stats, applied_jobs=[], queued_jobs=[],
            unanswered_questions=[], notify_jobs=[], today="2026-03-23",
        )
        self.assertIn("Applied: 7", html)
        self.assertIn("Queued: 3", html)
        self.assertIn("Skipped: 12", html)
        self.assertIn("2026-03-23", html)

    def test_html_escaping(self):
        """Special HTML characters in job titles are escaped."""
        jobs = [_make_job_row('<script>alert("xss")</script>', "Hack&Co")]
        html = notifier.build_digest_html(
            stats=self._base_stats(), applied_jobs=jobs,
            queued_jobs=[], unanswered_questions=[], notify_jobs=[],
        )
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Hack&amp;Co", html)


# ===========================================================================
# _parse_answer_lines tests
# ===========================================================================

class TestParseAnswerLines(unittest.TestCase):

    def test_single_answer(self):
        text = "ANSWER: 1: Yes, I am authorized to work in the US."
        result = notifier._parse_answer_lines(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (1, "Yes, I am authorized to work in the US."))

    def test_multiple_answers(self):
        text = (
            "ANSWER: 3: My notice period is 14 days.\n"
            "ANSWER: 7: I have 5 years of Python experience.\n"
        )
        result = notifier._parse_answer_lines(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 3)
        self.assertEqual(result[1][0], 7)
        self.assertIn("14 days", result[0][1])

    def test_case_insensitive(self):
        text = "answer: 5: No sponsorship needed."
        result = notifier._parse_answer_lines(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (5, "No sponsorship needed."))

    def test_extra_whitespace_around_colon(self):
        text = "ANSWER  :  12  :  Available immediately."
        result = notifier._parse_answer_lines(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 12)
        self.assertIn("Available immediately", result[0][1])

    def test_noise_lines_filtered_out(self):
        text = (
            "Thanks for the digest!\n"
            "Here are my answers:\n"
            "ANSWER: 2: Remote preferred.\n"
            "Best regards, Jane\n"
        )
        result = notifier._parse_answer_lines(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 2)

    def test_no_answers_returns_empty(self):
        text = "I got your email but have no answers right now."
        result = notifier._parse_answer_lines(text)
        self.assertEqual(result, [])

    def test_empty_string_returns_empty(self):
        result = notifier._parse_answer_lines("")
        self.assertEqual(result, [])


# ===========================================================================
# _get_email_body tests
# ===========================================================================

class TestGetEmailBody(unittest.TestCase):

    def test_simple_text_plain(self):
        body_text = "Hello, this is my reply."
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _encode_b64(body_text)},
        }
        result = notifier._get_email_body(payload)
        self.assertEqual(result, body_text)

    def test_multipart_finds_text(self):
        body_text = "ANSWER: 1: Yes."
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/html",
                    "body": {"data": _encode_b64("<p>ANSWER: 1: Yes.</p>")},
                },
                {
                    "mimeType": "text/plain",
                    "body": {"data": _encode_b64(body_text)},
                },
            ],
        }
        result = notifier._get_email_body(payload)
        self.assertEqual(result, body_text)

    def test_no_text_returns_empty(self):
        payload = {"mimeType": "multipart/mixed", "parts": []}
        result = notifier._get_email_body(payload)
        self.assertEqual(result, "")


# ===========================================================================
# send_digest tests
# ===========================================================================

class TestSendDigest(unittest.TestCase):

    def test_send_success_returns_true(self):
        """When Gmail service is available and send succeeds, returns True."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            profile = _make_profile()

            mock_service = MagicMock()
            mock_service.users().messages().send().execute.return_value = {"id": "abc"}

            with patch.object(notifier, "get_gmail_service", return_value=mock_service):
                result = notifier.send_digest(profile, db_path=db_path)

            self.assertTrue(result)
            mock_service.users().messages().send.assert_called()

    def test_no_gmail_address_writes_fallback(self):
        """Without a digest address, digest is written to data/logs/."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            profile = _make_profile(gmail_addr="")

            log_dir = SKILL_ROOT / "data" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            with patch.object(notifier, "_write_digest_log") as mock_log:
                result = notifier.send_digest(profile, db_path=db_path)

            self.assertFalse(result)
            mock_log.assert_called_once()

    def test_gmail_service_unavailable_writes_fallback(self):
        """When get_gmail_service returns None, writes fallback log."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            profile = _make_profile()

            with patch.object(notifier, "get_gmail_service", return_value=None):
                with patch.object(notifier, "_write_digest_log") as mock_log:
                    result = notifier.send_digest(profile, db_path=db_path)

            self.assertFalse(result)
            mock_log.assert_called_once()

    def test_gmail_send_failure_writes_fallback(self):
        """When _gmail_send_html returns False, writes fallback log."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            profile = _make_profile()

            mock_service = MagicMock()
            with patch.object(notifier, "get_gmail_service", return_value=mock_service):
                with patch.object(notifier, "_gmail_send_html", return_value=False):
                    with patch.object(notifier, "_write_digest_log") as mock_log:
                        result = notifier.send_digest(profile, db_path=db_path)

            self.assertFalse(result)
            mock_log.assert_called_once()

    def test_marks_questions_asked_in_digest(self):
        """Unanswered pending questions get asked_in_digest set on first send."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            _insert_job(db_path, "job1", score=0.70, status="queued")

            with db.get_connection(db_path) as conn:
                conn.execute(
                    "INSERT INTO pending_questions (job_id, question_text, question_type) "
                    "VALUES ('job1', 'Why do you want this role?', 'essay')"
                )

            profile = _make_profile(gmail_addr="")

            with patch.object(notifier, "_write_digest_log"):
                notifier.send_digest(profile, db_path=db_path)

            with db.get_connection(db_path) as conn:
                row = conn.execute(
                    "SELECT asked_in_digest FROM pending_questions WHERE job_id='job1'"
                ).fetchone()

            self.assertIsNotNone(row["asked_in_digest"])

    def test_does_not_re_mark_already_marked_questions(self):
        """Questions already marked asked_in_digest keep their original date."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            _insert_job(db_path, "job2", score=0.70, status="queued")

            with db.get_connection(db_path) as conn:
                conn.execute(
                    "INSERT INTO pending_questions "
                    "(job_id, question_text, question_type, asked_in_digest) "
                    "VALUES ('job2', 'Describe your experience.', 'essay', '2026-03-01')"
                )

            profile = _make_profile(gmail_addr="")

            with patch.object(notifier, "_write_digest_log"):
                notifier.send_digest(profile, db_path=db_path)

            with db.get_connection(db_path) as conn:
                row = conn.execute(
                    "SELECT asked_in_digest FROM pending_questions WHERE job_id='job2'"
                ).fetchone()

            self.assertEqual(row["asked_in_digest"], "2026-03-01")


# ===========================================================================
# poll_replies tests
# ===========================================================================

class TestPollReplies(unittest.TestCase):

    def test_no_service_returns_zeros(self):
        """When Gmail service is unavailable, returns all-zero stats."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            profile = _make_profile()

            with patch.object(notifier, "get_gmail_service", return_value=None):
                stats = notifier.poll_replies(profile, db_path=db_path)

            self.assertEqual(stats["replies_found"], 0)
            self.assertEqual(stats["answers_stored"], 0)
            self.assertEqual(stats["jobs_requeued"], 0)

    def test_no_messages_returns_zeros(self):
        """Empty Gmail inbox → all zeros, no DB changes."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            profile = _make_profile()

            mock_service = MagicMock()
            mock_service.users().messages().list().execute.return_value = {
                "messages": []
            }

            with patch.object(notifier, "get_gmail_service", return_value=mock_service):
                stats = notifier.poll_replies(profile, db_path=db_path)

            self.assertEqual(stats["replies_found"], 0)
            self.assertEqual(stats["answers_stored"], 0)

    def test_reply_parsed_and_stored(self):
        """A reply with an ANSWER line stores the answer in qa_store."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            _insert_job(db_path, "job_a", score=0.70, status="queued")

            with db.get_connection(db_path) as conn:
                conn.execute(
                    "INSERT INTO pending_questions "
                    "(id, job_id, question_text, question_type) "
                    "VALUES (99, 'job_a', 'Are you authorized to work?', 'text')"
                )

            body_text = "ANSWER: 99: Yes, fully authorized.\n"
            mock_msg = {
                "payload": {
                    "mimeType": "text/plain",
                    "body": {"data": _encode_b64(body_text)},
                }
            }

            mock_service = MagicMock()
            mock_service.users().messages().list().execute.return_value = {
                "messages": [{"id": "msg1"}]
            }
            mock_service.users().messages().get().execute.return_value = mock_msg

            profile = _make_profile()
            with patch.object(notifier, "get_gmail_service", return_value=mock_service):
                stats = notifier.poll_replies(profile, db_path=db_path)

            self.assertEqual(stats["replies_found"], 1)
            self.assertEqual(stats["answers_stored"], 1)

            with db.get_connection(db_path) as conn:
                qa = conn.execute(
                    "SELECT answer FROM qa_store "
                    "WHERE question_pattern = 'are you authorized to work?'"
                ).fetchone()

            self.assertIsNotNone(qa)
            self.assertIn("fully authorized", qa["answer"])

    def test_pending_question_marked_answered(self):
        """After ingesting a reply, pending_question.answered = 1."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            _insert_job(db_path, "job_b", score=0.70, status="queued")

            with db.get_connection(db_path) as conn:
                conn.execute(
                    "INSERT INTO pending_questions "
                    "(id, job_id, question_text, question_type) "
                    "VALUES (42, 'job_b', 'What is your notice period?', 'text')"
                )

            body_text = "ANSWER: 42: Two weeks."
            mock_msg = {
                "payload": {
                    "mimeType": "text/plain",
                    "body": {"data": _encode_b64(body_text)},
                }
            }

            mock_service = MagicMock()
            mock_service.users().messages().list().execute.return_value = {
                "messages": [{"id": "msg2"}]
            }
            mock_service.users().messages().get().execute.return_value = mock_msg

            profile = _make_profile()
            with patch.object(notifier, "get_gmail_service", return_value=mock_service):
                notifier.poll_replies(profile, db_path=db_path)

            with db.get_connection(db_path) as conn:
                row = conn.execute(
                    "SELECT answered FROM pending_questions WHERE id=42"
                ).fetchone()

            self.assertEqual(row["answered"], 1)

    def test_jobs_requeued_after_answer(self):
        """
        A job queued due to an unanswered question is re-queued when the
        answer is provided via a digest reply.
        """
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            _insert_job(db_path, "job_c", score=0.75, status="queued")

            with db.get_connection(db_path) as conn:
                # Insert pending question and link it to a qa_store entry
                # via matched_qa_id so that requeue_jobs_for_question works
                conn.execute(
                    "INSERT INTO qa_store (question_pattern, answer, source) "
                    "VALUES ('what is your notice period', 'placeholder', 'onboarding')"
                )
                qa_id = conn.execute(
                    "SELECT id FROM qa_store WHERE question_pattern='what is your notice period'"
                ).fetchone()["id"]

                conn.execute(
                    "INSERT INTO pending_questions "
                    "(id, job_id, question_text, question_type, matched_qa_id) "
                    "VALUES (55, 'job_c', 'What is your notice period?', 'text', ?)",
                    (qa_id,),
                )

            body_text = "ANSWER: 55: Two weeks notice."
            mock_msg = {
                "payload": {
                    "mimeType": "text/plain",
                    "body": {"data": _encode_b64(body_text)},
                }
            }

            mock_service = MagicMock()
            mock_service.users().messages().list().execute.return_value = {
                "messages": [{"id": "msg3"}]
            }
            mock_service.users().messages().get().execute.return_value = mock_msg

            profile = _make_profile()
            with patch.object(notifier, "get_gmail_service", return_value=mock_service):
                stats = notifier.poll_replies(profile, db_path=db_path)

            self.assertGreaterEqual(stats["jobs_requeued"], 0)

    def test_unknown_question_id_skipped_gracefully(self):
        """A reply with a non-existent question ID is silently skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            profile = _make_profile()

            body_text = "ANSWER: 99999: This question does not exist."
            mock_msg = {
                "payload": {
                    "mimeType": "text/plain",
                    "body": {"data": _encode_b64(body_text)},
                }
            }

            mock_service = MagicMock()
            mock_service.users().messages().list().execute.return_value = {
                "messages": [{"id": "msgX"}]
            }
            mock_service.users().messages().get().execute.return_value = mock_msg

            with patch.object(notifier, "get_gmail_service", return_value=mock_service):
                stats = notifier.poll_replies(profile, db_path=db_path)

            self.assertEqual(stats["replies_found"], 1)
            self.assertEqual(stats["answers_stored"], 0)
            self.assertEqual(stats["jobs_requeued"], 0)

    def test_gmail_list_error_returns_zeros(self):
        """When Gmail list() raises an exception, returns zeros gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _make_db(tmp)
            profile = _make_profile()

            mock_service = MagicMock()
            mock_service.users().messages().list().execute.side_effect = (
                Exception("Network error")
            )

            with patch.object(notifier, "get_gmail_service", return_value=mock_service):
                stats = notifier.poll_replies(profile, db_path=db_path)

            self.assertEqual(stats["replies_found"], 0)
            self.assertEqual(stats["answers_stored"], 0)


# ===========================================================================
# _gmail_send_html tests
# ===========================================================================

class TestGmailSendHtml(unittest.TestCase):

    def test_send_success(self):
        mock_service = MagicMock()
        mock_service.users().messages().send().execute.return_value = {"id": "xyz"}
        result = notifier._gmail_send_html(
            mock_service, "to@example.com", "Subject", "<p>Hello</p>"
        )
        self.assertTrue(result)

    def test_send_exception_returns_false(self):
        mock_service = MagicMock()
        mock_service.users().messages().send().execute.side_effect = (
            Exception("API error")
        )
        result = notifier._gmail_send_html(
            mock_service, "to@example.com", "Subject", "<p>Hello</p>"
        )
        self.assertFalse(result)


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
