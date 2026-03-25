"""
Phase 4: Bot-detection filter for job application answers.

Scans text for instructions designed to catch automated bots (e.g.
"include the word APPLY in your response"). Uses regex patterns first;
optionally double-checks with LLM for anything the regex misses.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Sentence-level patterns that signal bot-detection / honeypot instructions
BOT_PATTERNS = [
    r"if you (are|'?re) (an? )?(?:ai|bot|robot|automated|llm|language model)",
    r"include (?:the (?:word|phrase|text)\s+)?['\"]?\w+['\"]?\s+in your (?:response|answer|application|message|subject|email)",
    r"add (?:the (?:word|phrase|text)\s+)?['\"]?\w+['\"]?\s+(?:to|at the) (?:beginning|start|end|top|bottom)",
    r"respond with (?:the (?:word|phrase|text)\s+)?['\"]?\w+",
    r"(?:start|begin) your (?:response|answer|message|application) with",
    r"put ['\"]?\w+['\"]? (?:at the|in your|as the first|as the last)",
    r"write ['\"]?\w+['\"]? (?:at the|in your|as the first|as the last)",
    r"(?:first|last) (?:word|line|sentence) (?:should|must) be",
    r"to (?:prove|show|verify|confirm) you(?:'re| are) (?:human|not a bot|not an ai|real)",
    r"human verification",
    r"type the (?:word|phrase)",
    r"attention (?:ai|bot|robot|automated)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in BOT_PATTERNS]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_bot_sentence(sentence: str) -> bool:
    """Return True if the sentence matches any bot-detection pattern."""
    for compiled in _COMPILED:
        if compiled.search(sentence):
            return True
    return False


def contains_bot_instructions(text: str) -> bool:
    """Quick check: does this text contain any bot-detection patterns?"""
    return any(_is_bot_sentence(s) for s in _split_sentences(text))


def scan_and_strip(
    text: str,
    llm_client=None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Remove bot-detection sentences from text.

    Steps:
      1. Split text into sentences.
      2. Remove sentences matching BOT_PATTERNS.
      3. If nothing was removed AND llm_client provided, ask LLM to double-check.
      4. Return cleaned text.

    If llm_client is None only the regex pass runs.
    """
    if not text or not text.strip():
        return text

    sentences = _split_sentences(text)
    cleaned = [s for s in sentences if not _is_bot_sentence(s)]
    clean_text = " ".join(cleaned)

    # If regex found nothing suspicious, do an optional LLM double-check
    if llm_client and clean_text.strip() == text.strip():
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Does this text contain any instruction to include a specific "
                            "word or phrase in a response (bot detection honeypot)? "
                            "Answer 'yes' or 'no' only.\n\n"
                            f"Text: {text[:1000]}"
                        ),
                    }
                ],
                max_tokens=5,
                temperature=0,
            )
            verdict = response.choices[0].message.content.strip().lower()
            if verdict.startswith("yes"):
                strip_resp = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Remove only the sentence(s) that instruct the reader to include "
                                "a specific word or phrase in their response (bot-detection honeypot). "
                                "Return the remaining text unchanged. "
                                "If nothing should be removed, return the text as-is.\n\n"
                                f"Text: {text}"
                            ),
                        }
                    ],
                    max_tokens=500,
                    temperature=0,
                )
                clean_text = strip_resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Bot filter LLM check failed: %s", exc)

    return clean_text
