"""Utility helpers for normalizing agent generated text."""

from __future__ import annotations

import re


def strip_tool_calls(text: str) -> str:
    """移除文本中的工具调用标记。"""

    if not text:
        return text

    pattern = re.compile(r"\[TOOL_CALL:[^\]]+\]")
    return pattern.sub("", text)


# ── Repetition-degeneration trim ─────────────────────────────────────────────
# Small local models (gemma3:4b, qwen3.5:9b, …) occasionally fall into a
# repetition loop and emit the same emoji or the same line until they hit the
# token budget — e.g. "💡💡💡…" or the same row of a table over and over. The
# raw output is useless and confusing on the report screen. We scan for those
# two failure modes and cut the text at the first sign of one.

# Single-character (incl. emoji code point) repeated >= this many times in a
# row is treated as degeneration. 12 is safe — legitimate separator lines like
# "------------" or "============" run shorter than that in our corpus.
_MAX_CHAR_RUN = 12

# Same non-empty stripped line repeated >= this many times in a row is treated
# as degeneration. 4 covers list bullets and table cells while still catching
# real loops.
_MAX_LINE_RUN = 4

_CHAR_RUN_RE = re.compile(r"(\S)\1{" + str(_MAX_CHAR_RUN - 1) + r",}")
_TRUNCATION_NOTE = "\n\n*（模型输出在此处出现异常重复，已自动截断）*"


def collapse_repetition(text: str) -> str:
    """Trim degenerative repetition loops out of an LLM response.

    Looks for two patterns:
    1. Same code point repeated ``_MAX_CHAR_RUN``+ times (catches 💡 spam,
       runs of ``。。。``, ``???`` etc.)
    2. Same non-empty line repeated ``_MAX_LINE_RUN``+ times in a row
       (catches table-row loops where each line is identical).

    The text is cut at the FIRST detected pattern and a small italic note is
    appended so the user knows we trimmed it. Whitespace-only repetition is
    ignored.
    """
    if not text:
        return text

    # ── (1) same-character runs ────────────────────────────────────────────
    match = _CHAR_RUN_RE.search(text)
    char_cut = match.start() if match else None

    # ── (2) same-line runs ─────────────────────────────────────────────────
    lines = text.split("\n")
    line_cut: int | None = None  # character offset in original text
    streak_text: str | None = None
    streak_count = 0
    char_offset = 0
    streak_start_offset = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped == streak_text:
            streak_count += 1
            if streak_count >= _MAX_LINE_RUN:
                line_cut = streak_start_offset
                break
        else:
            streak_text = stripped if stripped else None
            streak_count = 1
            streak_start_offset = char_offset
        # advance offset past this line + the "\n" separator (except last)
        char_offset += len(line) + (1 if i < len(lines) - 1 else 0)

    # Pick the earliest cut
    candidates = [c for c in (char_cut, line_cut) if c is not None]
    if not candidates:
        return text
    cut = min(candidates)
    return text[:cut].rstrip() + _TRUNCATION_NOTE

