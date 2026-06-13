"""Deterministic question router for gvd_lite.

The router classifies the question into one of a small set of fixed
"routes", each of which is a 2-3 step tool sequence with question-derived
arguments. No LLM call is made here — the regexes are the same ones the
old GVD planner used, but applied as a Python decision rather than an
LLM-composed plan.

Each route returns a list of dicts:
    {"tool": "read_graph" | "look_at_frames", "args": {...}}

The agent executes them in order, accumulates observations, and then
makes a single one-shot decision call to the 4B model.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---- Regexes (mirroring gvd/agent.py heuristics) ----------------------- #

# [Time reference: HH:MM:SS - HH:MM:SS]  (note: en-dash or hyphen, and the
# vkg code allows both forms; we accept both).
_TIME_REF_RE = re.compile(
    r"\[Time reference:\s*([0-9:]+)\s*[-–]\s*([0-9:]+)\]"
)

_WHY_RE = re.compile(r"\bwhy\b|\breason\b|\bpurpose\b|\bcause[sd]?\b|"
                     r"\bmotivation\b|\bintent\b", re.I)
_TEMPORAL_RE = re.compile(r"\bbefore\b|\bafter\b|\bnext\b|\border\b|"
                          r"\bfirst\b|\bfollowing\b|\bprior\b|"
                          r"\bprevious\b|\bled up to\b", re.I)
_COUNT_OBJ_RE = re.compile(r"\bhow many\b|\bnumber of\b", re.I)
_COUNT_EVT_RE = re.compile(r"how many times|how often|how many .*\btimes\b", re.I)

# "What does X do" / "What happens when" / "What is X doing" — visual actions.
_WHAT_DOES_RE = re.compile(
    r"what does .* do|what happens when|what is .* doing|"
    r"describe what .* do|describe what happens",
    re.I,
)

# Visual detail / color / appearance / clothing
_VISUAL_DETAIL_RE = re.compile(
    r"\bwhat color\b|\bwhat does .* look like\b|\bwhat is .* wearing\b|"
    r"\bwhat object\b|\bwhat is on the\b",
    re.I,
)

# Speech / dialogue
_SPEECH_RE = re.compile(
    r"\bwhat does .* say\b|\bwhat is said\b|\bwhat does the dialogue\b|"
    r"\bwhat is spoken\b|\bquote\b",
    re.I,
)

# Summarization (rare in 4B-friendly set; falls into general but gets a wider window)
_SUMMARY_RE = re.compile(
    r"\bbest summarizes\b|\bmain content\b|\bsummarize\b|\boverview of\b",
    re.I,
)


# ---- Helpers ----------------------------------------------------------- #

def _parse_time_ref(question: str) -> Optional[Tuple[str, str]]:
    """Return (t0_str, t1_str) from a [Time reference: ...] tag, else None."""
    m = _TIME_REF_RE.search(question)
    if not m:
        return None
    return m.group(1), m.group(2)


def _pad(t0: str, t1: str, seconds: int) -> Tuple[str, str]:
    """Return (t0 - seconds, t1 + seconds), clamped at 0, formatted HH:MM:SS."""
    from gvd.timeutil import fmt, to_seconds
    s0 = max(0, to_seconds(t0) - seconds)
    s1 = to_seconds(t1) + seconds
    return fmt(s0), fmt(s1)


def _strip_time_ref(question: str) -> str:
    return _TIME_REF_RE.sub("", question).strip()


def _opt_text(q: str) -> str:
    """Strip the (A) (B) (C) (D) option lines so the router doesn't
    pattern-match on option text."""
    return re.sub(r"^\([A-D]\)\s*", "", q, flags=re.M)


# ---- Route functions ---------------------------------------------------- #
# Each takes (question, time_ref_tuple_or_None) and returns a list of steps.

def _route_count_events(question: str, ref) -> List[Dict]:
    """'How many times does X knock / hit / strike…' — dense frames, no graph."""
    t0, t1 = ref
    plain = _strip_time_ref(question)
    # Extract a short phrase for the question.
    target = plain.split("\n")[0][:200]
    return [
        {
            "tool": "look_at_frames",
            "args": {
                "time_ranges": [[t0, t1]],
                "question": ("Count every distinct occurrence. "
                             + target),
                "sampling": "dense",
            },
        },
    ]


def _route_count_objects(question: str, ref) -> List[Dict]:
    """'How many people / sticks / chairs are present' — graph entity count
    + detail frames for cross-check."""
    t0, t1 = ref
    plain = _strip_time_ref(question).split("\n")[0]
    return [
        {
            "tool": "read_graph",
            "args": {
                "query": plain,
                "time_start": t0,
                "time_end": t1,
                "focus": "all",
            },
        },
        {
            "tool": "look_at_frames",
            "args": {
                "time_ranges": [[t0, t1]],
                "question": ("List each instance of the target. Count them. "
                             + plain),
                "sampling": "detail",
            },
        },
    ]


def _route_why(question: str, ref) -> List[Dict]:
    """WHY / REASON — causal walk (graph) + visual verification (frames)."""
    t0, t1 = ref
    pt0, pt1 = _pad(t0, t1, 30)
    plain = _strip_time_ref(question).split("\n")[0]
    return [
        {
            "tool": "read_graph",
            "args": {
                # Worded as a 'why' query so the router inside read_graph
                # picks why_did_this_happen (which traces causes+effects+dialogue).
                "query": "why " + plain,
                "time_start": pt0,
                "time_end": pt1,
                "focus": "dialogue",
            },
        },
        {
            "tool": "look_at_frames",
            "args": {
                "time_ranges": [[t0, t1]],
                "question": ("What is happening visually? What are the people doing? "
                             "Quote any on-screen text. " + plain),
                "sampling": "event",
            },
        },
    ]


def _route_temporal(question: str, ref) -> List[Dict]:
    """BEFORE / AFTER / NEXT / ORDER — timeline walk + a check on the window itself."""
    t0, t1 = ref
    plain = _strip_time_ref(question).split("\n")[0]
    return [
        {
            "tool": "read_graph",
            "args": {
                "query": "what happens before and after " + plain,
                "time_start": t0,
                "time_end": t1,
                "focus": "all",
            },
        },
        {
            "tool": "look_at_frames",
            "args": {
                "time_ranges": [[t0, t1]],
                "question": plain,
                "sampling": "event",
            },
        },
    ]


def _route_what_does(question: str, ref) -> List[Dict]:
    """'What does X do' — visual action, with a graph read for context."""
    t0, t1 = ref
    plain = _strip_time_ref(question).split("\n")[0]
    return [
        {
            "tool": "read_graph",
            "args": {
                "query": plain,
                "time_start": t0,
                "time_end": t1,
                "focus": "actions",
            },
        },
        {
            "tool": "look_at_frames",
            "args": {
                "time_ranges": [[t0, t1]],
                # Encourage per-character breakdown (helps "what does X do" vs others).
                "question": ("Describe EACH person's action: what they do, what they "
                             "touch, what they say. " + plain),
                "sampling": "detail",
            },
        },
    ]


def _route_visual_detail(question: str, ref) -> List[Dict]:
    """Color / appearance / what someone is wearing / on-screen text — frames only."""
    t0, t1 = ref
    plain = _strip_time_ref(question).split("\n")[0]
    return [
        {
            "tool": "read_graph",
            "args": {
                "query": plain,
                "time_start": t0,
                "time_end": t1,
                "focus": "text",
            },
        },
        {
            "tool": "look_at_frames",
            "args": {
                "time_ranges": [[t0, t1]],
                "question": plain,
                "sampling": "detail",
            },
        },
    ]


def _route_speech(question: str, ref) -> List[Dict]:
    """Dialogue / what someone said — graph read, focus on dialogue."""
    t0, t1 = ref
    pt0, pt1 = _pad(t0, t1, 30)
    plain = _strip_time_ref(question).split("\n")[0]
    return [
        {
            "tool": "read_graph",
            "args": {
                "query": plain,
                "time_start": pt0,
                "time_end": pt1,
                "focus": "dialogue",
            },
        },
    ]


def _route_general(question: str, ref) -> List[Dict]:
    """Catch-all: a graph read of the window, then a visual look for confirmation."""
    if ref is None:
        # No time reference → fall back to a free-form graph search.
        plain = _strip_time_ref(question).split("\n")[0]
        return [
            {"tool": "read_graph", "args": {"query": plain}},
        ]
    t0, t1 = ref
    pt0, pt1 = _pad(t0, t1, 20)
    plain = _strip_time_ref(question).split("\n")[0]
    return [
        {
            "tool": "read_graph",
            "args": {
                "query": plain,
                "time_start": pt0,
                "time_end": pt1,
                "focus": "all",
            },
        },
        {
            "tool": "look_at_frames",
            "args": {
                "time_ranges": [[t0, t1]],
                "question": plain,
                "sampling": "auto",
            },
        },
    ]


# ---- Public entry point ------------------------------------------------ #

def route(question: str) -> List[Dict]:
    """Classify the question and return a fixed tool sequence.

    The returned list is a Python list of dicts:
        {"tool": "read_graph" | "look_at_frames", "args": {...}}
    The agent executes them in order before asking the model to decide.
    """
    q = _opt_text(question)              # strip (A)/(B)/(C)/(D) lines
    ref = _parse_time_ref(question)

    # No time reference → semantic search; the 4B can call read_graph itself
    # in the ReAct fallback if it needs more.
    if ref is None:
        plain = _strip_time_ref(q).split("\n")[0]
        return [{"tool": "read_graph", "args": {"query": plain}}]

    # Order matters: most specific pattern first.
    if _COUNT_EVT_RE.search(q):
        return _route_count_events(q, ref)
    if _COUNT_OBJ_RE.search(q):
        return _route_count_objects(q, ref)
    if _WHY_RE.search(q):
        return _route_why(q, ref)
    if _TEMPORAL_RE.search(q):
        return _route_temporal(q, ref)
    if _WHAT_DOES_RE.search(q):
        return _route_what_does(q, ref)
    if _VISUAL_DETAIL_RE.search(q):
        return _route_visual_detail(q, ref)
    if _SPEECH_RE.search(q):
        return _route_speech(q, ref)
    if _SUMMARY_RE.search(q):
        return _route_general(q, ref)
    return _route_general(q, ref)


def route_name(question: str) -> str:
    """For logging / debugging: which route did the classifier pick?"""
    q = _opt_text(question)
    ref = _parse_time_ref(question)
    if ref is None:
        return "no_time_ref"
    if _COUNT_EVT_RE.search(q): return "count_events"
    if _COUNT_OBJ_RE.search(q): return "count_objects"
    if _WHY_RE.search(q):       return "why"
    if _TEMPORAL_RE.search(q):  return "temporal"
    if _WHAT_DOES_RE.search(q): return "what_does"
    if _VISUAL_DETAIL_RE.search(q): return "visual_detail"
    if _SPEECH_RE.search(q):    return "speech"
    if _SUMMARY_RE.search(q):   return "summary"
    return "general"
