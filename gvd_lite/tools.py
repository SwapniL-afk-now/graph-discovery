"""Three unified tools for gvd_lite.

  read_graph(query, time_start?, time_end?, focus?)
    Python routes this to the right VKG query based on the query text
    and the time window. The 4B model only ever sees one tool name.

  look_at_frames(time_ranges, question)
    VLM over actual video frames. Wraps DVDCompatTools.inspect_frames
    with the same signature the old GVD exposed.

  answer(letter)
    Terminate the run with exactly A/B/C/D.

The underlying work (graph queries, frame extraction, VLM calls) is
delegated to the existing gvd.vkg_tools and gvd.dvd_compat modules.
"""

import re
from typing import Annotated as A, List, Optional, Tuple

from gvd.func_schema import as_json_schema, doc as D
from gvd.timeutil import fmt, to_seconds


# Keyword groups used to route read_graph → the right VKG query.
_WHY_KW = re.compile(
    r"\bwhy\b|\breason\b|\bpurpose\b|\bcause[sd]?\b|\bmotivation\b|\bintent\b",
    re.I,
)
_TEMP_KW = re.compile(
    r"\bbefore\b|\bafter\b|\bnext\b|\border\b|\bfirst\b|"
    r"\bfollowing\b|\bprior\b|\bprevious\b|\bled up to\b|\bthen\b",
    re.I,
)
# A short list of well-known character/role words — when the query contains
# one of these, prefer entity-timeline mode (today's find_entity).
_ENTITY_WORDS = re.compile(
    r"\bthe (protagonist|girl|boy|man|woman|child|merchant|guard|prince|princess|"
    r"king|queen|soldier|thief|hero|villain|monster|robot|vlogger|stranger|"
    r"detective|narrator|captain|doctor|servant|witch|wizard|dokkaebi|vampire|"
    r"giant|companion|creature|character|person|kid|teen|adult)\b",
    re.I,
)


class _ToolContext:
    """Bound state passed to every tool call. Created once per agent."""

    def __init__(self, toolkit, dvd):
        self.toolkit = toolkit
        self.dvd = dvd


# A module-level slot the agent fills in __init__ / per-question.
_CTX: Optional[_ToolContext] = None


def bind(ctx: _ToolContext) -> None:
    """Bind the toolkit + dvd-tools for the current agent. Called by GVDLiteAgent.run()."""
    global _CTX
    _CTX = ctx


def _require_ctx() -> _ToolContext:
    if _CTX is None:
        raise RuntimeError("gvd_lite.tools: bind(ctx) must be called before any tool runs")
    return _CTX


# ----------------------------------------------------------------------- #
# Tool 1: read_graph
# ----------------------------------------------------------------------- #

def read_graph(
    query: A[str, D("Short natural-language question or topic — e.g. 'why the woman was turned away', 'what the guard says', 'the explosion scene', 'the protagonist'. For WHY/REASON questions, ask about the cause. For BEFORE/AFTER questions, ask about the timeline. Otherwise just describe what you want to know.")],
    time_start: A[str, D("Optional window start 'HH:MM:SS' (or seconds). Use the [Time reference] from the question when present, with ±30s padding for why/after questions.")] = "",
    time_end: A[str, D("Optional window end 'HH:MM:SS' (or seconds). Pair with time_start to scope the query to one moment.")] = "",
    focus: A[str, D("Only used when no time window is given: 'all' (default) or 'dialogue' (only spoken lines). Most questions should leave this as 'all'.")] = "all",
) -> str:
    """
    One unified graph-query tool. The router picks the right VKG operation
    based on the query text and the time window:

      • Query mentions why/reason/cause/purpose + a time window  → causal walk
        (today's why_did_this_happen: traces causes + effects + dialogue)
      • Query mentions before/after/next/order/first + a time window → temporal
        walk (today's before_and_after: events before, during, after)
      • A known person/character in the query                     → entity timeline
        (today's find_entity: every appearance in chronological order)
      • Time window set (no why/temporal keyword)                → window read
        (today's read_moment: scenes, actions, speech, OCR in the window)
      • Otherwise                                                → semantic search
        (today's find: event / speech / text lookup, or entity match)

    Returns a plain-text observation the model can read directly.
    """
    ctx = _require_ctx()
    q = (query or "").strip()
    ts = (time_start or "").strip()
    te = (time_end or "").strip()
    has_window = bool(ts) and bool(te)

    try:
        t0 = to_seconds(ts) if ts else 0.0
        t1 = to_seconds(te) if te else 0.0
    except Exception as exc:
        return f"Bad time format ({exc}). Use 'HH:MM:SS' or seconds."

    if t1 <= t0 and has_window:
        return f"time_end ({fmt(t1)}) must be after time_start ({fmt(t0)})."

    # Route 1: why / reason
    if _WHY_KW.search(q) and has_window:
        return ctx.toolkit.why_did_this_happen(ts, te)

    # Route 2: temporal walk (before/after/first/...)
    if _TEMP_KW.search(q) and has_window:
        mid = fmt((t0 + t1) / 2.0)
        win = int(max(60, min(180, (t1 - t0) * 2)))
        return ctx.toolkit.before_and_after(mid, window=win)

    # Route 3: entity timeline (a person's name in the query)
    if _ENTITY_WORDS.search(q) and not has_window:
        # Pull out the noun phrase the user named.
        m = _ENTITY_WORDS.search(q)
        if m:
            return ctx.toolkit.find(m.group(1))

    # Route 4: window read
    if has_window:
        return ctx.toolkit.read_moment(ts, te, focus=focus or "all", query=q)

    # Route 5: semantic search / entity match
    return ctx.toolkit.find(q)


# ----------------------------------------------------------------------- #
# Tool 2: look_at_frames
# ----------------------------------------------------------------------- #

def look_at_frames(
    time_ranges: A[List[List[str]], D("List of [start, end] time ranges as 'HH:MM:SS' pairs, e.g. [[\"00:14:10\", \"00:14:25\"]]. One range is the most common case.")],
    question: A[str, D("The SPECIFIC visual question to answer from the frames — e.g. 'What color is the woman's dress?', 'Count each knock on the door.', 'What does the guard do after the woman speaks?'. Generic 'what happens' misses fine details — be specific.")],
    sampling: A[str, D("'auto' (recommended default), 'dense' (use for COUNTING repeated events: knocks, hits, knocks — and for fast actions; window should be under 60s), 'event' (use for 'what happens during' over a longer span), or 'detail' (one moment: color, pose, on-screen text, what someone is wearing).")] = "auto",
) -> str:
    """
    Look at the actual video frames in the given time ranges and have the
    vision model answer your question. Use this for anything VISUAL:
    what someone does, what they wear, what color something is, what text
    is on screen, counting how many TIMES an event happens. Returns a
    plain-text observation.

    The graph tells you what was SAID — frames tell you what was DONE.
    Use both, especially for WHY questions.
    """
    ctx = _require_ctx()
    if not time_ranges:
        return ("look_at_frames needs at least one [start, end] range, e.g. "
                "[[\"00:14:10\", \"00:14:25\"]].")
    if not ctx.dvd.video_path:
        return ("No raw video is configured for this session — frames cannot be "
                "inspected. Rely on the graph (read_graph) for this question.")
    # The current_mcq is used by inspect_frames to fold the full MCQ into the
    # VLM prompt, which materially helps the VLM map its description onto
    # the options. We set it in the agent right before the model calls.
    return ctx.dvd.inspect_frames(time_ranges, question, sampling=sampling)


# ----------------------------------------------------------------------- #
# Tool 3: count_objects
# ----------------------------------------------------------------------- #

def count_objects(
    time_range: A[List[str], D("A single [start, end] time range as 'HH:MM:SS' strings, e.g. [\"00:24:06\", \"00:24:40\"]. Keep it tight around the moment to count.")],
    object_phrase: A[str, D("What to count, as a short noun phrase the detector can ground, e.g. 'person', 'knife hanging on the wall', 'smoke machine'. Be specific.")],
) -> str:
    """
    COUNT how many people or objects are PRESENT in a moment, using a real
    object detector (bounding boxes) — far more reliable than counting by eye
    in frames. USE THIS for every 'how many X' question about things visible
    at one time. For counting repeated EVENTS over time (how many times she
    knocked), use look_at_frames with sampling='dense' instead.
    """
    ctx = _require_ctx()
    return ctx.dvd.count_objects(time_range, object_phrase)


# ----------------------------------------------------------------------- #
# Tool 4: answer
# ----------------------------------------------------------------------- #

class StopException(Exception):
    """Raised by answer() to end the run with the final letter."""


def answer(
    letter: A[str, D("The final answer: exactly one letter, one of A, B, C, or D.")],
) -> None:
    """
    Call this LAST with exactly one letter: A, B, C, or D. This is the ONLY
    way to finish a question. Calling it with anything else (or calling it
    before you have evidence for all four options) will end the run without
    a valid answer.
    """
    s = (letter or "").strip().strip(".").strip(")").strip("(")
    s = s.split()[0] if s else ""
    s = s.upper()
    if s not in "ABCD":
        raise ValueError(
            f"answer() must be one of A/B/C/D, got {letter!r}. "
            "Pick the option best supported by the evidence."
        )
    raise StopException(s)


# Public list of tools (in the order the prompt advertises them)
TOOLS = [read_graph, look_at_frames, count_objects, answer]
TOOL_SCHEMAS = [{"function": as_json_schema(t), "type": "function"} for t in TOOLS]
TOOL_NAMES = [t.__name__ for t in TOOLS]
