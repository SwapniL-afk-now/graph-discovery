"""GVDAgent — DVD's ReAct function-calling loop over a Video Knowledge Graph.

The loop is DVDCoreAgent's, unchanged in spirit: THINK → ACT → OBSERVE with
OpenAI function calling, `finish` raising StopException, and a forced finish
on the last iteration. What changes is the tool belt: DVD's three tools
(graph-backed re-implementations) plus the eight VKG tools, all returning
observations with affordance footers.

NOTE: no `from __future__ import annotations` — `finish` must keep a real
Annotated annotation for DVD's schema generator.
"""

import copy
import json
import os
import re
from typing import Annotated as A
from typing import Optional

# A counting question that counts THINGS present ("how many people/sticks") is a
# spatial count → the object detector. One that counts EVENTS over time ("how
# many times she knocks") is temporal → dense frames, NOT the detector.
_OBJ_COUNT_RE = re.compile(r"\bhow many\b|\bnumber of\b", re.I)
_EVENT_COUNT_RE = re.compile(r"how many times|how often|how many .*\btimes\b", re.I)
_TIMEREF_RE = re.compile(r"\[Time reference:\s*([0-9:]+)\s*[-–]\s*([0-9:]+)\]")
_LETTER_RE = re.compile(r"\b([A-D])\b")

from .dvd_compat import DVDCompatTools
from .func_schema import as_json_schema, doc as D
from .llm import LLMClient
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE
from .vkg_tools import VKGToolkit


class StopException(Exception):
    """Raised by `finish` to end the run with the final answer."""


def finish(answer: A[str, D("Answer to the user's question.")]) -> None:
    """Call this function after confirming the answer of the user's question, and finish the conversation."""
    raise StopException(answer)


def _load_graph(graph_path: str):
    from qvkg.schema import VKGraph
    return VKGraph.load(graph_path)


def _load_faiss(index_path: Optional[str]):
    if not index_path or not os.path.exists(index_path):
        return None
    try:
        from qvkg.faiss_index import load_faiss_index
        return load_faiss_index(index_path)
    except Exception as e:
        print(f"[gvd] FAISS index unavailable ({e}) — falling back to lexical search.")
        return None


class GVDAgent:
    def __init__(
        self,
        graph_path: str,
        video_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        text_encoder=None,                 # object with .encode_text([str]) -> np.ndarray
        llm: Optional[LLMClient] = None,
        max_iterations: int = 12,
        enable_dvd_tools: bool = True,
        enable_detector: bool = True,
    ):
        self.llm = llm or LLMClient()
        self.max_iterations = max_iterations

        graph = _load_graph(graph_path)
        self.graph = graph
        self.toolkit = VKGToolkit(
            graph,
            faiss_index=_load_faiss(faiss_index_path),
            text_encoder=text_encoder,
            llm_complete=self.llm.complete,
            inferred_edges_path=graph_path + ".inferred_edges.json",
        )

        self.tools = list(self.toolkit.tools())
        if enable_dvd_tools:
            self.dvd_tools = DVDCompatTools(self.toolkit, self.llm, video_path,
                                            enable_detector=enable_detector)
            self.tools += self.dvd_tools.tools()
        self.tools.append(finish)

        self.name_to_function_map = {t.__name__: t for t in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(t), "type": "function"}
            for t in self.tools
        ]

        # Whole-video sentinel nodes (e.g. the Narrator CharacterNode spans
        # 0–1e9) must not define the duration the agent plans windows with.
        video_length = int(max((n.t_end for n in graph.nodes.values()
                                if n.t_end < 1e6), default=0))
        present_types = ", ".join(sorted(graph.type_idx.keys())) or "(none)"
        self.messages = [
            {"role": "system",
             "content": SYSTEM_PROMPT.format(node_types=present_types)},
            {"role": "user", "content": USER_TEMPLATE.format(
                video_length=video_length, question="QUESTION_PLACEHOLDER")},
        ]

    # ------------------------------------------------------------------ #
    # Tool execution (DVD-style)
    # ------------------------------------------------------------------ #

    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        msgs.append({"tool_call_id": tool_call_id, "role": "tool",
                     "name": name, "content": str(content)})

    def _exec_tool(self, tool_call, msgs):
        name = tool_call["function"]["name"]
        if name not in self.name_to_function_map:
            self._append_tool_msg(
                tool_call["id"], name,
                f"Invalid function name: {name!r}. Available: "
                f"{', '.join(self.name_to_function_map)}", msgs)
            return
        try:
            args = json.loads(tool_call["function"]["arguments"] or "{}")
        except json.JSONDecodeError as exc:
            self._append_tool_msg(
                tool_call["id"], name,
                f"Error decoding arguments: {exc!s}. Re-emit the call with valid JSON.",
                msgs)
            return
        try:
            print(f"[gvd] {name}({args})")
            result = self.name_to_function_map[name](**args)
            self._append_tool_msg(tool_call["id"], name, result, msgs)
        except StopException:
            raise
        except Exception as exc:
            # Tool errors go back to the model as observations so it can adapt.
            self._append_tool_msg(
                tool_call["id"], name,
                f"Tool error in {name}: {exc!s}. Adjust the arguments or try a "
                "different tool.", msgs)

    # ------------------------------------------------------------------ #
    # Pre-routing & verification helpers
    # ------------------------------------------------------------------ #

    def _maybe_precount(self, question: str, msgs) -> None:
        """Deterministically run the object detector for spatial-count questions.

        The 4B model won't select count_objects on its own (0 calls in
        practice), so for "how many <things>" questions with a known time
        reference we run the detector here and inject the grounded count as
        evidence. Event counts ("how many times…") are temporal and stay with
        dense frame inspection instead."""
        dvd = getattr(self, "dvd_tools", None)
        if not (dvd and getattr(dvd, "_enable_detector", False) and dvd.video_path):
            return False
        if not _OBJ_COUNT_RE.search(question) or _EVENT_COUNT_RE.search(question):
            return False
        m = _TIMEREF_RE.search(question)
        if not m:
            return False
        try:
            phrase = self.llm.complete([
                {"role": "system", "content":
                 "Extract the single noun phrase being counted in this question. "
                 "Reply with ONLY the phrase, lowercase, no article, no quotes. "
                 "E.g. 'people carrying the sedan chair', or 'sticks'."},
                {"role": "user", "content": question}], max_tokens=30)
            phrase = phrase.strip().strip('".').splitlines()[0][:80]
            if not phrase:
                return
            result = dvd.count_objects([m.group(1), m.group(2)], phrase)
        except Exception:
            return False
        msgs.append({"role": "user", "content":
                     "AUTOMATED OBJECT-DETECTOR RESULT (a dedicated detector was run "
                     "for this counting question — treat it as strong evidence; only "
                     "override it if the frames clearly contradict it):\n" + result})
        return True

    def _ensure_letter(self, answer: str, last_thought: str) -> str:
        """Guarantee the returned answer carries a parseable A–D letter, so a
        finish that emitted prose (the uid-68 ERROR class) still resolves."""
        for src in (answer, last_thought):
            if src and _LETTER_RE.search(re.sub(r"<think>.*?</think>", "", src,
                                               flags=re.DOTALL)):
                return src
        # Last resort: coerce a letter out of the reasoning gathered so far.
        try:
            coerced = self.llm.complete([
                {"role": "system", "content":
                 "Read the reasoning and output ONLY the final answer as a single "
                 "letter: A, B, C, or D."},
                {"role": "user", "content":
                 f"Reasoning:\n{last_thought or answer}\n\nFinal letter:"}],
                max_tokens=4)
            if _LETTER_RE.search(coerced or ""):
                return coerced
        except Exception:
            pass
        return answer

    _TIME_REF_RE = re.compile(
        r"\[Time reference:\s*([\d:]+)\s*-\s*([\d:]+)\]")

    def _prefetch_graph_context(self, question: str, msgs) -> bool:
        """Run the graph tools up front and inject their output as context.

        Sections: overview (hierarchy + characters), semantic search on the
        question, and — when the question carries a [Time reference] — a close
        read of that window (±30s). Each section is capped so the seed stays
        a fraction of the context budget."""
        CAP = 4000

        def grab(label, fn, *args, **kw):
            try:
                text = str(fn(*args, **kw))
            except Exception as exc:
                return None
            if len(text) > CAP:
                text = text[:CAP] + "\n… (truncated — re-run the tool on a narrower target for the rest)"
            return f"### {label}\n{text}"

        sections = [grab("get_overview()", self.toolkit.get_overview)]

        m = self._TIME_REF_RE.search(question)
        if m:
            from .timeutil import to_seconds, fmt
            t0 = max(0, to_seconds(m.group(1)) - 30)
            t1 = to_seconds(m.group(2)) + 30
            sections.append(grab(
                f"read_moment({fmt(t0)}, {fmt(t1)}) — the referenced window ±30s",
                self.toolkit.read_moment, fmt(t0), fmt(t1)))
            # read_moment's cap is often consumed by scene/clip structure; the
            # dialogue is the answer for most why/who questions, so guarantee
            # it survives with its own section (wider window: speech that
            # explains a moment often comes a little before or after it).
            sections.append(grab(
                f"query_nodes(SpeechNode, {fmt(max(0, t0 - 60))}, {fmt(t1 + 60)}) — all dialogue around the window",
                self.toolkit.query_nodes, "SpeechNode",
                fmt(max(0, t0 - 60)), fmt(t1 + 60)))

        plain_q = self._TIME_REF_RE.sub("", question).split("\n\n")[0].strip()
        sections.append(grab(
            "search_events(question)", self.toolkit.search_events, plain_q))

        sections = [s for s in sections if s]
        if not sections:
            return False
        msgs.append({
            "role": "user",
            "content": "PRE-FETCHED GRAPH CONTEXT (gathered for you with the graph "
                       "tools — treat it as observations you already made):\n\n"
                       + "\n\n".join(sections)
                       + "\n\nUse this context first. Call further tools only for "
                         "what it does not answer (e.g. inspect_frames to verify "
                         "visually), then finish with the letter."})
        return True

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self, question: str):
        """ReAct loop. Returns (answer, full message transcript)."""
        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"] = msgs[-1]["content"].replace(
            "QUESTION_PLACEHOLDER", question)
        # Per-question state: the search-history guard must not bleed across
        # questions that reuse this agent/toolkit.
        self.toolkit._search_history = []
        if getattr(self, "dvd_tools", None):
            self.dvd_tools.current_mcq = question

        # Fix 1: deterministically ground spatial-count questions in the detector.
        precounted = self._maybe_precount(question, msgs)

        # Deterministic graph prefetch: a small model rarely explores the graph
        # on its own (it routes to ONE tool and finishes), so run the graph
        # tools for it and seed the transcript with their output. The loop then
        # spends its turns on frame verification, not retrieval.
        prefetched = self._prefetch_graph_context(question, msgs)

        answer = None
        plain_text_nudges = 0
        inspected = precounted         # has the agent LOOKED at frames yet?
        graph_grounded = prefetched    # the prefetch IS a graph consult
        nudged_sources = set()         # which missing-evidence nudges we've sent
        has_video = bool(getattr(getattr(self, "dvd_tools", None), "video_path", None))
        _GRAPH_TOOLS = {"get_overview", "search_events", "query_nodes", "follow_connections",
                        "trace_causes", "find_entity", "read_moment", "explain_why"}
        last_thought = ""
        for i in range(self.max_iterations):
            if i == self.max_iterations - 1:
                msgs.append({"role": "user",
                             "content": "Please call the `finish` function now with your "
                             "final answer (exactly one letter: A, B, C, or D)."})

            response = self.llm.chat_with_tools(msgs, self.function_schemas)
            msgs.append(response)
            if response.get("content"):
                last_thought = response["content"]  # the model's running reasoning

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                content = response.get("content", "") or ""
                # Don't silently accept plain text as the answer — the loop's
                # contract is to terminate via `finish`, whose answer is parsed
                # for the MCQ letter. Nudge once (unless we're out of turns).
                if plain_text_nudges < 3 and i < self.max_iterations - 1:
                    plain_text_nudges += 1
                    msgs.append({"role": "user",
                                 "content": "Do not answer in plain text. Call the `finish` "
                                 "function with your answer as exactly one letter: A, B, C, or D."})
                    continue
                answer = content
                break
            try:
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    # Two-source evidence floor: the dominant error mode was
                    # committing on a SINGLE source (one frame read, or one graph
                    # read). Require BOTH a visual look and a graph consult before
                    # finishing, so every answer is cross-checked. Each nudge is
                    # sent at most once, so this can never loop.
                    if name == "finish" and has_video and i < self.max_iterations - 1:
                        missing = None
                        if not inspected and "look" not in nudged_sources:
                            missing = ("look",
                                       "you have not LOOKED at the video yet. Inspect the "
                                       "relevant moment with inspect_frames (or "
                                       "count_objects for counting) before finishing.")
                        elif not graph_grounded and "graph" not in nudged_sources:
                            missing = ("graph",
                                       "you have not cross-checked against the graph. "
                                       "Read the moment's dialogue/events with read_moment "
                                       "(or query_nodes on SpeechNode/OCRNode) before finishing — "
                                       "confirm the visual answer agrees with what is said/shown.")
                        if missing:
                            nudged_sources.add(missing[0])
                            self._append_tool_msg(
                                tc["id"], name, "Do not finish yet — " + missing[1], msgs)
                            continue
                    if name in ("inspect_frames", "count_objects"):
                        inspected = True
                    elif name in _GRAPH_TOOLS:
                        graph_grounded = True
                    self._exec_tool(tc, msgs)
            except StopException as exc:
                answer = str(exc)
                break

        # Ran out of iterations without `finish` (e.g. kept inspecting frames).
        # Fall back to the model's last reasoning rather than returning nothing,
        # so the MCQ letter can still be parsed instead of yielding an ERROR.
        if not answer:
            answer = last_thought
        # Fix 3: guarantee a parseable letter (kills the prose-finish ERROR class).
        answer = self._ensure_letter(answer or "", last_thought)
        return answer, msgs
