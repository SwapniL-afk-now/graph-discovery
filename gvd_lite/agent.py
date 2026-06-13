"""GVDLiteAgent — a flexible ReAct loop for 4B models on video MCQs.

Pipeline (per question):

  Stage 1 (deterministic, no LLM):
    router.route(question) → fixed 1-2 step tool sequence
    Execute each step, accumulate "hints".

  Stage 2 (LLM, flexible ReAct):
    The model sees the hints and then takes its own tool calls. It may:
      - call any combination of read_graph / look_at_frames in a turn,
      - call the same tool many times with DIFFERENT arguments,
      - call answer(letter) when ready.
    The transcript records the full ReAct trajectory.

  Stage 3 (last resort):
    If the model exhausts its budget without calling answer, _ensure_letter
    scrapes a letter from its last reasoning.

Dedup: per-question, every (tool, normalized_args) tuple is cached. A
repeat call returns the previous result with a "you already asked this"
notice, so the model wastes no tool budget on duplicates.

Per-tool limits: a single tool can be called at most 6 times per question
(plenty for multi-window checks, well below loop-spam territory).
Total tool-call budget per question: 12 (cap in case the model gets
stuck in a loop).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from gvd.func_schema import as_json_schema
from gvd.timeutil import fmt, to_seconds
from gvd.vkg_tools import VKGToolkit
from gvd.dvd_compat import DVDCompatTools

from . import prompts
from . import router
from . import tools as lite_tools
from .tools import StopException, TOOL_SCHEMAS, TOOL_NAMES


# Cap how much of a single observation we paste into the decision prompt.
_MAX_OBS_CHARS = 6000

# Per-question limits.
_MAX_TOTAL_CALLS = 8
_MAX_PER_TOOL = 4
_MAX_TURNS = 8
_MAX_REPEATS = 1    # same (tool, args) tuple is "you already asked"


class GVDLiteAgent:

    def __init__(
        self,
        graph_path: str,
        video_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        text_encoder=None,
        llm=None,
        max_react_turns: int = 6,    # kept for compat; the new loop uses _MAX_TURNS
        enable_detector: bool = False,
    ):
        self.llm = llm
        self.max_react_turns = max_react_turns

        from gvd.agent import _load_faiss, _load_graph
        self._graph_path = graph_path
        graph = _load_graph(graph_path)
        self.graph = graph

        self.toolkit = VKGToolkit(
            graph,
            faiss_index=_load_faiss(faiss_index_path) if faiss_index_path else None,
            text_encoder=text_encoder,
            llm_complete=self.llm.complete if self.llm is not None else None,
            inferred_edges_path=graph_path + ".inferred_edges.json",
        )
        self.dvd = DVDCompatTools(
            self.toolkit, self.llm, video_path,
            enable_detector=True,
        )
        lite_tools.bind(lite_tools._ToolContext(self.toolkit, self.dvd))

    # ------------------------------------------------------------------ #
    # Per-question state
    # ------------------------------------------------------------------ #

    def _video_length(self) -> int:
        g = self.graph
        return int(max(
            (n.t_end for n in g.nodes.values() if n.t_end < 1e6),
            default=0,
        ))

    def _build_initial_messages(self, question: str) -> List[Dict]:
        return [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": prompts.USER_TEMPLATE.format(
                video_length=self._video_length(), question=question,
            )},
        ]

    # ------------------------------------------------------------------ #
    # Stage 1: router hints
    # ------------------------------------------------------------------ #

    def _execute_router_plan(self, question: str) -> List[Dict]:
        records: List[Dict] = []
        for step in router.route(question):
            tool = step["tool"]
            args = step.get("args", {})
            try:
                if tool == "read_graph":
                    result = lite_tools.read_graph(**args)
                elif tool == "look_at_frames":
                    self.dvd.current_mcq = question
                    result = lite_tools.look_at_frames(**args)
                else:
                    result = f"Unknown tool {tool!r}."
            except Exception as exc:
                result = f"Tool error in {tool}: {exc!s}"
            if isinstance(result, str) and len(result) > _MAX_OBS_CHARS:
                result = result[:_MAX_OBS_CHARS] + "\n… [truncated]"
            records.append({"tool": tool, "args": args, "result": result})
            self._tool_call_log.append({
                "source":    "router",
                "tool":      tool,
                "args":      args,
                "result":    result,
            })
        return records

    def _format_observations(self, records: List[Dict]) -> str:
        if not records:
            return ""
        blocks = []
        for i, r in enumerate(records, 1):
            arg_str = ", ".join(f"{k}={v!r}" for k, v in r["args"].items())
            blocks.append(f"### Hint {i}: {r['tool']}({arg_str})\n{r['result']}")
        return ("HINTS (router pre-fetch — verify with your own tool calls "
                "before answering, the options may hinge on details the "
                "hints missed):\n\n" + "\n\n".join(blocks))

    # ------------------------------------------------------------------ #
    # Stage 2: flexible ReAct
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canon_args(name: str, args: dict) -> tuple:
        """Normalize args for dedup. Time ranges get sorted/quantized so
        (00:00:10, 00:00:20) and (00:00:12, 00:00:18) collapse to the same
        bin — these are semantically the same window."""
        a = dict(args)
        if name == "look_at_frames" and "time_ranges" in a:
            ranges = []
            for r in (a["time_ranges"] or []):
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    ranges.append((str(r[0]), str(r[1])))
            a["time_ranges"] = tuple(sorted(ranges))
        if name == "read_graph":
            for k in ("time_start", "time_end"):
                if k in a and a[k]:
                    a[k] = str(a[k]).strip()
        if name == "count_objects" and isinstance(a.get("time_range"), (list, tuple)):
            a["time_range"] = tuple(str(x) for x in a["time_range"])
        # Any remaining unhashable values (nested lists from malformed calls)
        # are stringified so the dedup key never throws.
        for k, v in a.items():
            if isinstance(v, (list, dict)):
                a[k] = json.dumps(v, sort_keys=True)
        return (name, tuple(sorted(a.items())))

    def _execute_model_tool_call(self, name: str, args: dict) -> str:
        try:
            if name == "read_graph":
                return lite_tools.read_graph(**args)
            if name == "look_at_frames":
                self.dvd.current_mcq = self._last_question
                return lite_tools.look_at_frames(**args)
            if name == "count_objects":
                return lite_tools.count_objects(**args)
        except Exception as exc:
            return f"Tool error in {name}: {exc!s}"
        return f"Unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}."

    def _append_tool_result(self, msgs, tc, result):
        if isinstance(result, str) and len(result) > _MAX_OBS_CHARS:
            result = result[:_MAX_OBS_CHARS] + "\n… [truncated]"
        msgs.append({
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "name": tc.get("function", {}).get("name", ""),
            "content": result,
        })

    def _react(self, msgs: List[Dict]) -> Optional[str]:
        """The flexible ReAct loop. Returns the answer letter, or None if
        the model exhausted its budget."""
        per_tool_count: Dict[str, int] = {}
        seen_calls: Dict[tuple, str] = {}
        repeat_warns: Dict[tuple, int] = {}
        total_calls = 0
        turns_left = _MAX_TURNS

        # Send the hints as a user message.
        msgs.append({
            "role": "user",
            "content": self._format_observations(self._last_records),
        })

        while turns_left > 0:
            turns_left -= 1
            try:
                resp = self.llm.chat_with_tools(msgs, TOOL_SCHEMAS)
            except Exception as exc:
                if "context length" in str(exc) or "input_tokens" in str(exc):
                    self._shrink_messages(msgs)
                    try:
                        resp = self.llm.chat_with_tools(msgs, TOOL_SCHEMAS)
                    except Exception:
                        return None
                else:
                    return None
            msgs.append(resp)
            tcs = resp.get("tool_calls") or []
            if not tcs:
                # Plain text. Nudge once.
                if turns_left > 0:
                    msgs.append({
                        "role": "user",
                        "content": ("No tool call. Use the tools — call "
                                    "read_graph, look_at_frames, or "
                                    "answer(letter)."),
                    })
                    continue
                return None

            # Separate answer() calls from verification tool calls.
            # All tool_calls in one assistant message are part of the same
            # turn; we record them all but only terminate after the model
            # has actually invoked answer() AND was not blocked by guards.
            answer_letter: Optional[str] = None
            verify_blocked = False
            for tc in tcs:
                name = tc.get("function", {}).get("name", "")
                try:
                    raw_args = tc["function"].get("arguments") or "{}"
                    parsed = json.loads(raw_args)
                except Exception:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {}

                if name == "answer":
                    # Hints are NOT sufficient evidence: 105/106 of the 4B
                    # run's zero-exploration misses answered straight from
                    # them. When raw video exists, require one real
                    # look_at_frames before accepting a letter.
                    needs_look = (self.dvd.video_path
                                  and per_tool_count.get("look_at_frames", 0) == 0
                                  and total_calls < _MAX_TOTAL_CALLS
                                  and turns_left > 0)
                    if needs_look:
                        self._append_tool_result(
                            msgs, tc,
                            ("Hold on — hints are not evidence. Verify "
                             "visually first: call look_at_frames on the "
                             "question's time window, then answer."),
                        )
                        verify_blocked = True
                        continue
                    if total_calls == 0 and not self._last_records:
                        self._append_tool_result(
                            msgs, tc,
                            ("Hold on — make at least one read_graph or "
                             "look_at_frames call before answering."),
                        )
                        verify_blocked = True
                        continue
                    try:
                        lite_tools.answer(**parsed)
                    except StopException as e:
                        answer_letter = str(e)
                    except Exception as exc:
                        self._append_tool_result(
                            msgs, tc,
                            f"answer() rejected: {exc!s}. Pass exactly one letter.",
                        )
                        continue
                elif name in ("read_graph", "look_at_frames", "count_objects"):
                    total_calls += 1
                    # Per-tool cap.
                    if per_tool_count.get(name, 0) >= _MAX_PER_TOOL:
                        self._append_tool_result(
                            msgs, tc,
                            (f"You have already called {name} {per_tool_count[name]} "
                             f"times this question (max {_MAX_PER_TOOL}). "
                             "Switch tools: if the graph came up empty, "
                             "look_at_frames over the window (inspections are "
                             "saved into the graph); if frames were ambiguous, "
                             "read_graph or count_objects. If the evidence is "
                             "truly enough, call answer(letter)."),
                        )
                        continue
                    # Total cap.
                    if total_calls > _MAX_TOTAL_CALLS:
                        self._append_tool_result(
                            msgs, tc,
                            ("Tool-call budget exhausted for this question. "
                             "Call answer(letter) with your best letter."),
                        )
                        continue
                    # Dedup: same (tool, normalized args) too many times.
                    key = self._canon_args(name, parsed)
                    if key in seen_calls:
                        repeat_warns[key] = repeat_warns.get(key, 0) + 1
                        if repeat_warns[key] > _MAX_REPEATS:
                            self._append_tool_result(
                                msgs, tc,
                                (f"You have already called {name} with these "
                                 f"exact arguments and gotten this result. "
                                 f"Try a different tool, a different time "
                                 f"window, or call answer(letter) now."),
                            )
                            continue
                        prev = seen_calls[key]
                        prev_short = (prev[:500] + "…") if len(prev) > 500 else prev
                        self._append_tool_result(
                            msgs, tc,
                            (f"You already called {name} with these "
                             f"arguments. Previous result:\n{prev_short}\n"
                             "Use a DIFFERENT tool, a different time range, "
                             "or call answer(letter)."),
                        )
                        continue
                    # Real call.
                    per_tool_count[name] = per_tool_count.get(name, 0) + 1
                    result = self._execute_model_tool_call(name, parsed)
                    if isinstance(result, str) and len(result) > _MAX_OBS_CHARS:
                        result = result[:_MAX_OBS_CHARS] + "\n… [truncated]"
                    seen_calls[key] = result
                    self._append_tool_result(msgs, tc, result)
                    self._tool_call_log.append({
                        "source": "model",
                        "tool":   name,
                        "args":   parsed,
                        "result": result,
                    })
                else:
                    self._append_tool_result(
                        msgs, tc,
                        f"Unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}.",
                    )

            if answer_letter:
                return answer_letter
            if verify_blocked:
                # The model just tried to answer without verifying; don't
                # nudge again — let the next LLM turn speak for itself.
                continue
            if turns_left > 0 and total_calls < _MAX_TOTAL_CALLS:
                # End of turn; let the model decide what to do next.
                # No automatic nudge — keeps the model in charge.
                pass
        return None

    @staticmethod
    def _shrink_messages(msgs: List[Dict]) -> None:
        """Halve every user-role hint block to fit the context window."""
        for m in msgs:
            c = m.get("content") or ""
            if m.get("role") == "user" and c.startswith("HINTS ("):
                m["content"] = c[: len(c) // 2] + "\n… [shrunk to fit context]"

    # ------------------------------------------------------------------ #
    # Letter extraction (last-resort safety net)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_letter(answer: str, last_thought: str = "") -> str:
        for src in (answer, last_thought):
            if not src:
                continue
            clean = re.sub(r"<think>.*?</think>", "", src, flags=re.S)
            m = re.search(r"\b([A-D])\b", clean)
            if m:
                return m.group(1)
        return answer or ""

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self, question: str) -> Tuple[str, List[Dict]]:
        self.toolkit._search_history = []
        self._last_records: List[Dict] = []
        self._last_question = question
        self._tool_call_log: List[Dict] = []
        self._route_name: str = router.route_name(question)
        self._video_length_s: int = self._video_length()
        self._raw_answer: str = ""
        self._final_reasoning: str = ""

        msgs = self._build_initial_messages(question)
        self._last_records = self._execute_router_plan(question)
        answer_letter = self._react(msgs)
        self._raw_answer = answer_letter or ""

        if not answer_letter:
            last_thought = ""
            for m in reversed(msgs):
                if m.get("role") == "assistant" and m.get("content"):
                    last_thought = m["content"]
                    break
            self._final_reasoning = last_thought
            answer_letter = self._ensure_letter("", last_thought) or "A"
        else:
            for m in reversed(msgs):
                if m.get("role") == "assistant" and m.get("content"):
                    self._final_reasoning = m["content"]
                    break

        return answer_letter, msgs
