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
from typing import Annotated as A
from typing import Optional

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
            self.dvd_tools = DVDCompatTools(self.toolkit, self.llm, video_path)
            self.tools += self.dvd_tools.tools()
        self.tools.append(finish)

        self.name_to_function_map = {t.__name__: t for t in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(t), "type": "function"}
            for t in self.tools
        ]

        video_length = int(max((n.t_end for n in graph.nodes.values()), default=0))
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self, question: str):
        """ReAct loop. Returns (answer, full message transcript)."""
        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"] = msgs[-1]["content"].replace(
            "QUESTION_PLACEHOLDER", question)

        answer = None
        for i in range(self.max_iterations):
            if i == self.max_iterations - 1:
                msgs.append({"role": "user",
                             "content": "Please call the `finish` function to finish the task."})

            response = self.llm.chat_with_tools(msgs, self.function_schemas)
            msgs.append(response)

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                # Model answered in plain text without finish — accept it.
                answer = response.get("content", "")
                break
            try:
                for tc in tool_calls:
                    self._exec_tool(tc, msgs)
            except StopException as exc:
                answer = str(exc)
                break

        return answer, msgs
