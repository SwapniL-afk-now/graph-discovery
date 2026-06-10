"""Test VLLMToolClient + GVDAgent without a GPU.

Stubs the `vllm` module and the engine: the fake engine returns scripted
constrained-decode JSON actions, exactly what the real engine would emit under
StructuredOutputsParams. Verifies schema construction, transcript conversion,
tool-call round-tripping, and the full agent loop.

Run:  python3 -m gvd.tests.test_vllm_backend
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ---------------------------------------------------------------------------
# Stub the vllm package before importing gvd.vllm_llm
# ---------------------------------------------------------------------------

vllm_stub = types.ModuleType("vllm")
sp_stub = types.ModuleType("vllm.sampling_params")


class SamplingParams:
    def __init__(self, **kw):
        self.kw = kw


class StructuredOutputsParams:
    def __init__(self, json=None):
        self.json = json


vllm_stub.SamplingParams = SamplingParams
sp_stub.StructuredOutputsParams = StructuredOutputsParams
sys.modules["vllm"] = vllm_stub
sys.modules["vllm.sampling_params"] = sp_stub

import gvd  # noqa: F401
from gvd.agent import GVDAgent
from gvd.vllm_llm import (VLLMToolClient, _convert_messages,
                          _tool_action_schema)
from gvd.tests.test_smoke import build_demo_graph


class FakeOutput:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text)]


class FakeEngine:
    """Returns scripted constrained-JSON actions like the real engine would."""

    script = [
        {"thought": "I should orient first.", "tool": "vkg_overview", "arguments": {}},
        {"thought": "Find the leaving event.", "tool": "vkg_search",
         "arguments": {"query": "man leaves house suitcase"}},
        {"thought": "Trace why ev_3 happened.", "tool": "vkg_causal",
         "arguments": {"node_id": "ev_3", "direction": "why"}},
        {"thought": "Evidence is sufficient.", "tool": "finish",
         "arguments": {"answer": "He left because the argument over unpaid bills escalated."}},
    ]

    def __init__(self):
        self.i = 0
        self.seen_messages = []
        self.seen_sampling = []

    def chat(self, messages, sampling_params, chat_template_kwargs=None, use_tqdm=False):
        self.seen_messages.append(messages[0])
        self.seen_sampling.append(sampling_params)
        step = self.script[self.i]
        self.i += 1
        return [FakeOutput(json.dumps(step))]


def main():
    g = build_demo_graph()
    g.save("/tmp/gvd_demo_graph.json")

    # --- unit: action schema covers every tool with const-pinned names -------
    fake_schemas = [
        {"type": "function", "function": {"name": "t1", "description": "d",
                                          "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "t2", "description": "d", "parameters": {}}},
    ]
    schema = _tool_action_schema(fake_schemas)
    assert len(schema["anyOf"]) == 2
    assert schema["anyOf"][0]["properties"]["tool"] == {"const": "t1"}
    json.dumps(schema)
    print("schema construction OK")

    # --- unit: transcript conversion -----------------------------------------
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "thinking",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "vkg_search", "arguments": '{"query": "x"}'}}]},
        {"role": "tool", "name": "vkg_search", "content": "RESULT"},
        {"role": "user", "content": "follow-up"},
    ]
    conv = _convert_messages(msgs, catalog="CATALOG")
    assert conv[0]["role"] == "system" and "CATALOG" in conv[0]["content"]
    assert conv[2]["role"] == "assistant" and "vkg_search" in conv[2]["content"]
    # tool observation + following user turn merged into one user message
    assert conv[3]["role"] == "user" and "Observation from vkg_search" in conv[3]["content"]
    assert "follow-up" in conv[3]["content"]
    assert len(conv) == 4
    print("message conversion OK")

    # --- integration: full agent loop on the fake engine ---------------------
    engine = FakeEngine()
    llm = VLLMToolClient(engine=engine)
    agent = GVDAgent(graph_path="/tmp/gvd_demo_graph.json", llm=llm, max_iterations=8)
    answer, transcript = agent.run("Why did the man leave the house?")

    assert "unpaid bills" in answer, answer
    tool_names = [m["name"] for m in transcript if m.get("role") == "tool"]
    assert tool_names == ["vkg_overview", "vkg_search", "vkg_causal"], tool_names

    # The constrained schema sent to the engine covers all 12 tools.
    so = engine.seen_sampling[0].kw["structured_outputs"]
    tool_consts = [b["properties"]["tool"]["const"] for b in so.json["anyOf"]]
    assert "finish" in tool_consts and "vkg_causal" in tool_consts \
        and "frame_inspect_tool" in tool_consts, tool_consts
    print(f"agent loop OK — answer: {answer!r}")
    print(f"constrained schema covers {len(tool_consts)} tools: {', '.join(tool_consts)}")

    # Observations were folded into user turns for the next engine call.
    last_msgs = engine.seen_messages[-1]
    assert any(m["role"] == "user" and "Observation from vkg_causal" in m["content"]
               for m in last_msgs)
    print("observation feedback OK")

    print("\nALL VLLM BACKEND TESTS PASSED")


if __name__ == "__main__":
    main()
