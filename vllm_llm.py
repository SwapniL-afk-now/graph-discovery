"""In-process vLLM backend for GVD — local models (e.g. Qwen3.5-4B), no API.

Drop-in replacement for :class:`gvd.llm.LLMClient`: implements the same three
methods (``chat_with_tools``, ``complete``, ``complete_with_images``) so
``GVDAgent`` works unchanged — swapping API → vLLM is just passing a different
``llm=`` object.

Tool calling without a server-side tool parser:
  * The full tool catalog (names + JSON parameter schemas) is appended to the
    system prompt.
  * Each orchestrator step decodes under a **constrained JSON schema**
    (vLLM structured outputs): ``{"thought": str, "tool": <one of the tool
    names>, "arguments": <that tool's parameter schema>}`` — an ``anyOf`` over
    per-tool branches. A small model physically cannot emit a malformed or
    unknown tool call.
  * The decoded action is converted back into the OpenAI ``tool_calls`` shape
    DVD's loop consumes; tool results (role "tool") are folded back into the
    transcript as "Observation from <tool>" user turns, which every chat
    template accepts.

The engine is built with :func:`qvkg.vllm_client.build_llm` (prefix caching,
lazy load), or you can hand in an already-constructed engine to share it with
the VKG build phase.
"""

import json
from typing import Dict, List, Optional


_TOOL_PROMPT_SUFFIX = """

AVAILABLE TOOLS (you MUST respond by choosing exactly one per turn):
{catalog}

RESPONSE FORMAT — a single JSON object, nothing else:
{{"thought": "<your step-by-step reasoning for this move>", "tool": "<tool name>", "arguments": {{...}}}}

End the task by choosing the `finish` tool with your final answer."""


def _tool_action_schema(function_schemas: List[Dict]) -> Dict:
    """Constrained-decode schema: anyOf over one branch per registered tool."""
    branches = []
    for fs in function_schemas:
        f = fs["function"]
        params = dict(f.get("parameters") or {"type": "object", "properties": {}})
        params.setdefault("type", "object")
        branches.append({
            "type": "object",
            "required": ["thought", "tool", "arguments"],
            "properties": {
                "thought": {"type": "string", "maxLength": 2000},
                "tool": {"const": f["name"]},
                "arguments": params,
            },
            "additionalProperties": False,
        })
    return {"anyOf": branches}


def _render_catalog(function_schemas: List[Dict]) -> str:
    lines = []
    for fs in function_schemas:
        f = fs["function"]
        desc = " ".join((f.get("description") or "").split())
        props = (f.get("parameters") or {}).get("properties", {})
        args = ", ".join(props.keys()) or "(no arguments)"
        lines.append(f"- {f['name']}({args}): {desc}")
    return "\n".join(lines)


def _convert_messages(messages: List[Dict], catalog: str) -> List[Dict]:
    """OpenAI tool-protocol transcript → plain chat any template accepts.

    assistant+tool_calls → assistant text with the action JSON inline;
    role "tool" → user "Observation from <name>". Consecutive same-role
    messages are merged (some chat templates reject alternation breaks).
    """
    out: List[Dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append({"role": "system", "content": m["content"] + _TOOL_PROMPT_SUFFIX.format(catalog=catalog)})
            continue
        if role == "assistant" and m.get("tool_calls"):
            parts = [m.get("content") or ""]
            for tc in m["tool_calls"]:
                parts.append(json.dumps({
                    "thought": "",
                    "tool": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"] or "{}"),
                }))
            out.append({"role": "assistant", "content": "\n".join(p for p in parts if p)})
            continue
        if role == "tool":
            text = f"Observation from {m.get('name', 'tool')}:\n{m['content']}"
            out.append({"role": "user", "content": text})
            continue
        out.append({"role": role, "content": m["content"]})

    merged: List[Dict] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] and isinstance(m["content"], str) \
                and isinstance(merged[-1]["content"], str):
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


class VLLMToolClient:
    """LLMClient-compatible wrapper around an in-process vLLM engine."""

    def __init__(
        self,
        engine=None,                       # vllm.LLM or qvkg LazyLLM; built if None
        model: str = "Qwen/Qwen3.5-4B",
        gpu_memory_utilization: float = 0.65,
        tensor_parallel_size: int = 1,
        max_model_len: int = 65536,
        max_images_per_prompt: int = 10,
        enable_thinking: bool = False,     # reasoning lives in the "thought" field
        max_action_tokens: int = 2048,
        lazy: bool = True,
        action_temperature: float = 0.0,   # >0 for self-consistency voting runs
        sampling_seed: Optional[int] = None,
    ):
        if engine is None:
            from qvkg.vllm_client import build_llm
            engine = build_llm(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_images_per_prompt=max_images_per_prompt,
                lazy=lazy,
            )
        self.engine = engine
        self.max_images_per_prompt = max_images_per_prompt
        self.enable_thinking = enable_thinking
        self.max_action_tokens = max_action_tokens
        self.action_temperature = action_temperature
        self.sampling_seed = sampling_seed
        self._tc_counter = 0
        # Per-toolset cache: (catalog string, SamplingParams with schema)
        self._schema_cache: Dict[int, tuple] = {}

    # ------------------------------------------------------------------ #

    def _chat(self, messages: List[Dict], sampling_params) -> str:
        out = self.engine.chat(
            messages=[messages],
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            use_tqdm=False,
        )[0]
        comp = out.outputs[0]
        # Kept for failure diagnostics: WHY did the engine stop generating?
        self._last_stop = (getattr(comp, "finish_reason", None),
                           getattr(comp, "stop_reason", None),
                           len(getattr(comp, "token_ids", []) or []))
        return comp.text.strip()

    def _action_setup(self, function_schemas: List[Dict]):
        # Key by tool names, NOT id(list): CPython reuses ids of freed lists,
        # so id-keying can silently serve a stale grammar built for a
        # different toolset (per-question agents rebuild the schema list).
        key = tuple(fs["function"]["name"] for fs in function_schemas)
        if key not in self._schema_cache:
            from vllm import SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            catalog = _render_catalog(function_schemas)
            # Greedy by default; voting runs pass action_temperature > 0 so the
            # k trajectories decorrelate. Schema validity is enforced either way.
            sampling = SamplingParams(
                temperature=self.action_temperature,
                top_p=1.0 if self.action_temperature == 0.0 else 0.95,
                seed=self.sampling_seed,
                max_tokens=self.max_action_tokens,
                structured_outputs=StructuredOutputsParams(
                    json=_tool_action_schema(function_schemas)),
            )
            # The retry must actually DIFFER from the first attempt: greedy +
            # fixed seed would regenerate the identical broken output. Bump
            # temperature and seed so the model takes a different path, and
            # double the budget in case it really was truncation.
            retry = SamplingParams(
                temperature=max(0.3, self.action_temperature),
                top_p=0.95,
                seed=(self.sampling_seed or 0) + 1,
                max_tokens=self.max_action_tokens * 2,
                structured_outputs=StructuredOutputsParams(
                    json=_tool_action_schema(function_schemas)),
            )
            # Last resort: force a SHORT thought. Some failures stall inside a
            # long thought string; a tight maxLength makes the grammar close
            # the string early so the tool call itself still comes out intact.
            brief_schema = _tool_action_schema(function_schemas)
            for br in brief_schema["anyOf"]:
                br["properties"]["thought"] = {"type": "string", "maxLength": 240}
            brief = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.max_action_tokens,
                structured_outputs=StructuredOutputsParams(json=brief_schema),
            )
            self._schema_cache[key] = (catalog, sampling, retry, brief)
        return self._schema_cache[key]

    # ------------------------------------------------------------------ #
    # LLMClient interface
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_action(text: str):
        """Parse the action JSON, salvaging what strict json.loads rejects:
        raw control characters inside strings (the grammar permits them) and
        stray bytes around the object. Returns a dict or None."""
        cands = [text]
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            cands.append(text[i:j + 1])
        for cand in cands:
            for kw in ({}, {"strict": False}):
                try:
                    obj = json.loads(cand, **kw)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    return obj
        return None

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """One orchestrator step under constrained decoding.

        Returns the same assistant-message shape as the OpenAI client, so
        GVDAgent's loop and transcript format stay identical.
        """
        catalog, sampling, retry_sampling, brief_sampling = self._action_setup(tools)
        converted = _convert_messages(messages, catalog)
        text = self._chat(converted, sampling)

        action = self._parse_action(text)
        if action is None:
            print(f"[gvd] action JSON unparseable ({len(text)} chars, "
                  f"stop={getattr(self, '_last_stop', None)}): {text[:300]!r} "
                  "— retrying with varied sampling")
            text = self._chat(converted, retry_sampling)
            action = self._parse_action(text)
        if action is None:
            print(f"[gvd] retry unparseable ({len(text)} chars, "
                  f"stop={getattr(self, '_last_stop', None)}) — forcing a "
                  "short-thought generation")
            text = self._chat(converted, brief_sampling)
            action = self._parse_action(text)
            if action is None:
                print(f"[gvd] short-thought attempt also unparseable "
                      f"({len(text)} chars, stop={getattr(self, '_last_stop', None)}): "
                      f"{text[:300]!r} — falling back to plain text")
                return {"role": "assistant", "content": text}

        self._tc_counter += 1
        return {
            "role": "assistant",
            "content": action.get("thought", ""),
            "tool_calls": [{
                "id": f"vllm_call_{self._tc_counter}",
                "type": "function",
                "function": {
                    "name": action.get("tool", ""),
                    "arguments": json.dumps(action.get("arguments") or {}),
                },
            }],
        }

    def complete(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        from vllm import SamplingParams
        return self._chat(messages, SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_tokens))

    def complete_with_images(self, question: str, image_urls: List[str],
                             context: str = "", max_tokens: int = 768) -> str:
        from vllm import SamplingParams

        # Respect the engine's per-prompt image limit by even subsampling.
        urls = image_urls
        if len(urls) > self.max_images_per_prompt:
            step = len(urls) / self.max_images_per_prompt
            urls = [urls[int(i * step)] for i in range(self.max_images_per_prompt)]

        content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
        text = question if not context else f"{context}\n\n{question}"
        content.append({"type": "text", "text": text})
        messages = [
            {"role": "system",
             "content": "You are a meticulous visual analyst. Answer strictly from the provided video frames; say so explicitly if the frames do not show the answer."},
            {"role": "user", "content": content},
        ]
        return self._chat(messages, SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_tokens))
