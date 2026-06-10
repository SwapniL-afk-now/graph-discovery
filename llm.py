"""OpenAI-compatible LLM client for GVD.

Works against api.openai.com, Azure-style endpoints, or a local vLLM server
(e.g. Qwen3-VL via `vllm serve`) — anything speaking the chat-completions
protocol. Configure with env vars or constructor args:

    GVD_BASE_URL   e.g. http://localhost:8000/v1   (default: OpenAI)
    GVD_API_KEY    falls back to OPENAI_API_KEY, then "EMPTY" (vLLM)
    GVD_MODEL      orchestrator model name (default: gpt-4o)
    GVD_VLM_MODEL  frame-inspection model (default: same as GVD_MODEL)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        vlm_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        from openai import OpenAI

        self.model = model or os.environ.get("GVD_MODEL", "gpt-4o")
        self.vlm_model = vlm_model or os.environ.get("GVD_VLM_MODEL", self.model)
        self.temperature = temperature
        self._client = OpenAI(
            base_url=base_url or os.environ.get("GVD_BASE_URL") or None,
            api_key=(api_key or os.environ.get("GVD_API_KEY")
                     or os.environ.get("OPENAI_API_KEY") or "EMPTY"),
        )

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """One orchestrator step. Returns an assistant message dict with
        optional 'tool_calls' in the same shape DVD's loop consumes."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
        )
        msg = resp.choices[0].message
        out: Dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return out

    def complete(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        """Plain text completion (used by edge inference / browse synthesis)."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def complete_with_images(self, question: str, image_urls: List[str],
                             context: str = "", max_tokens: int = 768) -> str:
        """Multimodal call for frame inspection (b64 data URLs accepted)."""
        content = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
        text = question if not context else f"{context}\n\n{question}"
        content.append({"type": "text", "text": text})
        resp = self._client.chat.completions.create(
            model=self.vlm_model,
            messages=[
                {"role": "system",
                 "content": "You are a meticulous visual analyst. Answer strictly from the provided video frames; say so explicitly if the frames do not show the answer."},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
