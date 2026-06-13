"""OpenAI-compatible LLM client for gvd_lite.

Thin re-export of gvd.llm.LLMClient so the lite agent doesn't need a
separate backend. Honors the same env vars (GVD_BASE_URL, GVD_API_KEY,
GVD_MODEL, GVD_VLM_MODEL) as the full GVD package.
"""

from gvd.llm import LLMClient

__all__ = ["LLMClient"]
