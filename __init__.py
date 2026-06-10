"""GVD — Graph Video Discovery.

An agentic long-video understanding framework built on top of Deep Video
Discovery (DVD). It keeps DVD's ReAct tool-calling loop and its three tools
(global browse, clip search, frame inspect), but replaces the flat
NanoVectorDB data layer with a typed Video Knowledge Graph (qvkg) and adds
graph-native tools (search, query, traverse, causal, entity, window).

Every tool observation is serialized with explicit *affordances* — a footer
that tells the orchestrator model which follow-up tool calls the evidence
supports — so the agent always knows what to do next.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Make the sibling qvkg repo importable without installation (skipped if
# qvkg is pip-installed or already on the path).
_QVKG = os.path.join(_ROOT, "video-reasoning", "qvkg")
if os.path.isdir(_QVKG) and _QVKG not in sys.path:
    sys.path.insert(0, _QVKG)

from .agent import GVDAgent  # noqa: E402,F401
from .vkg_tools import VKGToolkit  # noqa: E402,F401
