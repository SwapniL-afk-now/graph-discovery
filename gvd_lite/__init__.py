"""GVD-lite — a simplified agentic framework for 4B models.

Replaces the 7-tool GVD pipeline (plan → execute → claim-verify → adjudicate)
with:

  1. A deterministic Python router that classifies the question and runs
     a fixed 2-3 step tool sequence (no LLM planning).
  2. Three unified tools:
       read_graph(query, time_start?, time_end?, focus?)
         — Python routes this to the right VKG query (why / temporal /
           read_moment / find), so the 4B never has to pick between 4 tools.
       look_at_frames(time_ranges, question)
         — VLM over actual video frames.
       answer(letter)
         — terminate with exactly "A", "B", "C", or "D".
  3. A one-shot decision call: the model sees the gathered observations and
     is asked to pick a letter. If it doesn't, a short 4-turn ReAct loop
     runs as a fallback.

Total target: ~1,100 LOC across 7 files (down from 3,889 LOC / 13 files in gvd/).

The gvd/ package is imported for the underlying toolkit (graph, frame
extraction, VLM) but the agent's loop and prompts are entirely new.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Make sibling gvd importable so we can re-use VKGToolkit and DVDCompatTools
sys.path.insert(0, _ROOT)

from .agent import GVDLiteAgent  # noqa: E402,F401
