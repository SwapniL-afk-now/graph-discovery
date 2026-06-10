# GVD — Graph Video Discovery

An agentic long-video understanding framework built **on top of Deep Video
Discovery (DVD)**. It keeps what DVD got right — the tool-based ReAct loop with
progressive granularity (global → clip → frame) — and replaces what it got
wrong: the flat NanoVectorDB data layer becomes a **typed Video Knowledge
Graph** (built once per video by `qvkg`), and the agent gets **graph-native
tools** to traverse it.

```
                     OFFLINE (once per video)                ONLINE (per question)
┌──────────────────────────────────────────────┐   ┌──────────────────────────────────┐
│ video.mp4                                    │   │ Question                         │
│   │  qvkg builder (frames + Whisper + OCR    │   │   │                              │
│   ▼  + VLM scene extraction)                 │   │   ▼                              │
│ Typed VKG: Episode→Scene→Clip hierarchy,     │   │ GVDAgent (DVD ReAct loop,        │
│ Action/Speech/OCR/Audio events,              │──▶│ OpenAI function calling)         │
│ Character/Object entities, 30 edge types     │   │   │ THINK → ACT → OBSERVE        │
│ (+ optional FAISS index)                     │   │   ▼                              │
└──────────────────────────────────────────────┘   │ 12 tools → observations with     │
                                                   │ node ids + edge affordances +    │
                                                   │ "what you can do next" footers   │
                                                   │   │                              │
                                                   │   ▼                              │
                                                   │ finish(answer)                   │
                                                   └──────────────────────────────────┘
```

## Why (vs. DVD)

| DVD gap (see `../dvd_gaps.md`) | GVD answer |
|---|---|
| Flat vector DB, no edges | qvkg typed graph: 10 node types, 30 edge types |
| No causal reasoning | `vkg_causal` chain traversal + `vkg_infer_causal` on-the-fly inference (cached) |
| No character tracking | `vkg_entity` timelines over `entity_id` links |
| No temporal hierarchy | Episode→Scene→Clip backbone, `vkg_overview` / `vkg_window` |
| Ad-hoc retrieval | Typed `vkg_query` is exhaustive; semantic `vkg_search` is dual-mode (FAISS or lexical fallback) |
| GPT-4o-only | Any OpenAI-compatible endpoint (vLLM + Qwen3-VL works) |
| Agent doesn't know what to do next | Every observation ends with an **affordance footer** of concrete follow-up calls |

## The tool belt (12 tools)

**DVD's original three** (same names & behavior, graph/video-backed):
`global_browse_tool`, `clip_search_tool`, `frame_inspect_tool`.

**Graph-native eight**:

| Tool | Question shape it serves |
|---|---|
| `vkg_overview` | "what is this video about" — hierarchy + character registry |
| `vkg_search` | locate events/speech/text semantically; returns traversable node ids |
| `vkg_query` | exhaustive typed access ("ALL speech between 10:00–12:00") |
| `vkg_traverse` | multi-hop expansion along one edge family (CAUSAL/ENTITY/TEMPORAL/SPEAKER/SIMILAR/CONTAINS/EMOTION) |
| `vkg_causal` | "why did X happen" (backward) / "what did X lead to" (forward) |
| `vkg_entity` | character/object timeline: appearances, actions, dialog |
| `vkg_window` | close-read one moment across all modalities |
| `vkg_infer_causal` | infer + cache causal edges where the offline graph has none |

Plus `finish(answer)`.

## The key design: observations that teach the next move

Every tool output is serialized by `serializer.py` with:

1. **Actionable node lines** — `(ev_12 | ActionNode | 00:14:10–00:14:25 conf=0.85)` —
   id, type, time, confidence: everything the next tool call needs as arguments.
2. **Inline edge affordances** — `[—CAUSES→ev_15, ←PERFORMS—char_3]` — the model
   *sees* which traversals are possible from each piece of evidence.
3. **Relevance-then-chronology** — hits capped by relevance, printed in temporal
   order (cap-before-sort).
4. **An affordance footer** — copy-pasteable suggested calls derived from the
   actual evidence (causal edges present → suggest `vkg_causal` on the densest
   node; entities present → `vkg_entity`; always a `frame_inspect_tool` CONFIRM
   suggestion, preserving DVD's grounding discipline).
5. **Instructive failures** — empty results and tool errors return recovery
   strategies, not stack traces.

The division of labor is stated in the system prompt: **the graph tells you
where to look; the frames tell you what's actually there.**

## Usage

```bash
# 1. Offline: build the graph once (see video-reasoning/qvkg)
python video-reasoning/qvkg/scripts/build_vkg.py --video movie.mp4 ...

# 2. Online: ask questions
export GVD_BASE_URL=http://localhost:8000/v1   # vLLM serving Qwen3-VL (or omit for OpenAI)
export GVD_MODEL=Qwen/Qwen3-VL-8B-Instruct
python -m gvd.run_gvd --graph movie.vkg.json --video movie.mp4 \
    --question "Why did the man leave the house?"
```

Python API:

```python
from gvd import GVDAgent
agent = GVDAgent(graph_path="movie.vkg.json", video_path="movie.mp4")
answer, transcript = agent.run("Why did the man leave the house?")
```

`--faiss-index` + a SigLIP `text_encoder` enable semantic search; without them
`vkg_search` degrades to lexical token-overlap search and everything still runs.

## Layout

```
gvd/
  agent.py        GVDAgent: DVD's ReAct loop, tool registration, finish
  vkg_tools.py    VKGToolkit: the 8 graph-native tools
  dvd_compat.py   DVD's 3 original tools, graph/video-backed
  serializer.py   evidence → text with edge affordances + next-step footers
  prompts.py      DVD's THINK→ACT→OBSERVE prompt + graph mental model + routing table
  edge inference  vkg_infer_causal caches to <graph>.inferred_edges.json
  llm.py          OpenAI-compatible client (OpenAI / Azure / local vLLM)
  timeutil.py     HH:MM:SS ↔ seconds
  run_gvd.py      CLI
  tests/          smoke test (no GPU/API needed): python3 -m gvd.tests.test_smoke
```

Dependencies: `qvkg` (sibling repo, path-injected by `gvd/__init__.py`),
`pydantic`, `numpy`, `sortedcontainers`; `openai` only for the API backend,
`vllm` only for the local backend; `faiss` + an encoder only for semantic
mode; `av`/`opencv` only for frame inspection. The schema generator is
vendored in `func_schema.py` — **no code from the DeepVideoDiscovery checkout
is imported**.
