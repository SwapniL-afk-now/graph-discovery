# Where Qwen3.5-9B can improve over the 4B run — error analysis of all 392 misses

**Source run:** `results_gvd_lite_full.jsonl` — Qwen3.5-4B, gvd_lite agent, 1274 questions, **882 correct (69.2%)**, 392 wrong.
**Method:** mechanical triage of every wrong answer (`analyze_4b_errors.py` → `errors_4b_triage.jsonl`) + manual reading of ~25 stratified transcripts. Every claim below cites uids you can open in `results_gvd_lite_full.jsonl.transcripts/`.

**Critical architectural fact** (verified in `gvd/llm.py:41` and `gvd/vllm_llm.py:284`): the *same* local model serves both the agent loop **and** the `look_at_frames` vision calls. Swapping 4B→9B therefore upgrades reasoning *and* per-question perception. What stays fixed is the VKG content itself — the Whisper transcript, OCR, and the 4B-built scene/event graph.

---

## Headline triage

| Bucket | n | % of errors | 9B-addressable? |
|---|---|---|---|
| **synthesis** — GT evidence was in front of the model, it still picked wrong | 143 | 36.5% | **Yes — primary upside** |
| **no_exploration** — answered after ≤1 tool call, trusting injected hints | 106 | 27.0% | **Mostly yes** |
| **synthesis_short_opt** — 1-word GT option present in evidence, picked wrong | 79 | 20.2% | Partially |
| **evidence_missing** — GT content never appeared in anything the model saw | 64 | 16.3% | **No — pipeline work** |

Cross-cutting signals:
- **Counting questions are a disaster zone:** 142 counting questions overall, only **41% accuracy**; 84 of the 392 errors (21%) are counts.
- **105 of the 106 no_exploration misses explicitly cite "hints"** in their first (and only) assistant turn — the model treats the pre-injected hint block as settled evidence and skips verification.
- **26 errors involve duplicate-call thrash** (the harness's "You already called X with these arguments" guard fired) — the model loops the same query instead of reformulating.
- Only **2 of 392** errors show the predicted option better-supported by evidence than the GT option — the model almost never "follows wrong evidence"; it either never sees the GT evidence or fails to use it.

---

## Bucket 1: synthesis (143, biggest 9B upside)

Evidence containing the GT answer was retrieved and shown; the final answer contradicts it. Heaviest in *event understanding* (69) and *reasoning* (35). Three mechanisms seen in transcripts:

**a) Flat contradiction of own evidence** — pure reasoning slips, the cleanest 9B win.
> uid 2498: tool returned *"two men standing near the front of the stage… the man on the left… positioned behind the podium"*. Final answer: B "Close to the center of the stage", explicitly *after* writing "Option (B) is contradicted by the visual evidence". GT: A (left side).

**b) Distractor anchoring** — evidence partially matches a distractor; the model latches onto the surface lexical match instead of the question's actual referent.
> uid 4702: dialogue evidence *"I'll give you the ball"* → picked C "the dog and the judge play the ball". But the question asks what happens *between dog and judge*; GT A (hide and seek) was also in the hints. The model matched "ball" without resolving who plays with whom.
> uid 1097: vision tool reported pigeons *and a dog* near the Charging Bull → picked B "Dog and pigeon"; GT A "Pigeon and crow". Part perception (the 4B VLM called a crow a dog or hallucinated), part anchoring.

**c) Premise-mismatch surrender** — hinted time window doesn't match the question's premise; instead of re-grounding elsewhere in the video, the model concludes the premise is false.
> uid 3128: question about a Chinese character on the *vlogger's home* wall; the hinted window shows a concert. Model searched only 15:25–16:15 twice, then answered D "no Chinese character". Never widened the search. This sub-type needs better agentic search policy — 9B helps somewhat, retrieval helps more.

**9B prediction:** types (a) and much of (b) should flip — these are exactly the multi-constraint binding errors that scale with model size. Expect 60–80 of these 143 to flip.

## Bucket 2: no_exploration / hint over-trust (106)

The agent gets pre-computed hints (a prior `read_graph`/`look_at_frames` digest) in its first user message; in 105/106 of these misses it answered immediately, citing the hints, with ≤1 verification call. The hints are coarse, and the questions that die here are fine-grained: exact action at a timestamp (uid 4762 — hint said "wiping face, raising arm", options were cry/wave/kiss/bow; guessed A, GT C), platform sign numbers (uid 3078), precise gesture (uid 5049: hand-near-wrist in hint → "holding hands"; GT "hugging").

**9B prediction:** mixed. Better instruction-following should make 9B actually run the mandated verification look (the system prompt demands it; 4B skips it), and the 9B VLM resolves fine-grained actions better when it does look. But some of these moments are genuinely ambiguous in 10 sampled frames. Expect maybe a third to flip. **Harness fix worth more:** make the first `look_at_frames` mandatory (reject `answer` at tool_calls==0/1), which costs one call per question and attacks all 106.

## Bucket 3: synthesis_short_opt (79)

One-keyword GT options ("Golden", "Tail", "Chicken"…) whose keyword appears *somewhere* in the evidence — mechanically indeterminate, so read individually. Sampled transcripts show this bucket is mostly **perception/VKG ceiling**, not reasoning:
- uid 2904: the dollar amount the vlogger paid is simply not in the Whisper transcript or OCR; `read_graph(focus=dialogue)` returned *audio peaks* instead of speech for that window (a real retrieval bug worth checking); the model probed 6 times then guessed.
- uid 1335: stirring direction (clockwise vs counter) — unrecoverable from sparse frames; model answered "no rotation".
- uid 4459: "sixth contestant" requires ordinal tracking across the whole video; the VKG has no contestant enumeration; model thrashed (3 identical `read_graph` calls).

**9B prediction:** small gains via better OCR/fine-detail reading in `look_at_frames` (color/text questions like uid 219 "Golden" or uid 1077 "Chicken" vs "Bird"); the ASR-gap and ordinal-tracking cases won't move. Expect ~15–25 flips.

## Bucket 4: evidence_missing (64, 9B won't fix — pipeline targets)

GT content never surfaced in any tool result or hint. Dominated by *entity recognition* (36) and *event understanding* (31). Mechanisms:

- **VLM miscounting upstream of the agent** (the biggest sub-group, overlaps the counting stat): uid 2264 — both `detail` and `dense` passes counted 10 smoke machines; GT 11. uid 674 — counted 6 knives twice; GT 8. The agent did everything right; the eye was wrong. *The 9B VLM may count slightly better, so a few of these can flip — but frame-sampling density and the absent `count_objects`-style detector are the real fix.*
- **Retrieval returning the same irrelevant nodes regardless of query**: uid 1923 — three differently-worded `read_graph` queries over a 58-minute window each returned the identical first speech nodes; the laureate list, if transcribed at all, was never reachable. `read_graph` appears to rank by window position, not query relevance, on wide windows.
- **VKG gaps**: content not captured at build time (uid 304 — insect count "on the board" not in scene graph; uid 2904's missing dollar amount).

**Pipeline recommendations, ranked by expected points:**
1. **Query-relevant ranking in `read_graph`** for wide windows (embedding or BM25 over node labels instead of positional order) — attacks uid-1923-style dead ends and the premise-mismatch surrender cases in bucket 1.
2. **Counting**: route "how many objects" to a detector pass (the `gvd` package already has `count_objects` per `dvd_compat.py`; gvd_lite's tool surface dropped it) — 84 errors are counts, 41% accuracy on the category.
3. **Check the `focus="dialogue"` path** that returned audio peaks instead of speech nodes (uid 2904) — looks like a filtering bug, cheap to fix.
4. Duplicate-call guard should *suggest a reformulation* (e.g. "widen the window or change focus") rather than just refuse — 26 errors thrashed.

---

## Bottom line: predicted 9B delta on identical VKGs

| Source of flips | Plausible range |
|---|---|
| synthesis reasoning slips (a+b) | +60–80 |
| no_exploration (better protocol-following + sharper VLM) | +25–40 |
| short-opt perception (OCR/fine detail) | +15–25 |
| evidence_missing (slightly better VLM counting) | +0–10 |
| **Total flips** | **+100–155 ≈ +8 to +12 points** (minus regressions, realistically **+5 to +9 points → ~74–78%**) |

The ceiling on these VKGs is set by the ~64 evidence_missing cases plus the unrecoverable share of buckets 2–3 (ASR gaps, motion direction, ordinals) — roughly **90–110 questions (~7–9 points) that no eval-model swap can reach**. Those need the four pipeline fixes above.

**Verification once the 9B run lands:** join `results_gvd_lite_full_9b.jsonl` against `errors_4b_triage.jsonl` on `uid` and compute per-bucket flip rates. If the synthesis bucket doesn't flip at well above the other buckets' rate, the capacity hypothesis is wrong and the pipeline fixes move to the front of the queue.
