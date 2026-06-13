# evidence_missing bucket — 64 uids, cause breakdown

| cause | count | uids |
|---|---|---|
| vlm_miscount | 20 | 223 252 674 1246 1964 2468 3187 3336 5038 304 764 1326 1855 2264 2606 3087 3320 3323 3325 3473 |
| bare_letter_tool_output | 12 | 67 71 1670 2477 2291 4187 811 1605 1636 2157 3378 3486 |
| vkg_build_gap | 10 | 2284 2411 3188 3974 19 1829 1835 2171 2596 3124 |
| vlm_misperception | 6 | 401 3385 3443 4498 5274 369 |
| question_unanswerable_from_tools | 6 | 1947 4706 1322 1333 1907 2896 |
| retrieval_positional | 4 | 4476 4482 1921 1923 |
| wrong_window | 4 | 2293 1897 1917 2098 |
| asr_gap | 2 | 3794 4846 |

## Systemic findings

### 1. bare_letter_tool_output is the dominant harness bug (12/64, ~19%)
`look_at_frames` with `sampling='dense'` ("dense count, N passes") frequently returns a **bare option letter** instead of a count or description. The agent then guesses what the letter means — sometimes "option C", sometimes "count = 3" — and is wrong either way. Worst case **uid 3486**: the same window/question returned bare 'B' on one call and bare 'C' on the next (nondeterministic). In **uid 67** the bare letter was actually correct, but the model distrusted it and answered from the hint instead. The dense-count fusion path is leaking a final MCQ letter instead of the per-segment count report (matches the known short-thought/forced-letter fallback). Affects: 67, 71, 811, 1605, 1636, 1670, 2157, 2291, 2477, 3378, 3486, 4187.

### 2. Counting via sparse frames systematically undercounts (20 vlm_miscount, mostly off by 1-2)
Static counts off by 1-2 (674: 6 vs 8 knives; 1246: 9 vs 10 trucks; 2264: 10 vs 11 machines; 1855: 3 vs 5 tiers; 3323: 5 vs 6 boxes). Event counts over long spans are worse: 10-frame budgets over 30-105 min miss whole events (223: 2 vs 4 rescues; 2468: 1 vs 2 women; 3187: 0 vs 3 bus trips). Segment-sum reports are also **truncated mid-output** (3336, 4706), forcing guesses from partial sums.

### 3. Dense action-detection can return flat zero on real events
3385 ("none in this segment" x5 vs GT 8 condiment applications) and 4498 (0 chair turns vs GT 3): fast hand/body actions that frames straddle produce a total miss, not just an undercount.

### 4. retrieval_positional: read_graph ignores the query on wide windows (4 uids)
On windows >45 min, different queries ("chair turns" / "piano player" / "list laureates") return byte-identical truncated structural dumps (4476, 4482, 1921, 1923). Wide-window truncation serves the same head-of-window nodes; the 3-call read_graph cap is burned on identical results.

### 5. vkg_build_gap (10): named visual content absent from graph
Second flamingo prop (2284), Buddha statue (2411), music box + coat changes (2171), final-score OCR (3974), Big Bang name cards beyond Daesung (3124), award-category cards (1835), later birthdays (19). Once absent from both graph and sampled frames, the model rationalizes the question as a "trick" and guesses.

### 6. Whole-video ordinal/identity tracking is out of scope for the toolset (6)
"How many hosts changed", "different outfits", "third morning", "29 dogs": require exhaustive identity/ordinal tracking nothing provides; tool outputs contradictory (1947: 2 vs 8 hosts in consecutive calls) or truncated (4706).

### 7. Behavioral multiplier
When evidence contradicts all options (frames show 0, options start at 1), the model falls back to the router hint's letter or a "closest option" heuristic (2291, 3385, 3320, 4498, 2284) — converting a broken evidence link directly into the hint's error.

### Other tool/harness bugs observed
- `read_graph` rejects zero-length windows ("time_end must be after time_start") for point-timestamp questions (2284, 2291), wasting a call.
- `look_at_frames` sometimes executes `sampling='detail'` as "dense count, N passes" (4187, 3378: response header disagrees with requested sampling).
- Duplicate-call guard + 3-call read_graph cap leave no recovery path when results are positional (1921, 1923, 4476, 4482).
- Timestamps >60 min in MM:SS ("69:06") are handled correctly but the model wastes calls doubting the 69:xx -> 01:09:xx conversion (369).
