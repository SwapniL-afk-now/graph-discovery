# no_exploration bucket — 106 questions answered after ≤1 tool call, citing pre-injected hints

## Cause counts

| cause | n | % |
|---|---|---|
| hint_wrong | 47 | 44% |
| counting_from_hint | 21 | 20% |
| hint_too_coarse | 16 | 15% |
| hint_ambiguous_options_close | 12 | 11% |
| hint_correct_option_misread | 5 | 5% |
| harness_quote_parse_loop | 2 | 2% |
| dataset_duplicate_options | 2 | 2% |
| unverifiable_anyway | 1 | 1% |

## Fixable by one forced verification look: ~40–45% (≈44/106)

- **hint_wrong (47)**: ~half fixable. Two sub-modes:
  - *Wrong moment / wrong entity sampled* (1231, 2288, 2301, 494, 4762, 5270, 2888): targeted look at the exact instant/person plausibly fixes — best ROI.
  - *Same-frames misdescription* (colors 2398/1898/25; OCR/sign reading 665/3078/3105/3481; identity 3961/4753/2154; spatial/perspective 1885/1937/1938/1863): same VLM re-looking would repeat the error; needs zoom/crop or different prompting, not "look again". Source is overwhelmingly the frame-captioning VLM; graph-sourced falsehoods: 979, 551, 1745, 4854 (e.g. 551: scene node literally labeled "Sky with Clouds" when GT is a water reflection — question is a trick on exactly that confusion).
- **counting_from_hint (21)**: low fixability (~25%). Re-counting from ~10 frames repeats off-by-one/occlusion errors (503, 3361, 3102, 2256). Notable: 4693 — VLM reported "all four legs" on a three-legged dog (prior-driven normalization). Event counts over time (3382 spoonfuls, 2121 speakers, 2490, 4694 audio "Yes" count) need dense/multi-window sampling or dialogue.
- **hint_too_coarse (16)**: ~half fixable — discriminating detail (bread-making location 2658, who Mikhail is 2877, vlogger name 2759) was one targeted call away.
- **hint_ambiguous_options_close (12)**: color shades (259, 648), round-vs-oval (769), adjacent emotions (491, 1179, 2069, 3994), tennis who-did-what summaries (3977, 3979). Mostly below any verbal summary's resolution; ~1 in 4 fixable.
- **hint_correct_option_misread (5)**: 1076, 2787, 3439, 4699, 1095 — hints already pointed to GT; model mis-mapped evidence to option or trusted graph label over frame evidence. Fixable by reasoning, not looking.

## Systemic patterns

1. **Model rubber-stamps confident pre-fetch captions.** In ~90% of cases the CoT quotes the look_at_frames hint as "explicit" evidence and declares no further calls needed, despite the prompt's "verify with your own tool calls" warning. The 4B model has no mechanism to doubt a confident caption.
2. **Frame-caption VLM error modes dominate**: dark-scene colors, OCR of small/foreign text, left/right and camera-vs-stage perspective, character identity, still-frame pose read as the action (5277 "shooting stance", 1736 hug read as slap), prior-driven normalization (4693).
3. **Nobel-stage cluster** (1856, 1861, 1863, 1878, 1885, 1898, 1899, 1937, 1938, 1961): same video family with repeated spatial/perspective and count errors — one bad pre-fetch convention poisons many questions.
4. **Harness bug (3183, 4705)**: nested double quote inside `thought` (quoting OCR like `Comment "LIKE"`) breaks tool-call JSON parsing; 5x "No tool call" loop, then a wrong fallback letter. In 4705 the hint contained the CORRECT answer (OCR "Then it's 4 woofs" = GT Four); the point was lost purely to the parser. Fix: escape quotes in thought or trigger the short-thought fallback earlier.
5. **Dataset flaws**: 5256 (options C and D both "1 to 2") and 1821 (B "desktop" vs D "tabletop" mics are synonyms) are coin flips; exclude from model-error accounting.

## Recommendation
A forced verification look helps most where the hint sampled the wrong moment/entity (~20-25 uids). For misdescription/counting errors the verification must change the observation, not repeat it: zoomed crops for OCR/counts, explicit left-to-right enumeration prompts, and "the hint may be wrong — check each option against the frames" framing. Also fix the quote-escaping harness bug (free 1-2 points).
