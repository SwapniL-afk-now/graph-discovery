#!/usr/bin/env python3
"""Per-bucket flip rates: 9B+fixes run vs the 392 Qwen3.5-4B misses.

Joins results_gvd_lite_9b_fixed.jsonl against errors_4b_triage.jsonl on uid.
Safe to run mid-eval — reports only the uids answered so far.
"""
import json, sys
from collections import Counter

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results_gvd_lite_9b_fixed.jsonl"
triage = {r["uid"]: r for r in map(json.loads, open("errors_4b_triage.jsonl"))}
fixed = {}
for line in open(RESULTS):
    r = json.loads(line)
    fixed[str(r["uid"])] = r  # latest row wins

done = {u: r for u, r in fixed.items() if u in triage}
if not done:
    print("flip_report: none of the 392 previously-wrong uids answered yet")
    raise SystemExit(0)

tot, flips = Counter(), Counter()
for u, r in done.items():
    b = triage[u]["bucket"]
    tot[b] += 1
    if r["correct"]:
        flips[b] += 1

n, f = len(done), sum(flips.values())
print(f"flip report: {n}/392 previously-wrong answered, {f} flipped to correct ({100*f/n:.1f}%)")
for b in sorted(tot, key=lambda b: -tot[b]):
    print(f"  {b:20s} {flips[b]:3d}/{tot[b]:3d} flipped ({100*flips[b]/tot[b]:.0f}%)")
