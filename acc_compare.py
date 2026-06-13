#!/usr/bin/env python3
"""Emit one v3-vs-v4 accuracy comparison line per completed video.

Watches results_m3.jsonl; when all of a video's questions are
answered, prints that video's v4 vs v3 score and the running overall
comparison restricted to the uids v4 has answered so far (apples to apples).
"""
import collections
import csv
import json
import os
import sys
import time

V3 = "/workspace/results_full_lvbench.jsonl"
V4 = "/workspace/results_m3.jsonl"
CSV = "/workspace/data/LVBench_full.csv"


def load(path):
    rows = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue  # partial line mid-write
                rows[r["uid"]] = r
    return rows


expected = collections.Counter()
for r in csv.DictReader(open(CSV)):
    expected[r["video_path"].rsplit(".", 1)[0]] += 1

v3 = load(V3)
reported = set()
while True:
    v4 = load(V4)
    done_by_vid = collections.Counter()
    for r in v4.values():
        done_by_vid[r["video"].rsplit(".", 1)[0]] += 1
    for vid, n in sorted(done_by_vid.items()):
        if vid in reported or n < expected[vid]:
            continue
        reported.add(vid)
        uids = [u for u, r in v4.items() if r["video"].rsplit(".", 1)[0] == vid]
        k4 = sum(v4[u]["correct"] for u in uids)
        in3 = [u for u in uids if u in v3]
        k3 = sum(v3[u]["correct"] for u in in3)
        # overall, restricted to uids both runs answered
        both = [u for u in v4 if u in v3]
        o4 = sum(v4[u]["correct"] for u in both)
        o3 = sum(v3[u]["correct"] for u in both)
        delta = (o4 - o3)
        print(f"[acc-m3] {vid}: v4 {k4}/{len(uids)} vs v3 {k3}/{len(in3)} | "
              f"overall on same {len(both)} qs: v4 {o4} ({o4/max(1,len(both)):.1%}) "
              f"vs v3 {o3} ({o3/max(1,len(both)):.1%}) [{'+' if delta >= 0 else ''}{delta}]",
              flush=True)
    sys.stdout.flush()
    time.sleep(120)
