#!/usr/bin/env python3
"""Print compact transcript digests for given uids (4B run misses).

Usage: python3 digest_uid.py UID [UID...]      # full digest
       python3 digest_uid.py --chars 300 UID...# tighter tool snippets
"""
import csv, json, sys, textwrap

args = sys.argv[1:]
chars = 700
if args and args[0] == "--chars":
    chars = int(args[1]); args = args[2:]

qm = {r["uid"]: r for r in csv.DictReader(open("/workspace/data/LVBench_full.csv"))}
res = {}
for line in open("/workspace/results_gvd_lite_full.jsonl"):
    r = json.loads(line)
    res[str(r["uid"])] = r

for uid in args:
    r = res.get(uid, {})
    q = qm.get(uid, {})
    print(f"\n======== uid={uid} pred={r.get('predicted')} GT={r.get('ground_truth')} "
          f"types={r.get('question_types')} time_ref={r.get('time_reference')}")
    print("Q:", textwrap.shorten((q.get("question") or "").replace("\n", " | "), 500))
    try:
        t = json.load(open(f"/workspace/results_gvd_lite_full.jsonl.transcripts/{uid}.json"))
    except FileNotFoundError:
        print("  (no transcript)")
        continue
    for m in t["messages"]:
        role = m["role"]
        if role == "system":
            continue
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            names = [c["function"]["name"] for c in tcs]
            print(f"[asst {names}] "
                  + textwrap.shorten((m.get("content") or "").replace("\n", " "), 400))
            for c in tcs:
                print("   args:", c["function"]["arguments"][:220].replace("\n", " "))
        elif role == "tool":
            print("[tool] " + textwrap.shorten(m["content"].replace("\n", " "), chars))
        elif role == "user":
            c = m.get("content") or ""
            if c.startswith("HINTS"):
                print("[hints] " + textwrap.shorten(c.replace("\n", " "), chars))
            else:
                print("[user] " + textwrap.shorten(c.replace("\n", " "), 250))
