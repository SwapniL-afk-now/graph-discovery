#!/usr/bin/env python3
"""Incremental failure triage for the 9B+fixes run.

Each call prints ONE line per newly-missed question since the last call
(state in /tmp/triage_9b_seen.txt), with a mechanical cause diagnosis:

  uid=514 [synthesis] pred=C gt=B tools=3 | GT evidence shown but rejected | gt_opt='...'

Buckets mirror analyze_4b_errors.py so 4B vs 9B causes are comparable.
"""
import csv, json, os, re, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/workspace/results_gvd_lite_9b_fixed.jsonl"
TDIR = RESULTS + ".transcripts"
CSV = "/workspace/data/LVBench_full.csv"
SEEN = "/tmp/triage_seen_" + os.path.basename(RESULTS) + ".txt"

STOP = set("the a an of in on at to and or is are was were with for from by as it its this that "
           "what which who when where how does did do video clip according following".split())

def kw(text):
    return {t for t in re.findall(r"[a-z0-9']+", text.lower()) if len(t) > 2 and t not in STOP}

def options(question):
    o = {m.group(1): m.group(2).strip()
         for m in re.finditer(r"\(([A-D])\)\s*(.*?)(?=\([A-D]\)|$)", question, re.S)}
    return o

def diagnose(r, qrow):
    uid = str(r["uid"])
    opts = options(qrow.get("question", ""))
    gt_text = opts.get(r["ground_truth"], "")
    tpath = os.path.join(TDIR, f"{uid}.json")
    notes, bucket = [], "?"
    cap_hit = dupe = False
    used = []
    gt_cov = 0.0
    if os.path.exists(tpath):
        msgs = json.load(open(tpath))["messages"]
        evid = " ".join((m.get("content") or "") for m in msgs
                        if m["role"] in ("tool", "user")).lower()
        gkw = kw(gt_text)
        gt_cov = (len(gkw & kw(evid)) / len(gkw)) if gkw else 0.0
        for m in msgs:
            for tc in (m.get("tool_calls") or []):
                used.append(tc["function"]["name"])
            if m["role"] == "tool":
                c = m["content"]
                cap_hit |= c.startswith("You have already called")
                dupe |= c.startswith("You already called")
    if r["predicted"] == "ERROR":
        bucket, why = "decode_error", "ERROR prediction (parse/crash)"
    elif r["tool_calls"] <= 1:
        bucket, why = "no_exploration", "answered after <=1 call despite the look gate"
    elif gt_cov >= 0.6 and len(kw(gt_text)) >= 2:
        bucket, why = "synthesis", "GT evidence was shown but rejected"
    elif gt_cov >= 1.0:
        bucket, why = "synthesis_short_opt", "1-keyword GT option appeared in evidence"
    else:
        bucket, why = "evidence_missing", "GT content never surfaced in any tool result"
    if cap_hit:
        why += "; hit per-tool cap"
    if dupe:
        why += "; repeated identical call"
    tools_str = ",".join(f"{t}x{used.count(t)}" for t in dict.fromkeys(used)) or "none"
    return (f"MISS uid={uid} [{bucket}] pred={r['predicted']} gt={r['ground_truth']} "
            f"({tools_str}) | {why} | gt_opt='{gt_text[:60]}'")

def main():
    seen = set()
    if os.path.exists(SEEN):
        seen = set(open(SEEN).read().split())
    qm = {str(row["uid"]): row for row in csv.DictReader(open(CSV))}
    if not os.path.exists(RESULTS):
        return
    new_seen = set(seen)
    for line in open(RESULTS):
        try:
            r = json.loads(line)
        except Exception:
            continue
        uid = str(r["uid"])
        if uid in seen:
            continue
        new_seen.add(uid)
        if not r["correct"]:
            print(diagnose(r, qm.get(uid, {})))
    with open(SEEN, "w") as f:
        f.write("\n".join(sorted(new_seen)))

if __name__ == "__main__":
    main()
