#!/usr/bin/env python3
"""Triage the 392 wrong answers from the Qwen3.5-4B gvd_lite run.

Buckets each wrong uid by mechanical signals from its transcript:
  decode_error      predicted == ERROR (harness/structured-output failure)
  no_exploration    <=1 tool call before answering
  iteration_ceiling hit the max-iteration budget (12)
  evidence_missing  GT-option content never appeared in any evidence shown
  synthesis         GT-option content WAS in the evidence, model still missed

Writes errors_4b_triage.jsonl and prints a bucket x question-type matrix.
"""
import csv, json, os, re, sys
from collections import Counter, defaultdict

RESULTS = "results_gvd_lite_full.jsonl"
TDIR = RESULTS + ".transcripts"
CSV = "data/LVBench_full.csv"
OUT = "errors_4b_triage.jsonl"
MAX_ITER = 12

STOP = set("the a an of in on at to and or is are was were with for from by as it its this that "
           "what which who when where how does did do video clip according following".split())

def keywords(text):
    toks = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in toks if len(t) > 2 and t not in STOP]

def parse_options(question):
    """Return {letter: option text} from MCQ question string."""
    opts = {}
    for m in re.finditer(r"\(([A-D])\)\s*(.*?)(?=\([A-D]\)|$)", question, re.S):
        opts[m.group(1)] = m.group(2).strip()
    if not opts:
        for m in re.finditer(r"^\s*([A-D])[.):]\s*(.*)$", question, re.M):
            opts[m.group(1)] = m.group(2).strip()
    return opts

def evidence_text(messages):
    """Concatenate everything the model saw as evidence: tool results and the
    initial user payload (VKG summary), excluding system prompt."""
    parts = []
    for m in messages:
        if m["role"] in ("tool", "user"):
            parts.append(m.get("content") or "")
    return " ".join(parts).lower()

def coverage(gt_text, evid):
    kws = keywords(gt_text)
    if not kws:
        return 0.0, 0
    hits = sum(1 for k in set(kws) if k in evid)
    return hits / len(set(kws)), len(set(kws))

def main():
    qmeta = {}
    with open(CSV) as f:
        for row in csv.DictReader(f):
            qmeta[str(row["uid"])] = row

    wrong = []
    with open(RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if not r["correct"]:
                wrong.append(r)

    rows = []
    for r in wrong:
        uid = str(r["uid"])
        meta = qmeta.get(uid, {})
        question = meta.get("question", "")
        opts = parse_options(question)
        gt_text = opts.get(r["ground_truth"], "")
        pred_text = opts.get(r["predicted"], "")

        tpath = os.path.join(TDIR, f"{uid}.json")
        n_msgs = 0
        gt_cov = pred_cov = 0.0
        gt_mentioned = gt_short_present = False
        if os.path.exists(tpath):
            t = json.load(open(tpath))
            msgs = t["messages"]
            n_msgs = len(msgs)
            evid = evidence_text(msgs)
            gt_cov, nkw = coverage(gt_text, evid)
            pred_cov, _ = coverage(pred_text, evid)
            gt_mentioned = gt_cov >= 0.6 and nkw >= 2
            gt_short_present = gt_cov >= 1.0 and nkw < 2

        if r["predicted"] == "ERROR":
            bucket = "decode_error"
        elif r["tool_calls"] <= 1:
            bucket = "no_exploration"
        elif r["tool_calls"] >= MAX_ITER:
            bucket = "iteration_ceiling"
        elif gt_mentioned:
            bucket = "synthesis"
        elif gt_short_present:
            bucket = "synthesis_short_opt"  # 1-keyword GT option, keyword did appear in evidence
        else:
            bucket = "evidence_missing"

        rows.append({
            "uid": uid, "video": r["video"], "bucket": bucket,
            "question_types": r["question_types"],
            "predicted": r["predicted"], "ground_truth": r["ground_truth"],
            "gt_option": gt_text[:120], "pred_option": pred_text[:120],
            "tool_calls": r["tool_calls"], "n_messages": n_msgs,
            "gt_evidence_coverage": round(gt_cov, 2),
            "pred_evidence_coverage": round(pred_cov, 2),
            "time_reference": r.get("time_reference"),
        })

    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    buckets = Counter(x["bucket"] for x in rows)
    print(f"wrong total: {len(rows)}")
    for b, n in buckets.most_common():
        print(f"  {b:18s} {n:4d}  ({100*n/len(rows):.1f}%)")

    print("\nbucket x question_type:")
    mat = defaultdict(Counter)
    for x in rows:
        for qt in x["question_types"]:
            mat[x["bucket"]][qt] += 1
    qts = sorted({qt for c in mat.values() for qt in c})
    print(f"{'':18s} " + " ".join(f"{qt[:12]:>12s}" for qt in qts))
    for b in buckets:
        print(f"{b:18s} " + " ".join(f"{mat[b][qt]:12d}" for qt in qts))

    print("\nwith time_reference but wrong:",
          sum(1 for x in rows if x["time_reference"]))

if __name__ == "__main__":
    main()
