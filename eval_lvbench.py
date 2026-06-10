#!/usr/bin/env python3
"""LVBench evaluation with the GVD agent on a local vLLM model (no API).

Mirrors video-reasoning/qvkg/scripts/eval_lvbench.py: same CSV format, same
per-video layout (<vkg_dir>/<video_id>/vkg.json + vkg.index), same resume-safe
JSONL output and per-question-type accuracy report — but answers come from
GVDAgent's ReAct tool-calling loop running on an in-process vLLM engine
(e.g. Qwen3.5-4B) with schema-constrained tool calls.

Usage:
    python3 -m gvd.eval_lvbench \
        --csv LVBench_full.csv \
        --vkg-dir /workspace/vkgs \
        --video-dir /workspace/videos \
        --out results_gvd.jsonl \
        --model /workspace/models/Qwen3.5-4B
"""

import argparse
import ast
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gvd  # noqa: F401  (path setup for qvkg/dvd siblings)


# ---------------------------------------------------------------------------
# CSV / results helpers (same conventions as qvkg's eval)
# ---------------------------------------------------------------------------

def load_questions(csv_path: str) -> Dict[str, List[dict]]:
    by_video: Dict[str, List[dict]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_video[row["video_path"]].append(row)
    return by_video


def load_answered_uids(out_path: str) -> set:
    answered = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    answered.add(json.loads(line)["uid"])
    return answered


def parse_qt(raw: str) -> List[str]:
    try:
        v = ast.literal_eval(raw)
        return list(v) if isinstance(v, (list, tuple)) else [str(v)]
    except Exception:
        return [raw] if raw else []


def print_accuracy(results: List[dict]) -> None:
    if not results:
        print("No results.")
        return
    c = sum(1 for r in results if r["correct"])
    print(f"\nOverall: {c}/{len(results)} = {100 * c / len(results):.1f}%")
    by_qt: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        for qt in r.get("question_types", []):
            by_qt[qt][1] += 1
            if r["correct"]:
                by_qt[qt][0] += 1
    print("Per question type:")
    for qt, (cc, tt) in sorted(by_qt.items()):
        print(f"  {qt:30s} {cc}/{tt} = {100 * cc / tt:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",       required=True, help="LVBench CSV path")
    p.add_argument("--vkg-dir",   required=True, help="Directory with per-video VKG subdirs")
    p.add_argument("--video-dir", required=True, help="Directory containing .mp4 files")
    p.add_argument("--out",       required=True, help="Output JSONL path (appended, resume-safe)")
    p.add_argument("--model",     default="Qwen/Qwen3.5-4B")
    p.add_argument("--tp",        type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    p.add_argument("--max-model-len",  type=int, default=65536)
    p.add_argument("--max-iterations", type=int, default=10,
                   help="Max ReAct tool-calling turns per question")
    p.add_argument("--limit",     type=int, default=None, help="Max questions (debug)")
    p.add_argument("--no-siglip", action="store_true",
                   help="Skip SigLIP — vkg_search falls back to lexical mode")
    p.add_argument("--video",     default=None,
                   help="Restrict evaluation to one video id (e.g. for run_rolling)")
    args = p.parse_args()

    from gvd.agent import GVDAgent
    from gvd.vllm_llm import VLLMToolClient
    from qvkg.vllm_client import extract_mcq_answer

    # SigLIP first (small), engine lazily — same load order qvkg uses so the
    # encoder is on GPU before vLLM reserves its memory budget.
    siglip = None
    if not args.no_siglip:
        from qvkg.vllm_client import build_siglip_encoder
        print("Loading SigLIP encoder...")
        siglip = build_siglip_encoder()

    llm = VLLMToolClient(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        lazy=True,
    )

    questions_by_video = load_questions(args.csv)
    answered_uids = load_answered_uids(args.out)
    all_results: List[dict] = []
    total_done = 0

    with open(args.out, "a") as out_f:
        for video_filename, questions in questions_by_video.items():
            video_id = os.path.splitext(video_filename)[0]
            if args.video and video_id != args.video:
                continue

            vkg_path   = os.path.join(args.vkg_dir, video_id, "vkg.json")
            index_path = os.path.join(args.vkg_dir, video_id, "vkg.index")
            video_path = os.path.join(args.video_dir, video_filename)

            if not os.path.exists(vkg_path):
                print(f"  [SKIP] No VKG for {video_filename}")
                continue

            pending = [q for q in questions if q["uid"] not in answered_uids]
            if args.limit:
                pending = pending[: max(0, args.limit - total_done)]
            if not pending:
                print(f"\nVideo: {video_filename} — all questions already answered.")
                continue

            print(f"\nVideo: {video_filename} ({len(pending)} pending / {len(questions)} total)")
            agent = GVDAgent(
                graph_path=vkg_path,
                video_path=video_path if os.path.exists(video_path) else None,
                faiss_index_path=index_path if os.path.exists(index_path) else None,
                text_encoder=siglip,
                llm=llm,
                max_iterations=args.max_iterations,
            )

            for q in pending:
                uid       = q["uid"]
                gt_answer = q["answer"].strip().upper()
                qt        = parse_qt(q.get("question_type", ""))
                time_ref  = (q.get("time_reference") or "").strip() or None

                mcq_question = (
                    q["question"]
                    + "\n\nThis is a multiple-choice question. When you call finish, "
                      "the answer must contain exactly one letter: A, B, C, or D."
                )
                if time_ref:
                    mcq_question = f"[Time reference: {time_ref}]\n{mcq_question}"

                try:
                    answer, msgs = agent.run(mcq_question)
                    pred = extract_mcq_answer(answer or "") or "ERROR"
                    n_calls = sum(len(m.get("tool_calls") or []) for m in msgs
                                  if m.get("role") == "assistant")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    pred, n_calls = "ERROR", 0

                correct = pred == gt_answer
                record = {
                    "uid":            uid,
                    "video":          video_filename,
                    "question_types": qt,
                    "time_reference": time_ref,
                    "predicted":      pred,
                    "ground_truth":   gt_answer,
                    "correct":        correct,
                    "tool_calls":     n_calls,
                    "mode":           "gvd_agent",
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                all_results.append(record)
                answered_uids.add(uid)
                total_done += 1

                status = "✓" if correct else "✗"
                print(f"  {status} uid={uid} pred={pred} gt={gt_answer} tools={n_calls}")

                if args.limit and total_done >= args.limit:
                    break
            if args.limit and total_done >= args.limit:
                break

    print_accuracy(all_results)


if __name__ == "__main__":
    main()
