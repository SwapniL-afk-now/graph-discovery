#!/usr/bin/env python3
"""LVBench evaluation with the GVD-lite agent.

Mirrors gvd/eval_lvbench.py: same CSV format, same per-video layout, same
resume-safe JSONL output and per-question-type accuracy report — but
answers come from GVDLiteAgent (3 tools, one-shot decision, short ReAct
fallback) instead of the 4-stage GVD pipeline.

Usage:
    python3 -m gvd_lite.eval_lvbench \\
        --csv LVBench_full.csv \\
        --vkg-dir /workspace/vkgs \\
        --video-dir /workspace/videos \\
        --out results_gvd_lite.jsonl \\
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

import gvd_lite  # noqa: F401  (package init)


# ---------------------------------------------------------------------------
# CSV / results helpers (mirrors gvd/eval_lvbench.py conventions)
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
    p.add_argument("--max-react-turns", type=int, default=4,
                   help="Max ReAct fallback turns per question (Stage 3)")
    p.add_argument("--limit",     type=int, default=None, help="Max questions (debug)")
    p.add_argument("--no-siglip", action="store_true",
                   help="Skip SigLIP — search_events falls back to lexical mode")
    p.add_argument("--video",     default=None,
                   help="Restrict evaluation to one video id (e.g. for run_rolling)")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name; logs running accuracy per question")
    p.add_argument("--api-model", default=None,
                   help="Use a remote OpenAI-compatible model for orchestration "
                        "and vision instead of local vLLM (e.g. MiniMax-M3). "
                        "Endpoint/key: GVD_BASE_URL / GVD_API_KEY env vars.")
    args = p.parse_args()

    from gvd_lite.agent import GVDLiteAgent

    # We import the same MCQ-letter extractor qvkg ships, so we stay
    # 1:1 comparable with results_full_lvbench.jsonl.
    from qvkg.vllm_client import extract_mcq_answer

    # SigLIP first (small), engine lazily.
    siglip = None
    if not args.no_siglip:
        from qvkg.vllm_client import build_siglip_encoder
        print("Loading SigLIP encoder...")
        siglip = build_siglip_encoder()

    if args.api_model:
        from gvd_lite.llm import LLMClient
        llm = LLMClient(model=args.api_model)
        print(f"Using API model {args.api_model} via "
              f"{os.environ.get('GVD_BASE_URL', 'api.openai.com')}")
    else:
        from gvd.vllm_llm import VLLMToolClient
        llm = VLLMToolClient(
            model=args.model,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            lazy=True,
        )

    questions_by_video = load_questions(args.csv)
    answered_uids = load_answered_uids(args.out)

    wb = None
    wb_total = wb_correct = 0
    if args.wandb_project:
        import wandb
        if os.path.exists(args.out):
            with open(args.out) as f:
                for line in f:
                    if line.strip():
                        wb_total += 1
                        wb_correct += bool(json.loads(line).get("correct"))
        wb = wandb.init(
            project=args.wandb_project,
            id=os.environ.get("WANDB_RUN_ID") or None,
            resume="allow",
            mode="online",
            config={"model": args.model, "max_react_turns": args.max_react_turns,
                    "csv": args.csv, "out": args.out,
                    "framework": "gvd_lite"},
        )
    all_results: List[dict] = []
    total_done = 0

    with open(args.out, "a") as out_f, open(args.out + ".verbose.log", "a") as vlog_f:
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

            agent = GVDLiteAgent(
                graph_path=vkg_path,
                video_path=video_path if os.path.exists(video_path) else None,
                faiss_index_path=index_path if os.path.exists(index_path) else None,
                text_encoder=siglip,
                llm=llm,
                max_react_turns=args.max_react_turns,
                enable_detector=False,
            )

            def build_mcq(q):
                mcq = (
                    q["question"]
                    + "\n\nThis is a multiple-choice question. When you call answer, "
                      "the answer must contain exactly one letter: A, B, C, or D."
                )
                time_ref = (q.get("time_reference") or "").strip() or None
                if time_ref:
                    mcq = f"[Time reference: {time_ref}]\n{mcq}"
                return mcq, time_ref

            transcripts_dir = args.out + ".transcripts"
            os.makedirs(transcripts_dir, exist_ok=True)

            def run_traj(mcq, uid=None):
                try:
                    answer, msgs = agent.run(mcq)
                    n_calls = sum(len(m.get("tool_calls") or []) for m in msgs
                                  if m.get("role") == "assistant")
                    if uid is not None:
                        with open(os.path.join(transcripts_dir, f"{uid}.json"), "w") as tf:
                            json.dump({"answer": answer, "messages": msgs}, tf,
                                      indent=1, default=str)
                    return answer, n_calls
                except Exception:
                    import traceback
                    traceback.print_exc()
                    return "ERROR", 0

            for q in pending:
                uid = q["uid"]
                gt_answer = q["answer"].strip().upper()
                question_text = q.get("question", "")
                mcq_question, _ = build_mcq(q)

                pred, n_calls = run_traj(mcq_question, uid)
                # Normalize: ensure we return a single A/B/C/D letter.
                pred_norm = extract_mcq_answer(pred or "") or pred or "ERROR"
                if pred_norm not in "ABCD":
                    pred_norm = "ERROR"

                # Snapshot what the agent did, for post-hoc debugging.
                tool_log = list(getattr(agent, "_tool_call_log", []))
                last_reasoning = (getattr(agent, "_final_reasoning", "") or "").strip()
                transcript_path = os.path.join(
                    transcripts_dir, f"{uid}.json"
                )
                search_history = list(
                    getattr(agent.toolkit, "_search_history", []) or []
                )

                record = {
                    "uid":            uid,
                    "video":          video_filename,
                    "question":       question_text,
                    "options":        [q.get(f"option_{c}", "") for c in "ABCD"],
                    "question_types": parse_qt(q.get("question_type", "")),
                    "time_reference": (q.get("time_reference") or "").strip() or None,
                    "route":          getattr(agent, "_route_name", ""),
                    "video_length_s": getattr(agent, "_video_length_s", 0),
                    "raw_predicted":  getattr(agent, "_raw_answer", "") or pred,
                    "predicted":      pred_norm,
                    "ground_truth":   gt_answer,
                    "correct":        pred_norm == gt_answer,
                    "tool_calls":     n_calls,
                    "tool_call_log":  tool_log,
                    "vkg_search_log": [list(s) for s in search_history],
                    "final_reasoning": last_reasoning,
                    "transcript":     transcript_path,
                    "mode":           "gvd_lite",
                }

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                all_results.append(record)
                answered_uids.add(uid)

                # ----- Human-readable verbose log per question ----- #
                status = "✓" if record["correct"] else "✗"
                vlog_f.write(
                    f"\n=== {status} uid={uid} video={video_filename} "
                    f"route={record['route']} "
                    f"pred={record['predicted']} gt={record['ground_truth']} "
                    f"tools={record['tool_calls']} ===\n"
                )
                vlog_f.write(f"Q: {record['question']}\n")
                if record["options"]:
                    vlog_f.write(
                        "  (A) " + record["options"][0] + "\n"
                        "  (B) " + record["options"][1] + "\n"
                        "  (C) " + record["options"][2] + "\n"
                        "  (D) " + record["options"][3] + "\n"
                    )
                vlog_f.write(
                    f"time_ref={record['time_reference']} "
                    f"video_len={record['video_length_s']}s "
                    f"qtypes={record['question_types']}\n"
                )
                vlog_f.write(
                    f"raw_pred={record['raw_predicted']!r} "
                    f"final_pred={record['predicted']!r}\n"
                )
                vlog_f.write(f"transcript: {record['transcript']}\n")
                for i, t in enumerate(tool_log, 1):
                    arg_repr = json.dumps(t.get("args", {}), default=str)[:300]
                    res_prev = (t.get("result", "") or "")[:600]
                    vlog_f.write(
                        f"  [{i}] {t['source']} {t['tool']}({arg_repr})\n"
                        f"      -> {res_prev}"
                        + ("\n      …[truncated]" if len(t.get("result", "") or "") > 600 else "")
                        + "\n"
                    )
                if record["final_reasoning"]:
                    vlog_f.write(
                        "final_reasoning: " + record["final_reasoning"][:1500]
                        + ("\n…[truncated]" if len(record["final_reasoning"]) > 1500 else "")
                        + "\n"
                    )
                vlog_f.flush()
                print(f"  {status} uid={record['uid']} pred={record['predicted']} "
                      f"gt={record['ground_truth']} tools={record['tool_calls']}")

                if wb is not None:
                    wb_total += 1
                    wb_correct += int(record["correct"])
                    wb.log({
                        "eval/total": wb_total,
                        "eval/correct": wb_correct,
                        "eval/accuracy": wb_correct / wb_total,
                        "eval/tool_calls": n_calls,
                    })

                total_done += 1
                if args.limit and total_done >= args.limit:
                    break
            if args.limit and total_done >= args.limit:
                break

    print_accuracy(all_results)


if __name__ == "__main__":
    main()
