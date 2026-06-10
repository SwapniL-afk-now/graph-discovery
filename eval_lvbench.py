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
                   help="Skip SigLIP — search_events falls back to lexical mode")
    p.add_argument("--no-detector", action="store_true",
                   help="Disable the open-vocabulary count_objects")
    p.add_argument("--video",     default=None,
                   help="Restrict evaluation to one video id (e.g. for run_rolling)")
    p.add_argument("--votes",     type=int, default=1,
                   help="Self-consistency: run k sampled trajectories per question "
                        "(batched through vLLM together) and majority-vote the letter")
    p.add_argument("--vote-temperature", type=float, default=0.7,
                   help="Action-sampling temperature for voting trajectories")
    p.add_argument("--vote-seed", type=int, default=None,
                   help="Base seed for voting trajectories (vote i uses seed+i); "
                        "omit for non-reproducible sampling")
    p.add_argument("--uids", default=None,
                   help="Comma-separated uid list to (re-)run, ignoring resume state")
    p.add_argument("--wandb-project", default=None,
                   help="Report per-question results online to this W&B project. "
                        "Set WANDB_RUN_ID in the env to make sequential per-video "
                        "processes append to one run (resume='allow').")
    p.add_argument("--parallel", type=int, default=1,
                   help="Answer up to N questions of a video concurrently, their "
                        "per-step LLM calls batched through vLLM together "
                        "(single greedy trajectory per question; incompatible "
                        "with --votes > 1)")
    args = p.parse_args()
    if args.votes > 1 and args.parallel > 1:
        p.error("--votes and --parallel cannot be combined")

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

    # Voting: k clients share ONE engine through a micro-batcher, so the k
    # trajectories' per-step calls decode as a single vLLM batch.
    batched = None
    vote_clients = []
    if args.votes > 1 or args.parallel > 1:
        from gvd.batch_engine import BatchedEngine
        batched = BatchedEngine(llm.engine)
    if args.votes > 1:
        vote_clients = [
            VLLMToolClient(
                engine=batched,
                action_temperature=args.vote_temperature,
                sampling_seed=(args.vote_seed + v) if args.vote_seed is not None else None,
            )
            for v in range(args.votes)
        ]

    questions_by_video = load_questions(args.csv)
    answered_uids = load_answered_uids(args.out)

    wb = None
    wb_total = wb_correct = 0
    if args.wandb_project:
        import wandb
        # Seed cumulative counters from previous (resumed) processes so the
        # running-accuracy curve is continuous across per-video invocations.
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
            config={"model": args.model, "max_iterations": args.max_iterations,
                    "votes": args.votes, "parallel": args.parallel,
                    "csv": args.csv, "out": args.out},
        )
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

            if args.uids:
                wanted = {u.strip() for u in args.uids.split(",")}
                pending = [q for q in questions if q["uid"] in wanted]
            else:
                pending = [q for q in questions if q["uid"] not in answered_uids]
            if args.limit:
                pending = pending[: max(0, args.limit - total_done)]
            if not pending:
                print(f"\nVideo: {video_filename} — all questions already answered.")
                continue

            print(f"\nVideo: {video_filename} ({len(pending)} pending / {len(questions)} total)")

            def make_agent(client):
                return GVDAgent(
                    graph_path=vkg_path,
                    video_path=video_path if os.path.exists(video_path) else None,
                    faiss_index_path=index_path if os.path.exists(index_path) else None,
                    text_encoder=siglip,
                    llm=client,
                    max_iterations=args.max_iterations,
                    enable_detector=not args.no_detector,
                )

            # One agent per concurrent trajectory: agent/toolkit state (search
            # history, inferred edges, lazy detector) must not be shared
            # across threads.
            agents = ([make_agent(c) for c in vote_clients]
                      if args.votes > 1 else [make_agent(llm)])

            def build_mcq(q):
                mcq = (
                    q["question"]
                    + "\n\nThis is a multiple-choice question. When you call finish, "
                      "the answer must contain exactly one letter: A, B, C, or D."
                )
                time_ref = (q.get("time_reference") or "").strip() or None
                if time_ref:
                    mcq = f"[Time reference: {time_ref}]\n{mcq}"
                return mcq, time_ref

            transcripts_dir = args.out + ".transcripts"
            os.makedirs(transcripts_dir, exist_ok=True)

            def run_traj(a, mcq, uid=None):
                try:
                    answer, msgs = a.run(mcq)
                    pred = extract_mcq_answer(answer or "") or "ERROR"
                    n_calls = sum(len(m.get("tool_calls") or []) for m in msgs
                                  if m.get("role") == "assistant")
                    if uid is not None:
                        with open(os.path.join(transcripts_dir, f"{uid}.json"), "w") as tf:
                            json.dump({"answer": answer, "messages": msgs}, tf,
                                      indent=1, default=str)
                    return pred, n_calls
                except Exception:
                    import traceback
                    traceback.print_exc()
                    return "ERROR", 0

            def make_record(q, pred, n_calls, outcomes):
                time_ref = (q.get("time_reference") or "").strip() or None
                gt = q["answer"].strip().upper()
                record = {
                    "uid":            q["uid"],
                    "video":          video_filename,
                    "question_types": parse_qt(q.get("question_type", "")),
                    "time_reference": time_ref,
                    "predicted":      pred,
                    "ground_truth":   gt,
                    "correct":        pred == gt,
                    "tool_calls":     n_calls,
                    "mode":           ("gvd_agent" if args.votes == 1
                                       else f"gvd_agent_vote{args.votes}"),
                }
                if args.votes > 1:
                    record["votes"] = [p for p, _ in outcomes]
                    record["vote_tool_calls"] = [n for _, n in outcomes]
                return record

            def emit(record, votes_suffix=""):
                nonlocal wb_total, wb_correct
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                all_results.append(record)
                answered_uids.add(record["uid"])
                status = "✓" if record["correct"] else "✗"
                print(f"  {status} uid={record['uid']} pred={record['predicted']} "
                      f"gt={record['ground_truth']} "
                      f"tools={record['tool_calls']}{votes_suffix}")
                if wb is not None:
                    wb_total += 1
                    wb_correct += bool(record["correct"])
                    wb.log({
                        "question/correct": int(record["correct"]),
                        "question/tool_calls": record["tool_calls"],
                        "question/uid": int(record["uid"]),
                        "cumulative/answered": wb_total,
                        "cumulative/accuracy": wb_correct / wb_total,
                        "video": record["video"],
                        "predicted": record["predicted"],
                        "ground_truth": record["ground_truth"],
                        "question_types": ",".join(record["question_types"]),
                    }, step=wb_total)

            if args.parallel > 1:
                from concurrent.futures import ThreadPoolExecutor
                emit_lock = __import__("threading").Lock()

                def answer_one(q):
                    # Own client + agent per question: trajectories are
                    # independent; only the engine is shared (via the batcher).
                    agent = make_agent(VLLMToolClient(engine=batched))
                    mcq, _ = build_mcq(q)
                    try:
                        pred, n_calls = run_traj(agent, mcq, q["uid"])
                    finally:
                        batched.unregister()
                    record = make_record(q, pred, n_calls, None)
                    with emit_lock:
                        emit(record)
                    return record

                for start in range(0, len(pending), args.parallel):
                    chunk = pending[start:start + args.parallel]
                    # Register all workers BEFORE any thread starts, so the
                    # first arrival can't flush a single-request batch.
                    batched.register(len(chunk))
                    with ThreadPoolExecutor(max_workers=len(chunk)) as ex:
                        list(ex.map(answer_one, chunk))
                    total_done += len(chunk)
                    if args.limit and total_done >= args.limit:
                        break
                if args.limit and total_done >= args.limit:
                    break
                continue

            for q in pending:
                uid       = q["uid"]
                gt_answer = q["answer"].strip().upper()
                mcq_question, time_ref = build_mcq(q)

                def run_single(a):
                    return run_traj(a, mcq_question, uid)

                if args.votes > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    # Register all workers BEFORE any thread starts, so the
                    # first arrival can't flush a single-request batch.
                    batched.register(args.votes)

                    def run_vote(a):
                        try:
                            return run_single(a)
                        finally:
                            batched.unregister()

                    with ThreadPoolExecutor(max_workers=args.votes) as ex:
                        outcomes = list(ex.map(run_vote, agents))
                    # Majority vote; ERROR / zero-tool-call runs abstain so an
                    # ungrounded guess can't propagate past the grounded runs.
                    ballots = [(p, n) for p, n in outcomes
                               if p in "ABCD" and n > 0] or \
                              [(p, n) for p, n in outcomes if p in "ABCD"]
                    if ballots:
                        counts = defaultdict(int)
                        for p, _ in ballots:
                            counts[p] += 1
                        top = max(counts.values())
                        tied = [p for p, c in counts.items() if c == top]
                        # Tie-break: the letter backed by the most tool calls
                        # (i.e. the most-grounded trajectories).
                        pred = max(tied, key=lambda l: sum(
                            n for p, n in ballots if p == l))
                    else:
                        pred = "ERROR"
                    n_calls = sum(n for _, n in outcomes)
                else:
                    pred, n_calls = run_single(agents[0])
                    outcomes = [(pred, n_calls)]

                record = make_record(q, pred, n_calls, outcomes)
                votes_str = (" votes=" + "".join(p for p, _ in outcomes)
                             if args.votes > 1 else "")
                emit(record, votes_str)
                total_done += 1

                if args.limit and total_done >= args.limit:
                    break
            if args.limit and total_done >= args.limit:
                break

    print_accuracy(all_results)
    if wb is not None:
        wb.summary["overall/answered"] = wb_total
        wb.summary["overall/accuracy"] = (wb_correct / wb_total) if wb_total else 0.0
        wb.finish()


if __name__ == "__main__":
    main()
