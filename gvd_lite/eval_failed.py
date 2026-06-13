#!/usr/bin/env python3
"""Run gvd_lite ONLY on the uids that the baseline (results_full_lvbench.jsonl)
got wrong. Same output format as the regular eval so we can compare.

Usage:
    python3 -m gvd_lite.eval_failed --limit 20
"""
import argparse, ast, csv, json, os, sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gvd_lite  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",       default="/workspace/data/LVBench_full.csv")
    p.add_argument("--vkg-dir",   default="/workspace/vkgs")
    p.add_argument("--video-dir", default="/workspace/videos")
    p.add_argument("--baseline",  default="/workspace/results_full_lvbench.jsonl",
                   help="JSONL of previous run; we re-run the UIDs it got wrong")
    p.add_argument("--out",       default="/workspace/results_gvd_lite_failed.jsonl")
    p.add_argument("--model",     default="/workspace/models/Qwen3.5-4B")
    p.add_argument("--tp",        type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    p.add_argument("--max-model-len",  type=int, default=65536)
    p.add_argument("--max-react-turns", type=int, default=4)
    p.add_argument("--limit",     type=int, default=None)
    p.add_argument("--video",     default=None, help="Restrict to one video id")
    p.add_argument("--only-qt",   default=None,
                   help="Restrict to one question type (e.g. 'reasoning')")
    p.add_argument("--no-siglip", action="store_true")
    args = p.parse_args()

    # Load baseline, collect failed UIDs
    failed_by_uid = {}
    with open(args.baseline) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("correct") or r.get("predicted") == "ERROR":
                failed_by_uid[r["uid"]] = r
    print(f"Failed UIDs in baseline: {len(failed_by_uid)}")

    if args.only_qt:
        failed_by_uid = {u: r for u, r in failed_by_uid.items()
                          if args.only_qt in (r.get("question_types") or [])}
        print(f"  after --only-qt={args.only_qt}: {len(failed_by_uid)}")

    if args.video:
        failed_by_uid = {u: r for u, r in failed_by_uid.items()
                          if r["video"].replace(".mp4", "") == args.video}
        print(f"  after --video={args.video}: {len(failed_by_uid)}")

    # Load CSV, group by video, filter
    questions_by_video = defaultdict(list)
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["uid"] in failed_by_uid:
                questions_by_video[row["video_path"]].append(row)
    total_q = sum(len(v) for v in questions_by_video.values())
    print(f"To re-run: {total_q} questions across {len(questions_by_video)} videos")

    # Resume: skip already answered
    answered = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    answered.add(json.loads(line)["uid"])
    print(f"Already in --out: {len(answered)} (will skip)")

    # LLM
    if not args.no_siglip:
        from qvkg.vllm_client import build_siglip_encoder
        print("Loading SigLIP encoder...")
        siglip = build_siglip_encoder()
    else:
        siglip = None
    from gvd.vllm_llm import VLLMToolClient
    llm = VLLMToolClient(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        lazy=True,
    )
    from qvkg.vllm_client import extract_mcq_answer

    from gvd_lite.agent import GVDLiteAgent
    from gvd_lite import router

    def parse_qt(raw):
        try:
            v = ast.literal_eval(raw)
            return list(v) if isinstance(v, (list, tuple)) else [str(v)]
        except Exception:
            return [raw] if raw else []

    all_results = []
    total_done = 0
    transcripts_dir = args.out + ".transcripts"
    os.makedirs(transcripts_dir, exist_ok=True)

    with open(args.out, "a") as out_f:
        for video_filename, questions in questions_by_video.items():
            video_id = os.path.splitext(video_filename)[0]
            if args.video and video_id != args.video:
                continue
            vkg_path   = os.path.join(args.vkg_dir, video_id, "vkg.json")
            index_path = os.path.join(args.vkg_dir, video_id, "vkg.index")
            video_path = os.path.join(args.video_dir, video_filename)
            if not os.path.exists(vkg_path):
                print(f"  [SKIP] no VKG for {video_filename}")
                continue
            pending = [q for q in questions if q["uid"] not in answered]
            if args.limit:
                pending = pending[: max(0, args.limit - total_done)]
            if not pending:
                continue
            print(f"\nVideo: {video_filename} ({len(pending)} failed-q)")

            agent = GVDLiteAgent(
                graph_path=vkg_path,
                video_path=video_path if os.path.exists(video_path) else None,
                faiss_index_path=index_path if os.path.exists(index_path) else None,
                text_encoder=siglip,
                llm=llm,
                max_react_turns=args.max_react_turns,
                enable_detector=False,
            )

            for q in pending:
                uid = q["uid"]
                gt_answer = q["answer"].strip().upper()
                qts = parse_qt(q.get("question_type", ""))
                time_ref = (q.get("time_reference") or "").strip() or None
                mcq = q["question"]
                if time_ref:
                    mcq = f"[Time reference: {time_ref}]\n{mcq}"
                mcq += ("\n\nThis is a multiple-choice question. When you call answer, "
                        "the answer must contain exactly one letter: A, B, C, or D.")
                route = router.route_name(mcq)

                try:
                    pred, msgs = agent.run(mcq)
                    pred_norm = extract_mcq_answer(pred or "") or pred or "ERROR"
                    if pred_norm not in "ABCD":
                        pred_norm = "ERROR"
                    n_calls = sum(len(m.get("tool_calls") or []) for m in msgs
                                  if m.get("role") == "assistant")
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    pred_norm = "ERROR"
                    n_calls = 0
                    msgs = []

                record = {
                    "uid":            uid,
                    "video":          video_filename,
                    "question_types": qts,
                    "time_reference": time_ref,
                    "predicted":      pred_norm,
                    "ground_truth":   gt_answer,
                    "correct":        pred_norm == gt_answer,
                    "tool_calls":     n_calls,
                    "mode":           "gvd_lite",
                    "route":          route,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                try:
                    with open(os.path.join(transcripts_dir, f"{uid}.json"), "w") as tf:
                        json.dump(msgs, tf, default=str)
                except Exception:
                    pass
                all_results.append(record)
                answered.add(uid)
                status = "✓" if record["correct"] else "✗"
                flip = " (was wrong, now right)" if record["correct"] else " (still wrong)"
                print(f"  {status} uid={uid} route={route:14s} "
                      f"pred={pred_norm} gt={gt_answer} tools={n_calls}{flip}")
                total_done += 1
                if args.limit and total_done >= args.limit:
                    break
            if args.limit and total_done >= args.limit:
                break

    if all_results:
        c = sum(1 for r in all_results if r["correct"])
        print(f"\n=== gvd_lite on baseline-failed questions ===")
        print(f"Recovered: {c}/{len(all_results)} = {100*c/len(all_results):.1f}%")


if __name__ == "__main__":
    main()
