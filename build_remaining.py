#!/usr/bin/env python3
"""Build-only driver: construct every missing VKG with one resident engine.
No question answering happens here — evals run separately via the API model.

Usage:
    python3 build_remaining.py [N_VIDEOS]   # default: all missing
"""

import csv
import os
import sys
import types

sys.path.insert(0, "/workspace")
import gvd  # noqa: F401  (path setup for qvkg sibling)

CSV       = "/workspace/data/LVBench_full.csv"
VKG_DIR   = "/workspace/vkgs"
VIDEO_DIR = "/workspace/videos"
MODEL     = "/workspace/models/Qwen3.5-4B"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 999

sys.path.insert(0, "/workspace/video-reasoning/qvkg/scripts")
from build_vkg import load_question_time_refs, load_question_vocab  # noqa: E402


def missing_videos(n):
    order, seen = [], set()
    for r in csv.DictReader(open(CSV)):
        v = r["video_path"].rsplit(".", 1)[0]
        if v not in seen:
            seen.add(v)
            order.append(v)
    return [v for v in order
            if not os.path.exists(os.path.join(VKG_DIR, v, "vkg.json"))][:n]


def main():
    targets = missing_videos(N)
    if not targets:
        print("[builds] no missing graphs.")
        return
    print(f"[builds] {len(targets)} graphs to build: {targets}")

    from qvkg.vllm_client import build_llm, build_siglip_encoder

    print("[builds] loading SigLIP once...")
    siglip = build_siglip_encoder()
    print("[builds] building vLLM engine once...")
    engine = build_llm(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.60,
        max_model_len=65536,
        lazy=True,
    )
    sys.argv = ["build_remaining.py"]   # EngineCore re-reads argv on spawn
    engine._ensure()
    print("[builds] engine resident on GPU")

    from qvkg.builder import VKGBuilder

    whisper_ns = types.SimpleNamespace(_model_size="large-v3-turbo",
                                       _compute_type="int8_float16")

    ok = failed = 0
    for vid in targets:
        video_filename = vid + ".mp4"
        out_dir = os.path.join(VKG_DIR, vid)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[builds] building VKG for {vid} ({ok + failed + 1}/{len(targets)})")
        config = {
            "frame_budget":          500,
            "semantic_threshold":    0.78,
            "semantic_k_neighbors":  10,
            "causal_min_confidence": 0.6,
            "hard_boundary_thresh":  0.75,
            "soft_boundary_thresh":  0.5,
            "question_time_refs":    load_question_time_refs(CSV, video_filename),
            "asr_vocab":             load_question_vocab(CSV, video_filename),
            "video_type":            None,
            "subtitle_path":         None,
            "coarse_fps":            1.0,
            "coarse_frame_cap":      0,
            "flow_max_dim":          256,
            "use_optical_flow":      True,
            "whisper_compute_type":  "int8_float16",
        }
        try:
            builder = VKGBuilder(engine, whisper_ns, siglip, config)
            graph = builder.build(os.path.join(VIDEO_DIR, video_filename),
                                  out_dir, phase="all")
        except Exception as exc:
            graph = None
            print(f"[builds] build EXCEPTION for {vid}: {exc}")
        if graph is None or not os.path.exists(os.path.join(out_dir, "vkg.json")):
            failed += 1
            print(f"[builds] build FAILED for {vid}")
        else:
            ok += 1
            print(f"[builds] {vid} done")

    print(f"[builds] DONE — {ok} built, {failed} failed.")


if __name__ == "__main__":
    main()
