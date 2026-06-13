#!/usr/bin/env python3
"""Persistent-engine campaign driver: load SigLIP + the vLLM engine ONCE and
keep them resident on the GPU across videos — build each missing VKG and run
the eval in the same process, instead of the per-video load/unload cycle of
run_full_eval.sh.

Usage:
    python3 run_persistent.py [N_VIDEOS]   # default 2 (test run)
"""

import csv
import json
import os
import sys
import types

sys.path.insert(0, "/workspace")
import gvd  # noqa: F401  (path setup for qvkg sibling)

CSV       = "/workspace/data/LVBench_full.csv"
VKG_DIR   = "/workspace/vkgs"
VIDEO_DIR = "/workspace/videos"
MODEL     = "/workspace/models/Qwen3.5-4B"
RESULTS   = os.environ.get("GVD_RESULTS", "/workspace/results_full_lvbench.jsonl")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def next_unanswered_videos(n):
    order, seen = [], set()
    for r in csv.DictReader(open(CSV)):
        v = r["video_path"].rsplit(".", 1)[0]
        if v not in seen:
            seen.add(v)
            order.append(v)
    answered = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            if line.strip():
                answered.add(json.loads(line)["video"].rsplit(".", 1)[0])
    return [v for v in order if v not in answered][:n]


# Reuse the exact loaders from scripts/build_vkg.py (the sampler expects
# parse_time_reference's (start, end) tuples, not raw strings).
sys.path.insert(0, "/workspace/video-reasoning/qvkg/scripts")
from build_vkg import load_question_time_refs, load_question_vocab  # noqa: E402


def main():
    targets = next_unanswered_videos(N)
    if not targets:
        print("No unanswered videos left.")
        return
    print(f"[persistent] target videos: {targets}")

    from qvkg.vllm_client import build_llm, build_siglip_encoder

    print("[persistent] loading SigLIP once...")
    siglip = build_siglip_encoder()
    print("[persistent] building vLLM engine once...")
    engine = build_llm(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.65,
        max_model_len=65536,
        lazy=True,
    )
    # vLLM's EngineCore worker re-reads sys.argv on spawn — it must initialize
    # while argv is clean, NOT lazily after we've swapped in eval flags.
    sys.argv = ["run_persistent.py"]
    engine._ensure()
    print("[persistent] engine resident on GPU")

    # --- share the engine/encoder with everything downstream -------------
    import gvd.vllm_llm as vllm_llm_mod
    _OrigClient = vllm_llm_mod.VLLMToolClient
    shared_engine = engine

    class SharedEngineClient(_OrigClient):
        def __init__(self, *, engine=None, **kw):
            if engine is None:
                # drop engine-construction kwargs; reuse the resident engine
                for k in ("model", "tensor_parallel_size",
                          "gpu_memory_utilization", "max_model_len", "lazy"):
                    kw.pop(k, None)
                engine = shared_engine
            super().__init__(engine=engine, **kw)

    vllm_llm_mod.VLLMToolClient = SharedEngineClient

    import qvkg.vllm_client as qvc
    qvc.build_siglip_encoder = lambda *a, **kw: siglip

    # --- build + eval loop, one resident engine ---------------------------
    import gvd.eval_lvbench as ev
    from qvkg.builder import VKGBuilder

    whisper_ns = types.SimpleNamespace(_model_size="large-v3-turbo",
                                       _compute_type="int8_float16")

    for vid in targets:
        vkg_json = os.path.join(VKG_DIR, vid, "vkg.json")
        if not os.path.exists(vkg_json):
            video_filename = vid + ".mp4"
            out_dir = os.path.join(VKG_DIR, vid)
            os.makedirs(out_dir, exist_ok=True)
            print(f"[persistent] building VKG for {vid}")
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
            builder = VKGBuilder(engine, whisper_ns, siglip, config)
            graph = builder.build(os.path.join(VIDEO_DIR, video_filename),
                                  out_dir, phase="all")
            if graph is None or not os.path.exists(vkg_json):
                print(f"[persistent] build FAILED for {vid} — skipping eval")
                continue

        print(f"[persistent] evaluating {vid} (engine stays loaded)")
        argv_save = sys.argv
        sys.argv = [
            "eval_lvbench",
            "--csv", CSV, "--vkg-dir", VKG_DIR, "--video-dir", VIDEO_DIR,
            "--out", RESULTS, "--model", MODEL,
            "--max-iterations", "10", f"--video={vid}",
            "--wandb-project", "gvd-lvbench",
        ]
        try:
            ev.main()
        finally:
            sys.argv = argv_save

    print("[persistent] DONE — all target videos processed with one engine load.")


if __name__ == "__main__":
    main()
