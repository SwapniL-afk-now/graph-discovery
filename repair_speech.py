#!/usr/bin/env python3
"""Repair speech-less VKGs without a full rebuild.

For each video id given on the command line:
  1. transcribe its `.audio.*` sidecar (downloaded separately; the original
     .mp4s have no audio stream) with the fixed Whisper worker,
  2. build SpeechNodes exactly like VKGBuilder._build_speech_nodes,
  3. merge them into the existing vkg.json,
  4. rebuild the FAISS index so search_events can retrieve them,
  5. refresh speech_nodes.pkl and remove the WHISPER_FAILED marker.

Usage:
    python3 repair_speech.py VTCDQYYKA9o T1yhBv1ytzw ...
"""

import glob
import os
import pickle
import sys

sys.path.insert(0, "/workspace")
import gvd  # noqa: F401
sys.path.insert(0, "/workspace/video-reasoning/qvkg")

from qvkg.sampler import _whisper_transcribe_worker
from qvkg.schema import VKGraph, VKGNode
from qvkg.faiss_index import build_faiss_index
from qvkg.vllm_client import build_siglip_encoder

VKG_DIR = "/workspace/vkgs"
VIDEO_DIR = "/workspace/videos"

siglip = None  # loaded lazily, once


def build_speech_nodes(segments):
    nodes = []
    for seg in segments:
        text = getattr(seg, "text", "").strip()
        if not text:
            continue
        nodes.append(VKGNode(
            id=f"speech_{len(nodes):05d}",
            node_type="SpeechNode",
            label=text,
            level=0,
            t_start=float(getattr(seg, "start", 0)),
            t_end=float(getattr(seg, "end", 0)),
            confidence=float(getattr(seg, "avg_logprob", 0.8) + 1.0),
            metadata={"source": "whisper"},
        ))
    return nodes


def repair(vid):
    global siglip
    out_dir = os.path.join(VKG_DIR, vid)
    vkg_path = os.path.join(out_dir, "vkg.json")
    if not os.path.exists(vkg_path):
        print(f"[repair] {vid}: no vkg.json — build it first, skipping")
        return
    audio = sorted(glob.glob(os.path.join(VIDEO_DIR, f"{vid}.audio.*")))
    src = audio[0] if audio else os.path.join(VIDEO_DIR, f"{vid}.mp4")
    print(f"[repair] {vid}: transcribing {os.path.basename(src)}")
    segments = _whisper_transcribe_worker(src, "large-v3-turbo", "int8_float16")
    nodes = build_speech_nodes(segments)
    print(f"[repair] {vid}: {len(nodes)} speech segments")
    if not nodes:
        print(f"[repair] {vid}: nothing to merge, leaving graph untouched")
        return

    graph = VKGraph.load(vkg_path)
    existing = {n.id for n in graph.nodes.values()}
    added = 0
    for n in nodes:
        if n.id in existing:  # stale 0/1-segment runs used the same ids
            graph.nodes.pop(n.id, None)
        graph.add_node(n)
        added += 1
    graph.save(vkg_path)
    with open(os.path.join(out_dir, "speech_nodes.pkl"), "wb") as f:
        pickle.dump(nodes, f)
    print(f"[repair] {vid}: merged {added} SpeechNodes "
          f"({len(graph.nodes)} total), rebuilding index")

    if siglip is None:
        siglip = build_siglip_encoder()
    index_path = os.path.join(out_dir, "vkg.index")
    import faiss
    index = build_faiss_index(graph, siglip, index_path)
    faiss.write_index(index, index_path)

    marker = os.path.join(out_dir, "WHISPER_FAILED")
    if os.path.exists(marker):
        os.remove(marker)
    print(f"[repair] {vid}: done")


if __name__ == "__main__":
    for vid in sys.argv[1:]:
        repair(vid)
