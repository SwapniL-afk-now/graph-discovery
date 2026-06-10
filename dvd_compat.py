"""DVD's original three tools, re-implemented over the VKG + raw video.

Same names, same calling shape, same observation style as the Deep Video
Discovery paper, so the agent keeps the paper's progressive-granularity
behavior (global → clip → frame) — but backed by the knowledge graph instead
of NanoVectorDB, and by any OpenAI-compatible VLM instead of Azure GPT-4o.

NOTE: no `from __future__ import annotations` — tool signatures must keep
real Annotated annotations for DVD's schema generator.
"""

import json
from typing import Annotated as A
from typing import List, Optional

from .llm import LLMClient
from .timeutil import fmt, to_seconds
from .vkg_tools import D, VKGToolkit

MAX_INSPECT_FRAMES = 24


class DVDCompatTools:
    def __init__(self, toolkit: VKGToolkit, llm: LLMClient,
                 video_path: Optional[str] = None):
        self.toolkit = toolkit
        self.llm = llm
        self.video_path = video_path

    # ------------------------------------------------------------------ #

    def clip_search_tool(
        self,
        event_description: A[str, D("A textual description of the event to search for.")],
        top_k: A[int, D("The maximum number of top results to retrieve. Just use the default value.")] = 16,
    ) -> str:
        """
        Searches the video for clips matching an event description and returns the
        top-k most relevant clip/scene captions in chronological order (classic DVD
        flat retrieval). For typed access with traversable node ids, prefer vkg_search.
        """
        tk = self.toolkit
        if tk.faiss_index is not None and tk.text_encoder is not None:
            hits = tk._semantic_search(event_description, top_k * 3)
        else:
            hits = tk._lexical_search(event_description, top_k * 3)
        clips = [n for n in hits
                 if n.node_type in ("ClipNode", "SceneNode")][:top_k]
        if not clips:
            clips = hits[:top_k]
        clips.sort(key=lambda n: n.t_start)
        lines = [f"[{fmt(n.t_start)} - {fmt(n.t_end)}] {n.label} (node {n.id})"
                 for n in clips]
        return ("Here is the searched video clip scripts:\n\n" + "\n".join(lines)
                + "\n\n(Tip: pass any node id above to vkg_traverse / vkg_causal "
                  "to follow its edges, or frame_inspect_tool on its time range.)")

    def global_browse_tool(
        self,
        query: A[str, D("A textual description or question used to browse the whole video for relevant content.")],
    ) -> str:
        """
        Analyzes the whole video to answer a broad question: retrieves the most
        relevant clip captions plus the graph's character registry, then synthesizes
        a global answer. For raw structure without synthesis, use vkg_overview.
        """
        tk = self.toolkit
        if tk.faiss_index is not None and tk.text_encoder is not None:
            hits = tk._semantic_search(query, 60)
        else:
            hits = tk._lexical_search(query, 60)
        hits.sort(key=lambda n: n.t_start)
        captions = "\n".join(
            f"[{fmt(n.t_start)}] {n.label}" for n in hits)

        registry = {}
        for c in tk.graph.get_all_character_mentions():
            key = c.entity_id or c.id
            registry.setdefault(key, c.canonical_description or c.label)

        answer = self.llm.complete([
            {"role": "system",
             "content": "You are a knowledgeable assistant specializing in analyzing video content and providing detailed, insightful answers."},
            {"role": "user",
             "content": ("Below are descriptions of video clips with timestamps. "
                         "Carefully review the sequence of events, object details and movements, "
                         "and people's actions and poses, then answer the question, "
                         "referencing key events and timestamps.\n"
                         f"Question: {query}\n\n{captions}")},
        ])
        return json.dumps({"subject_registry": registry,
                           "query_related_event": answer})

    def frame_inspect_tool(
        self,
        time_ranges: A[List[List[str]], D("List of [start, end] time ranges to inspect, each as 'HH:MM:SS' strings, e.g. [[\"00:14:10\", \"00:14:25\"]]. Keep ranges short (<60s each).")],
        question: A[str, D("The specific visual question to answer from the frames, e.g. 'What color is the car?'")],
    ) -> str:
        """
        Extracts real frames from the raw video in the given time ranges and asks a
        vision model the question — the ground-truth visual check. Use it to verify
        details the graph cannot capture (colors, counts, poses, on-screen text) and
        to CONFIRM answers before finishing. Localize with graph tools first.
        """
        if not self.video_path:
            return ("No raw video file is configured for this session, so frames "
                    "cannot be inspected. Rely on graph evidence: vkg_window over "
                    "the same time ranges gives the densest available detail.")
        from qvkg.query.frame_extractor import (extract_frames_for_window,
                                                frames_to_b64_urls)

        budget = max(2, MAX_INSPECT_FRAMES // max(1, len(time_ranges)))
        urls: List[str] = []
        spans: List[str] = []
        for rng in time_ranges:
            t0, t1 = to_seconds(rng[0]), to_seconds(rng[1])
            if t1 <= t0:
                t1 = t0 + 10.0
            frames = extract_frames_for_window(
                self.video_path, t0, t1, max_frames=budget)
            urls.extend(frames_to_b64_urls(frames))
            spans.append(f"{fmt(t0)}–{fmt(t1)}")
        if not urls:
            return "Could not decode any frames in those ranges — check the timestamps against the video length."

        # Give the VLM the graph's notes for the same windows as context.
        ctx_parts = []
        for rng in time_ranges[:3]:
            t0, t1 = to_seconds(rng[0]), to_seconds(rng[1])
            nodes = self.toolkit.graph.get_nodes_in_window(t0, t1, buffer_sec=0.0)
            labels = [f"[{fmt(n.t_start)}] {n.label}" for n in
                      sorted(nodes, key=lambda n: n.t_start)[:10]]
            if labels:
                ctx_parts.append("\n".join(labels))
        context = ("Context notes from the video's knowledge graph for these "
                   "windows (may be incomplete):\n" + "\n".join(ctx_parts)) if ctx_parts else ""

        answer = self.llm.complete_with_images(
            question=f"Frames are from time range(s): {', '.join(spans)}.\nQuestion: {question}",
            image_urls=urls[:MAX_INSPECT_FRAMES],
            context=context,
        )
        return f"Frame inspection over {', '.join(spans)} ({min(len(urls), MAX_INSPECT_FRAMES)} frames):\n{answer}"

    def tools(self):
        return [self.global_browse_tool, self.clip_search_tool,
                self.frame_inspect_tool]
