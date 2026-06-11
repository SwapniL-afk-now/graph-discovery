"""DVD's original three tools, re-implemented over the VKG + raw video.

Same names, same calling shape, same observation style as the Deep Video
Discovery paper, so the agent keeps the paper's progressive-granularity
behavior (global → clip → frame) — but backed by the knowledge graph instead
of NanoVectorDB, and by any OpenAI-compatible VLM instead of Azure GPT-4o.

NOTE: no `from __future__ import annotations` — tool signatures must keep
real Annotated annotations for DVD's schema generator.
"""

import json
import re
from typing import Annotated as A
from typing import List, Optional, Tuple

from .llm import LLMClient
from .timeutil import fmt, to_seconds
from .vkg_tools import D, VKGToolkit

# The vision engine accepts at most ~10 images per prompt, so density comes
# from running MULTIPLE passes of this many frames each, not one giant prompt.
FRAMES_PER_PASS = 8
MAX_PASSES = 5                 # cap total VLM vision calls per inspection
_COUNT_RE = re.compile(r"\b(how many|how often|number of|count|times|how much)\b", re.I)
# Sub-second actions (a stroke, a shot, a fall) vanish between ~1fps samples;
# these route to the chunked high-fps walk with a sequence-preserving fuse.
_FAST_ACTION_RE = re.compile(
    r"\b(shot|stroke|smash|serve|rally|hit[s]?\b|kick|throw|catch|swing|punch|"
    r"goal|score[ds]?\b|(point|game|match) (was |is )?(won|lost)|"
    r"w[io]ns? the (point|game|match)|ball|jump|fall[s]?\b|trip|"
    r"slam|spike|dunk|volley|backhand|forehand|lob|replay|"
    r"turn(s|ed)? (around|back|first)|in what order|which .{0,30}first)\b", re.I)


class DVDCompatTools:
    def __init__(self, toolkit: VKGToolkit, llm: LLMClient,
                 video_path: Optional[str] = None, detector=None,
                 enable_detector: bool = True):
        self.toolkit = toolkit
        self.llm = llm
        self.video_path = video_path
        self._detector = detector          # lazily built on first object_count call
        self._enable_detector = enable_detector
        # The full MCQ being answered (set by the agent per run). The VLM maps
        # visual evidence onto the options far better than the small
        # orchestrator maps a free-text description onto them afterwards.
        self.current_mcq: Optional[str] = None



    # ------------------------------------------------------------------ #
    # Adaptive frame extraction
    # ------------------------------------------------------------------ #

    def _grab_frames(self, t0: float, t1: float, density: str) -> List[Tuple[float, str]]:
        """Return [(timestamp, b64_url)] for one window at the chosen density.

        density 'dense'  → many frames, motion-blind (see every instant: counts,
                           rapid actions like knocks/strikes).
        density 'event'  → motion-ranked keyframes (the moments where things
                           change — best for "what happens" over a longer span).
        density 'detail' → a handful of evenly-spaced frames (one moment, a pose,
                           on-screen text, a color).
        """
        from qvkg.query.frame_extractor import (extract_frames_for_window,
                                                frames_to_b64_urls)
        span = max(0.0, t1 - t0)
        if density == "dense":
            cap = min(FRAMES_PER_PASS * MAX_PASSES, max(8, int(span * 10)))
            frames = extract_frames_for_window(self.video_path, t0, t1, max_frames=cap)
        elif density == "event":
            cap = FRAMES_PER_PASS * min(MAX_PASSES, 3)
            frames = extract_frames_for_window(self.video_path, t0, t1,
                                               max_frames=cap, motion_rank=span > 20)
        else:  # detail
            frames = extract_frames_for_window(self.video_path, t0, t1, max_frames=FRAMES_PER_PASS)
        frames = [f for f in frames if f.image is not None]
        urls = frames_to_b64_urls(frames)
        return list(zip((f.timestamp for f in frames), urls))

    def _window_multimodal_context(self, time_ranges) -> str:
        """Dialogue (speech), on-screen text (OCR) and action notes the graph
        holds for these windows — folded into every inspection so visual,
        spoken and textual evidence are fused, not siloed."""
        g = self.toolkit.graph
        speech, ocr, action = [], [], []
        for rng in time_ranges:
            t0, t1 = to_seconds(rng[0]), to_seconds(rng[1])
            for n in sorted(g.get_nodes_in_window(t0, t1, buffer_sec=2.0),
                            key=lambda n: n.t_start):
                line = f"[{fmt(n.t_start)}] {n.label}"
                if n.node_type == "SpeechNode":
                    speech.append(line)
                elif n.node_type == "OCRNode":
                    ocr.append(line)
                elif n.node_type in ("ActionNode", "InteractionNode", "StateChangeNode"):
                    action.append(line)
        parts = []
        if speech:
            parts.append("DIALOGUE spoken in this window (transcript):\n" + "\n".join(speech[:15]))
        if ocr:
            parts.append("ON-SCREEN TEXT (OCR) in this window:\n" + "\n".join(ocr[:10]))
        if action:
            parts.append("Action notes from the graph:\n" + "\n".join(action[:10]))
        return "\n\n".join(parts)

    def inspect_frames(
        self,
        time_ranges: A[List[List[str]], D("List of [start, end] time ranges to inspect, each as 'HH:MM:SS' strings, e.g. [[\"00:14:10\", \"00:14:25\"]].")],
        question: A[str, D("The specific visual question to answer from the frames, e.g. 'What color is the car?'")],
        sampling: A[str, D("Frame selection: 'auto' (recommended), 'dense' (slow-motion chunked sweep — COUNTING repeated events AND the exact ORDER of fast actions: who moved first, sequence of strokes; window must be under ~60s), 'event' (what unfolds over a longer span — motion keyframes), or 'detail' (one moment: pose, color, sign).")] = "auto",
    ) -> str:
        """
        Inspect the raw video: selects up to 10 real frames at an adaptive density,
        tags each with its timestamp, and asks the vision model the question in ONE
        pass — with the window's dialogue (speech) and on-screen text (OCR) from the
        graph folded into the same prompt.
        USE THIS FOR:
          • "how many times did X knock/hit/strike" → sampling="dense"
          • "who moved/turned FIRST", "in what ORDER did the fast actions happen" → sampling="dense"
          • "what does X see through the window" → sampling="auto"
          • "what color is X", "what is X wearing" → sampling="detail"
          • "what happens during X" → sampling="event"
        For COUNTING how many people/objects are PRESENT (not events over time),
        use count_objects instead — a real detector counts better than the eye.
        """
        if not self.video_path:
            return ("No raw video file is configured for this session, so frames "
                    "cannot be inspected. Rely on graph evidence: read_moment over "
                    "the same time ranges gives the densest available detail.")

        # Adaptive density, but always a SINGLE 10-frame pass — splitting an
        # event across batches and re-fusing the summaries loses coherence and
        # measurably hurt accuracy, so the model sees the moment whole.
        total_span = sum(max(0.0, to_seconds(r[1]) - to_seconds(r[0])) for r in time_ranges)
        if sampling == "auto":
            density = "event" if total_span > 45 else "detail"
        else:
            density = sampling if sampling in ("dense", "event", "detail") else "detail"

        # COUNTING needs to see every instant: one 10-frame pass over a long
        # window misses repeated events (4 knocks in 40s ≈ 0.25 fps saw 1).
        # Dense mode therefore walks the window in consecutive chunks, counts
        # per chunk, and fuses the per-chunk reports.
        # Only genuine counting questions take the multi-pass path: the model
        # also picks "dense" for why/what questions, and a count-oriented fuse
        # over speculative segment reports is poison there.
        counting = bool(_COUNT_RE.search(question)
                        or (self.current_mcq and _COUNT_RE.search(self.current_mcq)))
        # Fast actions (a stroke, a shot, where the ball lands) resolve in
        # well under a second — a single 10-frame pass over a >5s window
        # samples below 2fps and the VLM narrates a guess. Walk such windows
        # chunk-by-chunk instead: per-chunk counts fused into a total for
        # counting questions, a sequence-preserving narration otherwise.
        # The model choosing sampling="dense" is honored directly — the regex
        # only rescues fast-action questions where it picked "auto".
        fast_action = bool(_FAST_ACTION_RE.search(question)
                           or (self.current_mcq
                               and _FAST_ACTION_RE.search(self.current_mcq)))
        if len(time_ranges) == 1 and (density == "dense"
                                      or (fast_action and density != "event")):
            t0, t1 = to_seconds(time_ranges[0][0]), to_seconds(time_ranges[0][1])
            if t1 <= t0:
                t1 = t0 + 10.0
            span = t1 - t0
            if 4.0 < span and (counting or span <= 60.0):
                mode = "count" if counting else "action"
                return self._dense_count_inspect(t0, t1, question, mode=mode)
        # Dense over multiple ranges or a >60s non-counting span can't take the
        # chunked path — motion keyframes beat blind uniform sampling there.
        if density == "dense" and not counting and total_span > 45:
            density = "event"

        frames: List[Tuple[float, str]] = []
        spans: List[str] = []
        per_range = max(2, 10 // max(1, len(time_ranges)))
        for rng in time_ranges:
            t0, t1 = to_seconds(rng[0]), to_seconds(rng[1])
            if t1 <= t0:
                t1 = t0 + 10.0
            grabbed = self._grab_frames(t0, t1, density)
            # cap per-range so the combined set stays within one prompt
            if len(grabbed) > per_range:
                step = len(grabbed) / per_range
                grabbed = [grabbed[int(i * step)] for i in range(per_range)]
            frames.extend(grabbed)
            spans.append(f"{fmt(t0)}–{fmt(t1)}")
        frames.sort(key=lambda p: p[0])
        if not frames:
            return ("Could not decode any frames in those ranges — check the timestamps "
                    "against the video length, or read the graph with read_moment instead.")

        tss = ", ".join(fmt(ts) for ts, _ in frames)
        context = self._window_multimodal_context(time_ranges)

        # Build on prior close looks at this window instead of repeating them.
        t_lo = min(to_seconds(r[0]) for r in time_ranges)
        t_hi = max(to_seconds(r[1]) for r in time_ranges)
        priors = self.toolkit.prior_observations(t_lo, t_hi)
        if priors:
            pblock = "\n".join(f"[{fmt(n.t_start)}] {n.label}" for n in priors[:8])
            context = (context + "\n\n" if context else "") + \
                "Earlier inspections of this window (lower confidence):\n" + pblock

        q = (f"The {len(frames)} frames are at timestamps (in order): {tss}.\n"
             f"Question: {question}")
        if self.current_mcq:
            q += ("\n\nThis inspection serves the following multiple-choice "
                  "question:\n" + self.current_mcq +
                  "\n\nFirst describe what the frames actually show, then state "
                  "which option the VISUAL EVIDENCE best supports and why. Judge "
                  "only from the frames and the provided dialogue/text — note "
                  "explicitly if the frames match no option well.\n"
                  "IMPORTANT: if the question asks WHY something happens (motivation, "
                  "reason, cause), frames alone usually cannot decide it. In that "
                  "case do NOT pick an option — say 'the frames cannot decide this; "
                  "rely on the dialogue' and only report what is visually observable.")
        answer = self.llm.complete_with_images(
            question=q, image_urls=[u for _, u in frames], context=context)

        # Write the look back into the graph as a neutral, reusable caption.
        self._enrich_graph(t_lo, t_hi, question, answer, source="frame_inspection")

        return (f"Frame inspection [{density}] over {', '.join(spans)} "
                f"({len(frames)} frames"
                f"{', fused with dialogue/OCR' if context else ''}"
                f"{', informed by prior looks' if priors else ''}):\n{answer}")

    def _dense_count_inspect(self, t0: float, t1: float, question: str,
                             mode: str = "count") -> str:
        """Walk [t0,t1] in consecutive chunks at true counting density.

        Each chunk gets its own ≤FRAMES_PER_PASS-frame VLM pass (~3fps when the
        budget allows), so no instant is skipped; the per-chunk reports are then
        fused by a text pass that totals events across chunk boundaries."""
        span = t1 - t0
        # action mode: ~10fps in 10-frame passes (engine image limit) so
        # sub-second motion (a stroke, the ball's bounce) is actually seen;
        # counting keeps the cheaper ~3fps walk.
        if mode == "action":
            per_pass, fps, max_passes = 10, 10.0, 10
        else:
            per_pass, fps, max_passes = FRAMES_PER_PASS, 3.0, MAX_PASSES
        n_chunks = min(max_passes, max(2, int(span * fps / per_pass) + 1))
        bounds = [t0 + span * i / n_chunks for i in range(n_chunks + 1)]
        context = self._window_multimodal_context([[fmt(t0), fmt(t1)]])

        reports = []
        for c0, c1 in zip(bounds, bounds[1:]):
            frames = self._grab_frames(c0, c1, "dense")
            if len(frames) > per_pass:
                step = len(frames) / per_pass
                frames = [frames[int(i * step)] for i in range(per_pass)]
            if not frames:
                continue
            tss = ", ".join(fmt(ts) for ts, _ in frames)
            if mode == "count":
                seg_task = ("Report each occurrence of the relevant event with its "
                            "timestamp; say 'none in this segment' if nothing happens.")
            else:
                seg_task = ("Describe the action frame-by-frame: body posture, "
                            "limb/racket/ball positions and where the ball/object "
                            "ends up. Report only what is visibly happening; say "
                            "'nothing relevant in this segment' if so.")
            q = (f"These {len(frames)} consecutive frames cover {fmt(c0)}–{fmt(c1)} "
                 f"(timestamps in order: {tss}).\n"
                 f"Question (answer ONLY for this segment): {question}\n" + seg_task)
            try:
                ans = self.llm.complete_with_images(
                    question=q, image_urls=[u for _, u in frames], context=context)
            except Exception as exc:
                ans = f"(inspection failed: {exc})"
            reports.append(f"[segment {fmt(c0)}–{fmt(c1)}]\n{ans}")

        if not reports:
            return ("Could not decode any frames in that range — check the "
                    "timestamps against the video length, or read the graph with "
                    "read_moment instead.")

        joined = "\n\n".join(reports)
        mcq = (f"\n\nThe overall multiple-choice question:\n{self.current_mcq}"
               if self.current_mcq else "")
        if mode == "count":
            system = ("You combine sequential video-segment reports into one answer. "
                      "Sum event counts across segments, but an event spanning a "
                      "segment boundary counts once.")
            final_ask = "Final answer to the question, with the total count:"
        else:
            system = ("You combine sequential video-segment reports into one answer. "
                      "Preserve temporal order; reconstruct the action as it unfolds "
                      "across segments. Trust concrete frame observations over "
                      "narrative guesses — if segments disagree, say so.")
            final_ask = ("Final answer to the question, citing the segment "
                         "observations that support it:")
        fused = self.llm.complete([
            {"role": "system", "content": system},
            {"role": "user", "content":
             f"Question: {question}{mcq}\n\nSegment reports (consecutive, "
             f"non-overlapping, covering {fmt(t0)}–{fmt(t1)}):\n\n{joined}\n\n"
             + final_ask}],
            max_tokens=300)

        self._enrich_graph(t0, t1, question, fused, source="frame_inspection")
        return (f"Frame inspection [dense {mode}, {len(reports)} passes] over "
                f"{fmt(t0)}–{fmt(t1)}"
                f"{', fused with dialogue/OCR' if context else ''}:\n{fused}")

    def _enrich_graph(self, t0: float, t1: float, question: str,
                      answer: str, source: str) -> None:
        """Distil an inspection answer into a neutral caption and store it as an
        ObservationNode so later queries can retrieve it. Best-effort: a bad
        distillation must never break the inspection that already succeeded."""
        try:
            caption = self.llm.complete([
                {"role": "system", "content":
                 "Rewrite the answer as ONE neutral, self-contained sentence "
                 "describing what is in the video at this moment — no question, no "
                 "'the answer is', just the observed fact. If the answer is "
                 "uncertain or says nothing was found, reply with the single word "
                 "NONE."},
                {"role": "user", "content":
                 f"Question: {question}\nAnswer: {answer}\n\nNeutral one-sentence "
                 "description of what the video shows here:"}],
                max_tokens=80).strip()
            if caption and caption.upper() != "NONE" and len(caption) > 8:
                self.toolkit.write_observation(t0, t1, caption, source=source,
                                               confidence=0.5)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Grounded counting via a dedicated open-vocabulary detector
    # ------------------------------------------------------------------ #

    def _get_detector(self):
        if self._detector is None:
            from .detector import GroundingDinoDetector
            self._detector = GroundingDinoDetector()
        return self._detector

    def count_objects(
        self,
        time_range: A[List[str], D("A single [start, end] time range as 'HH:MM:SS' strings, e.g. [\"00:24:06\", \"00:24:40\"]. Keep it tight around the moment to count.")],
        object_phrase: A[str, D("What to count, as a short noun phrase the detector can ground, e.g. 'person', 'person carrying a sedan chair', 'lit candle', 'sword'. Be specific.")],
        box_threshold: A[float, D("Detection confidence cutoff 0-1 (default 0.30). Lower it (e.g. 0.2) if you expect small/occluded instances; raise it if it over-counts.")] = 0.30,
    ) -> str:
        """
        COUNT how many people or objects are PRESENT in a moment. Uses a real
        object detector (bounding boxes) — far more reliable than eyeballing frames.
        USE THIS FOR: "how many people carry the chair", "how many candles",
        "how many soldiers are sitting". Reports per-frame counts + max visible.
        For counting REPEATED EVENTS over time (how many times did she knock),
        use inspect_frames with sampling="dense" instead.
        """
        if not self.video_path:
            return "No raw video file configured — cannot run the detector."
        if not self._enable_detector:
            return ("Object detector is disabled this session. Count with "
                    "inspect_frames(sampling=\"dense\") instead.")
        t0, t1 = to_seconds(time_range[0]), to_seconds(time_range[1])
        if t1 <= t0:
            t1 = t0 + 6.0
        from qvkg.query.frame_extractor import extract_frames_for_window
        frames = [f for f in extract_frames_for_window(
            self.video_path, t0, t1, max_frames=12) if f.image is not None]
        if not frames:
            return ("Could not decode frames in that range — check the timestamps "
                    "or use inspect_frames.")
        try:
            det = self._get_detector()
        except Exception as exc:
            return (f"Object detector unavailable ({exc!s}). Fall back to "
                    "inspect_frames(sampling=\"dense\") for counting.")

        per_frame = []
        best = (-1, None, [])  # (count, timestamp, dets)
        for f in frames:
            try:
                dets = [d for d in det.detect(f.image, [object_phrase],
                                              box_threshold=box_threshold)]
            except Exception as exc:
                return (f"Detection failed ({exc!s}). Fall back to "
                        "inspect_frames(sampling=\"dense\").")
            c = len(dets)
            per_frame.append((f.timestamp, c))
            if c > best[0]:
                best = (c, f.timestamp, dets)

        lines = [f"OBJECT-DETECTION COUNT of \"{object_phrase}\" over "
                 f"{fmt(t0)}–{fmt(t1)} ({len(frames)} frames, threshold={box_threshold}):"]
        lines.append("Per-frame instance counts: "
                     + ", ".join(f"{fmt(ts)}={c}" for ts, c in per_frame))
        lines.append(f"\nMost instances simultaneously visible: {best[0]} "
                     f"(at {fmt(best[1])}).")
        if best[2]:
            confs = ", ".join(f"{d['score']:.2f}" for d in best[2])
            lines.append(f"Detection confidences at that frame: {confs}")
        lines.append("\nFor 'how many X are present', the answer is the maximum "
                     "simultaneous count above. If counts vary a lot across frames, "
                     "verify the busiest frame with inspect_frames.")

        # Persist the count back into the graph for reuse.
        if best[0] >= 0:
            self.toolkit.write_observation(
                t0, t1, f"{best[0]} × {object_phrase} visible (object detector)",
                source="object_detector", confidence=0.5,
                extra={"count": best[0], "phrase": object_phrase})
        return "\n".join(lines)

    def tools(self):
        belt = [self.inspect_frames]
        if self._enable_detector:
            belt.append(self.count_objects)
        return belt
