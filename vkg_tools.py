"""VKG tools: the graph-native tool belt registered into DVD's ReAct loop.

Each public method on :class:`VKGToolkit` is a tool the orchestrator can call.
Methods take only model-visible arguments (the graph/index/encoder are bound
state), return plain strings, and every observation ends with an affordance
footer suggesting concrete follow-up calls.

Tool map (mirrors `draft_plan.txt` / `current_vs_desired_framework.md`):

  find               — one search door: events/dialogue/text via FAISS+lexical,
                       or a person/object's full timeline if the query names one
  read_moment        — everything in a time window grouped by modality (focus=
                       'dialogue'/'text'/'actions' filters), with before/after
                       context and in-window causal links
  before_and_after   — chronological timeline around a moment (temporal walk, time-addressed)
  why_did_this_happen — causal chains + dialogue for a window (causal walk, time-addressed)

Internal (not registered as tools): get_overview (prefetched every question),
search_events, query_nodes, find_entity (merged into find/read_moment), and the
node-id tools the small policy never picked: follow_connections, trace_causes,
explain_why.

NOTE: no `from __future__ import annotations` here — DVD's schema generator
needs real (non-string) Annotated annotations on tool signatures.
"""

from typing import Annotated as A
from typing import Optional

from qvkg.query import graph_ops
from qvkg.schema import CAUSAL_EDGE_TYPES, VKGraph

from .func_schema import doc as D
from .serializer import (affordance_footer, node_line, serialize_chain,
                         serialize_nodes)
from .timeutil import fmt, fmt_span, to_seconds


_EVENT_TYPES = {"ActionNode", "InteractionNode", "StateChangeNode",
                "SpeechNode", "OCRNode", "AudioEventNode"}

_RELATION_HELP = ("one of: causal (why/effect), entity (same person/object), "
                  "speaker (who said it), temporal (before/after/during), "
                  "emotion (emotional shifts), similar (semantically related), "
                  "contains (hierarchy: episode→scene→clip)")


class VKGToolkit:
    """Binds a VKGraph (+ optional FAISS index, encoder, LLM) into callable tools."""

    def __init__(
        self,
        graph: VKGraph,
        faiss_index=None,
        text_encoder=None,           # object with .encode_text([str]) -> np.ndarray
        llm_complete=None,           # callable(messages) -> str, for edge inference
        inferred_edges_path: Optional[str] = None,
    ):
        self.graph = graph
        self.faiss_index = faiss_index
        self.text_encoder = text_encoder
        self.llm_complete = llm_complete
        self.inferred_edges_path = inferred_edges_path
        self._search_history: list[tuple[str, str]] = []
        self._obs_counter = 0
        self._load_cached_inferred_edges()

    # ------------------------------------------------------------------ #
    # Graph enrichment: write on-demand inspection results back as nodes
    # ------------------------------------------------------------------ #

    def write_observation(self, t0: float, t1: float, caption: str,
                          source: str, confidence: float = 0.5,
                          extra: Optional[dict] = None) -> str:
        """Persist an on-demand inspection/detection as an ObservationNode.

        The graph is built offline and sparse; every close look the agent takes
        is higher-fidelity than the build-time caption. Writing it back makes
        that detail retrievable by later queries (read_moment finds it by time,
        search_events's lexical branch by text) instead of re-paying the vision
        model. Tagged low-confidence with provenance so retrieval can flag it as
        inspected-not-verified."""
        from qvkg.schema import VKGNode
        caption = " ".join((caption or "").split())[:300]
        if not caption:
            return ""
        self._obs_counter += 1
        nid = f"obs_{self._obs_counter:04d}"
        meta = {"source": source, "provenance": "on_demand_inspection"}
        if extra:
            meta.update(extra)
        self.graph.add_node(VKGNode(
            id=nid, node_type="ObservationNode", label=caption, level=0,
            t_start=float(t0), t_end=float(t1), confidence=confidence,
            metadata=meta))
        return nid

    def prior_observations(self, t0: float, t1: float):
        """ObservationNodes already written for an overlapping window."""
        return [n for n in self.graph.get_nodes_in_window(t0, t1, buffer_sec=1.0)
                if n.node_type == "ObservationNode"]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_node(self, node_id: str):
        node = self.graph.get_node(node_id.strip())
        if node is None:
            # Tolerate the model pasting a label instead of an id.
            cands = list(self.graph.nodes.values())
            node = self.graph.find_event_by_description(node_id, cands)
        return node

    # Words that carry no retrieval signal — dropped from lexical queries so a
    # single content word ("incense", "sedan") is not drowned out.
    _STOPWORDS = frozenset(
        "the a an of to in on at and or for with from into over under "
        "is are was were be been being do does did what who how many when "
        "where why which that this these those his her their its it he she "
        "they them you i we as by about after before during while".split())

    def _query_terms(self, query: str):
        return {w for w in query.lower().replace(",", " ").split()
                if len(w) > 1 and w not in self._STOPWORDS}

    def _lexical_search(self, query: str, top_k: int):
        """Token-overlap fallback used when FAISS/encoder are unavailable.

        Scores by coverage of the query's content words (how many of them the
        node mentions), with a bonus for phrase substrings. Coverage — not
        Jaccard over the union — so a long, detailed node label is not
        penalised for being long, which previously made search return nothing.
        """
        q_words = self._query_terms(query)
        if not q_words:
            return []
        q_phrase = query.lower().strip()
        scored = []
        for n in self.graph.nodes.values():
            text = f"{n.label} {n.canonical_description or ''}".lower()
            if not text.strip():
                continue
            n_words = set(text.split())
            inter = len(q_words & n_words)
            # substring catches morphology / multiword terms word-split misses
            substr = sum(1 for w in q_words if w in text)
            hit = max(inter, substr)
            if hit == 0:
                continue
            score = hit / len(q_words)
            if q_phrase and len(q_phrase) > 4 and q_phrase in text:
                score += 1.0  # exact phrase present — strong signal
            scored.append((score, n))
        scored.sort(key=lambda p: -p[0])
        return [n for _, n in scored[:top_k]]

    def _semantic_search(self, query: str, top_k: int):
        ids = graph_ops.faiss_search(
            self.graph, self.faiss_index, self.text_encoder, query, k=top_k)
        return [self.graph.nodes[i] for i in ids if i in self.graph.nodes]

    def _load_cached_inferred_edges(self) -> None:
        import json
        import os
        if not self.inferred_edges_path or not os.path.exists(self.inferred_edges_path):
            return
        from qvkg.schema import VKGEdge
        with open(self.inferred_edges_path) as f:
            for ed in json.load(f):
                self.graph.add_edge(VKGEdge.from_dict(ed))

    def _cache_inferred_edges(self, edges) -> None:
        import json
        import os
        if not self.inferred_edges_path:
            return
        existing = []
        if os.path.exists(self.inferred_edges_path):
            with open(self.inferred_edges_path) as f:
                existing = json.load(f)
        existing.extend(e.to_dict() for e in edges)
        with open(self.inferred_edges_path, "w") as f:
            json.dump(existing, f, indent=2)

    # ------------------------------------------------------------------ #
    # Tools (model-visible)
    # ------------------------------------------------------------------ #

    def get_overview(self) -> str:
        """
        Get the structured overview of the video: episode → scene hierarchy with
        time spans and summaries, plus the registry of tracked characters/entities.
        Call this FIRST to understand the video's narrative structure before
        searching. This is the graph-native version of global browsing.
        """
        g = self.graph
        lines = ["VIDEO KNOWLEDGE GRAPH OVERVIEW", ""]

        episodes = g.get_episodes()
        if episodes:
            lines.append(f"Narrative hierarchy ({len(episodes)} episodes):")
            for ep in episodes:
                role = ep.metadata.get("narrative_role", "")
                role_s = f" [{role}]" if role else ""
                lines.append(f"({ep.id} | {fmt_span(ep.t_start, ep.t_end)}){role_s} {ep.label}")
                for sc in g.get_children(ep):
                    lines.append(f"    ({sc.id} | {fmt_span(sc.t_start, sc.t_end)}) {sc.label}")
        else:
            scenes = sorted(g.get_nodes_by_type("SceneNode"), key=lambda n: n.t_start)
            lines.append(f"Scenes ({len(scenes)}):")
            lines += [f"  {node_line(g, sc, with_edges=False)}" for sc in scenes[:40]]

        chars = g.get_all_character_mentions()
        if chars:
            seen = {}
            for c in chars:
                key = c.entity_id or c.id
                seen.setdefault(key, c)
            lines.append("")
            lines.append(f"Characters/entities tracked ({len(seen)}):")
            for key, c in list(seen.items())[:30]:
                n_app = len(g.entity_idx.get(c.entity_id, [])) if c.entity_id else 1
                desc = c.canonical_description or c.label
                lines.append(f"  {key}: {desc} ({n_app} appearances)")

        n_causal = sum(1 for es in g.edges.values() for e in es
                       if e.relation_type in CAUSAL_EDGE_TYPES)
        lines.append("")
        lines.append(f"Graph stats: {len(g.nodes)} nodes, {n_causal} causal edges, "
                     f"node types: {', '.join(sorted(g.type_idx.keys()))}")
        lines.append("")
        lines.append("— What you can do next —")
        lines.append('• locate question-relevant events: search_events(query="...")')
        lines.append('• read one scene/episode closely: read_moment(time_start, time_end)')
        lines.append('• follow a character through the video: find_entity(name="...")')
        return "\n".join(lines)

    def search_events(
        self,
        query: A[str, D("Natural-language description of the event, object, speech, or detail to find.")],
        top_k: A[int, D("Max results to return. Default 10.")] = 10,
        node_types: A[str, D("Optional comma-separated node types to restrict to, e.g. 'SpeechNode,OCRNode'. Use 'any' for no filter.")] = "any",
    ) -> str:
        """
        Semantic search over ALL nodes of the video knowledge graph (events, speech,
        on-screen text, objects, scenes). Returns matching nodes with their ids,
        timestamps, and the edges available from each — use those ids with
        before_and_after / why_did_this_happen / find_entity to follow the structure.
        """
        g = self.graph
        present_types = set(g.type_idx.keys())

        # --- 1. Guard against the re-query-with-a-shorter-string loop. -----
        norm_q = " ".join(self._query_terms(query))
        sig = (norm_q, (node_types or "any").strip().lower())
        repeats = sum(1 for s in self._search_history if s == sig)
        # also catch "same content words as a previous call" (the shrink loop)
        subset_of_prior = any(
            norm_q and set(norm_q.split()) <= set(prev.split())
            for prev, _ in self._search_history)
        self._search_history.append(sig)
        if repeats >= 1 or (subset_of_prior and len(self._search_history) > 1):
            return (
                f'You have already searched for these terms ("{query}"). '
                "Re-running search with the same or a shorter query will not return "
                "anything new. CHANGE STRATEGY:\n"
                "• If you know the time (e.g. a [Time reference]) → "
                'read_moment(time_start="...", time_end="...") to read everything there.\n'
                "• To enumerate a category exhaustively → "
                'query_nodes(node_type="SpeechNode"/"OCRNode"/"ActionNode", time_start, time_end).\n'
                "• For a fine visual detail (count, color, pose, on-screen text) → "
                "inspect_frames on the time range. The graph localises; the frames "
                "give the visual answer.")

        # --- 2. Validate / clean the node_types filter. --------------------
        allowed = None
        note = ""
        if node_types and node_types.strip().lower() != "any":
            requested = {t.strip() for t in node_types.replace(",", " ").split() if t.strip()}
            allowed = {t for t in requested if t in present_types}
            unknown = requested - present_types
            if unknown:
                note += (f"\n(note: ignored node type(s) not in this graph: "
                         f"{', '.join(sorted(unknown))}. Present types: "
                         f"{', '.join(sorted(present_types))}.)")
            if not allowed:
                allowed = None  # all requested types bogus → don't filter to empty

        # --- 3. Retrieve: blend semantic + lexical so keyword hits surface. -
        if self.faiss_index is not None and self.text_encoder is not None:
            ranked = self._semantic_search(query, top_k * 3)
            mode = "semantic+lexical"
            seen = {n.id for n in ranked}
            for n in self._lexical_search(query, top_k * 3):
                if n.id not in seen:
                    ranked.append(n)
                    seen.add(n.id)
        else:
            ranked = self._lexical_search(query, top_k * 3)
            mode = "lexical (no FAISS index loaded)"

        hits = ranked if allowed is None else [n for n in ranked if n.node_type in allowed]

        # --- 4. Zero-hit fallback: retry unfiltered before giving up. ------
        if not hits and allowed is not None:
            hits = ranked
            note += "\n(note: no results matched the type filter — showing all types instead.)"

        hits = hits[:top_k]
        return serialize_nodes(
            g, hits,
            title=f'Search results for "{query}" [{mode}]',
        ) + note

    def query_nodes(
        self,
        node_type: A[str, D("Node type to fetch: SceneNode, EpisodeNode, ClipNode, ActionNode, InteractionNode, StateChangeNode, SpeechNode, OCRNode, AudioEventNode, CharacterNode, ObjectNode — or 'any'.")],
        time_start: A[str, D("Optional window start as HH:MM:SS or seconds. Empty = video start.")] = "",
        time_end: A[str, D("Optional window end as HH:MM:SS or seconds. Empty = video end.")] = "",
        label_contains: A[str, D("Optional case-insensitive substring the node label must contain.")] = "",
    ) -> str:
        """
        Structured query over the graph by node type, time window, and label —
        deterministic and exhaustive, unlike semantic search. Use it to enumerate
        ALL speech in a window, ALL on-screen text, ALL state changes, etc.
        """
        g = self.graph
        if node_type.strip().lower() == "any":
            nodes = list(g.nodes.values())
        else:
            nodes = g.get_nodes_by_type(node_type.strip())
            if not nodes:
                return (f"No nodes of type {node_type!r}. Types present in this graph: "
                        f"{', '.join(sorted(g.type_idx.keys()))}")
        t0 = to_seconds(time_start) if str(time_start).strip() else None
        t1 = to_seconds(time_end) if str(time_end).strip() else None
        if t0 is not None:
            nodes = [n for n in nodes if n.t_end >= t0]
        if t1 is not None:
            nodes = [n for n in nodes if n.t_start <= t1]
        if label_contains.strip():
            sub = label_contains.strip().lower()
            nodes = [n for n in nodes
                     if sub in n.label.lower()
                     or sub in (n.canonical_description or "").lower()]
        nodes.sort(key=lambda n: n.t_start)
        win = ""
        if t0 is not None or t1 is not None:
            win = f" in {fmt(t0 or 0)}–{fmt(t1) if t1 is not None else 'end'}"
        return serialize_nodes(
            self.graph, nodes,
            title=f"{node_type} nodes{win}"
                  + (f' with label ~ "{label_contains}"' if label_contains else ""),
            max_nodes=40,
        )

    def follow_connections(
        self,
        node_id: A[str, D("The id of the node to start from (as returned by other vkg tools, e.g. 'ev_12').")],
        relation: A[str, D(_RELATION_HELP)],
        hops: A[int, D("How many hops to expand (1-3). Default 1.")] = 1,
    ) -> str:
        """
        Follow typed edges from a node to discover connected evidence. USE THIS for:
        "what happens before/after X" → relation="temporal"
        "who said what to whom" → relation="speaker"
        "what caused this" → relation="causal"
        "what else involves this character" → relation="entity"
        """
        node = self._resolve_node(node_id)
        if node is None:
            return (f"Node {node_id!r} not found. Use the exact parenthesized id from a "
                    "previous observation, e.g. follow_connections(node_id=\"ev_12\", ...).")
        rel = relation.lower().strip()
        if rel not in graph_ops.RELATION_EDGE_TYPES:
            return f"Unknown relation {relation!r}. Valid: {', '.join(graph_ops.VALID_RELATIONS)}."

        frontier = {node.id}
        collected = {node.id}
        all_edges = []
        for _ in range(max(1, min(int(hops), 3))):
            new_ids, edges = graph_ops.expand(self.graph, frontier, rel, k=6)
            all_edges.extend(edges)
            new_ids -= collected
            if not new_ids:
                break
            collected |= new_ids
            frontier = new_ids

        nodes = [self.graph.nodes[i] for i in collected if i in self.graph.nodes]
        header = (f"Traversal from ({node.id}) \"{node.label[:60]}\" via {rel} "
                  f"({len(all_edges)} edges):")
        body = serialize_nodes(self.graph, nodes, title=header)
        if rel == "causal" and all_edges:
            body = serialize_chain(self.graph, all_edges, "Causal links traversed:") + "\n\n" + body
        if not all_edges and rel == "causal":
            body += ("\nNo pre-computed causal edges here — try "
                     f'explain_why(time_start="{fmt(node.t_start - 60)}", '
                     f'time_end="{fmt(node.t_end + 60)}") to infer them on the fly.')
        return body

    def trace_causes(
        self,
        node_id: A[str, D("Id of the event node to explain (e.g. 'ev_12').")],
        direction: A[str, D("'why' = trace causes backward, 'effect' = trace consequences forward, 'both' = both.")] = "both",
        depth: A[int, D("Max chain length to follow (1-5). Default 3.")] = 3,
    ) -> str:
        """
        Answer WHY questions by tracing the causal chain. Walks backward from an
        event to find root causes (why it happened) and forward to find consequences
        (what it led to). USE THIS for: "why did X happen", "what caused X",
        "what is the reason". First find the event with search_events or
        read_moment, then call this on its node id.
        """
        node = self._resolve_node(node_id)
        if node is None:
            return f"Node {node_id!r} not found — pass an id from a previous observation."
        depth = max(1, min(int(depth), 5))
        d = direction.lower().strip()

        chains = []
        sections = []
        if d in ("why", "both"):
            frontier, seen = [node.id], {node.id}
            for _ in range(depth):
                nxt = []
                for nid in frontier:
                    for e in self.graph.get_incoming_edges(nid):
                        if e.relation_type in CAUSAL_EDGE_TYPES and e.source_id not in seen:
                            chains.append(e)
                            seen.add(e.source_id)
                            nxt.append(e.source_id)
                frontier = nxt
                if not frontier:
                    break
            sections.append(serialize_chain(
                self.graph, chains,
                f"WHY ({node.id} happened) — causes, walking backward:"))
        if d in ("effect", "both"):
            chains2 = []
            frontier, seen = [node.id], {node.id}
            for _ in range(depth):
                nxt = []
                for nid in frontier:
                    for e in self.graph.get_edges(nid):
                        if e.relation_type in CAUSAL_EDGE_TYPES and e.target_id not in seen:
                            chains2.append(e)
                            seen.add(e.target_id)
                            nxt.append(e.target_id)
                frontier = nxt
                if not frontier:
                    break
            sections.append(serialize_chain(
                self.graph, chains2,
                f"EFFECTS (what {node.id} led to) — consequences, walking forward:"))
            chains.extend(chains2)

        out = f'Causal analysis of ({node.id}) "{node.label[:80]}" @{fmt(node.t_start)}\n\n'
        out += "\n\n".join(sections)
        endpoint_ids = {e.source_id for e in chains} | {e.target_id for e in chains}
        endpoints = [self.graph.nodes[i] for i in endpoint_ids if i in self.graph.nodes]
        if endpoints:
            out += "\n" + affordance_footer(self.graph, endpoints)
        return out

    def before_and_after(
        self,
        time: A[str, D("The moment of interest as HH:MM:SS or seconds — e.g. the time the question refers to.")],
        window: A[int, D("Seconds of context to show on each side (15-300). Default 90.")] = 90,
    ) -> str:
        """
        See what happens just BEFORE and just AFTER a moment, in strict
        chronological order. USE THIS for: "what happens before/after X",
        "what does X do next", "what led up to this", "which came first /
        in what order", and whenever the answer may lie OUTSIDE the
        question's time window. No node ids needed — give it a timestamp
        and it walks the video's timeline for you.
        """
        t = to_seconds(time)
        w = float(max(15, min(int(window), 300)))
        ev_types = _EVENT_TYPES | {"SceneNode"}
        nodes = [n for n in self.graph.get_nodes_in_window(max(0.0, t - w), t + w, buffer_sec=0.0)
                 if n.node_type in ev_types]
        if not nodes:
            return (f"No events within {int(w)}s of {fmt(t)}. Widen the window "
                    f'(before_and_after(time="{fmt(t)}", window={int(min(w * 2, 300))})) '
                    "or localize the moment first with search_events.")
        before = sorted((n for n in nodes if n.t_end < t), key=lambda n: n.t_start)
        during = sorted((n for n in nodes if n.t_start <= t <= n.t_end), key=lambda n: n.t_start)
        after = sorted((n for n in nodes if n.t_start > t), key=lambda n: n.t_start)

        # Explicit temporal edges can reach events beyond the fixed window.
        linked = []
        if during:
            here = {n.id for n in nodes}
            ids, _ = graph_ops.expand(self.graph, {n.id for n in during}, "temporal", k=6)
            linked = [self.graph.nodes[i] for i in ids
                      if i in self.graph.nodes and i not in here]

        out = [f"TIMELINE around {fmt(t)} (±{int(w)}s), in chronological order:"]
        for title, group, keep in ((f"BEFORE {fmt(t)}", before, before[-20:]),
                                   (f"AT {fmt(t)}", during, during[:20]),
                                   (f"AFTER {fmt(t)}", after, after[:20])):
            if not group:
                continue
            extra = f", showing the {len(keep)} nearest" if len(group) > len(keep) else ""
            out.append(f"\n{title} ({len(group)} events{extra}):")
            out += ["  " + node_line(self.graph, n) for n in keep]
        if linked:
            linked.sort(key=lambda n: n.t_start)
            out.append("\nLINKED by explicit temporal edges (beyond this window):")
            out += ["  " + node_line(self.graph, n) for n in linked[:8]]
        out.append(affordance_footer(self.graph, before[-6:] + during[:4] + after[:6]))
        return "\n".join(out)

    def why_did_this_happen(
        self,
        time_start: A[str, D("Start of the moment to explain, as HH:MM:SS or seconds.")],
        time_end: A[str, D("End of the moment to explain, as HH:MM:SS or seconds.")],
    ) -> str:
        """
        Explain WHY the events in a time window happened and what they led to.
        USE THIS for: "why did X happen", "what is the reason / cause /
        purpose / motivation", "what did this lead to". Traces the graph's
        cause-and-effect links backward (causes) and forward (consequences)
        from every event in the window, and quotes the dialogue around the
        moment — the reason is usually SPOKEN out loud. If no causal links
        exist yet, it infers them from the surrounding evidence.
        """
        t0, t1 = to_seconds(time_start), to_seconds(time_end)
        if t1 <= t0:
            return f"time_end ({fmt(t1)}) must be after time_start ({fmt(t0)})."
        events = [n for n in self.graph.get_nodes_in_window(t0, t1, buffer_sec=0.0)
                  if n.node_type in _EVENT_TYPES]
        if not events:
            return (f"No events recorded in {fmt_span(t0, t1)} — localize the moment "
                    "first with search_events or before_and_after.")

        causes, effects = [], []
        seen = {n.id for n in events}
        frontier = list(seen)
        for _ in range(3):
            nxt = []
            for nid in frontier:
                for e in self.graph.get_incoming_edges(nid):
                    if e.relation_type in CAUSAL_EDGE_TYPES and e.source_id not in seen:
                        causes.append(e)
                        seen.add(e.source_id)
                        nxt.append(e.source_id)
            frontier = nxt
            if not frontier:
                break
        seen = {n.id for n in events}
        frontier = list(seen)
        for _ in range(3):
            nxt = []
            for nid in frontier:
                for e in self.graph.get_edges(nid):
                    if e.relation_type in CAUSAL_EDGE_TYPES and e.target_id not in seen:
                        effects.append(e)
                        seen.add(e.target_id)
                        nxt.append(e.target_id)
            frontier = nxt
            if not frontier:
                break

        sections = []
        if causes:
            sections.append(serialize_chain(
                self.graph, causes,
                f"CAUSES — why the events in {fmt_span(t0, t1)} happened:"))
        if effects:
            sections.append(serialize_chain(
                self.graph, effects, "EFFECTS — what they led to:"))

        speech = sorted((n for n in self.graph.get_nodes_in_window(
                             max(0.0, t0 - 45), t1 + 45, buffer_sec=0.0)
                         if n.node_type == "SpeechNode"), key=lambda n: n.t_start)
        if speech:
            lines = [f'  @{fmt(n.t_start)}: "{n.label[:120]}"' for n in speech[:25]]
            sections.append("DIALOGUE around this moment (the reason is often said out loud):\n"
                            + "\n".join(lines))

        if not causes and not effects:
            sections.append("No pre-computed causal links here — inferring from the evidence:\n"
                            + self.explain_why(time_start, time_end))

        endpoint_ids = (({e.source_id for e in causes} | {e.target_id for e in effects})
                        or {n.id for n in events})
        endpoints = [self.graph.nodes[i] for i in endpoint_ids if i in self.graph.nodes]
        out = (f"WHY-ANALYSIS of {fmt_span(t0, t1)} ({len(events)} events):\n\n"
               + "\n\n".join(sections))
        out += "\n" + affordance_footer(self.graph, endpoints)
        return out

    def find_entity(
        self,
        name: A[str, D("A person or character name (e.g. 'the protagonist', 'the man'), an entity id (e.g. 'entity_3'), or a node id of a CharacterNode/ObjectNode.")],
    ) -> str:
        """
        Track one person or character across the whole video: every appearance in
        chronological order, plus what they do, who they interact with, and what
        they say. USE THIS when the question asks about a specific person:
        "what did the protagonist do", "what does the man say", "who is X",
        "how many times does X appear", "does X show up earlier/again",
        "have they seen / been to / done this BEFORE" (recurrence checks —
        verify against the whole timeline, not just one window).
        """
        g = self.graph
        key = name.strip()
        entity_id = key if key in g.entity_idx else None
        if entity_id is None:
            node = g.get_node(key)
            if node is not None and node.entity_id:
                entity_id = node.entity_id
        if entity_id is None:
            kl = key.lower()
            best = None
            for c in g.get_all_character_mentions() + g.get_nodes_by_type("ObjectNode"):
                hay = f"{c.label} {c.canonical_description or ''}".lower()
                if kl in hay or hay in kl:
                    best = c
                    break
            if best is not None:
                entity_id = best.entity_id
                if entity_id is None:
                    return node_line(g, best) + "\n(no entity_id — this mention is not linked across clips)"
        if entity_id is None:
            known = sorted(g.entity_idx.keys())[:30]
            return (f"No entity matching {name!r}. Known entity ids: {', '.join(known) or '(none)'}. "
                    "Try get_overview to see the character registry, or search_events for the description.")

        appearances = graph_ops.trace_entity(g, entity_id)
        lines = [f"ENTITY TIMELINE for {entity_id} — {len(appearances)} appearances:"]
        for n in appearances:
            lines.append("  " + node_line(g, n))
            for e in g.get_edges(n.id):
                if e.relation_type in ("PERFORMS", "INTERACTS_WITH"):
                    tgt = g.nodes.get(e.target_id)
                    if tgt:
                        lines.append(f"      {e.relation_type} → ({tgt.id}) {tgt.label[:70]}")
            for e in g.get_incoming_edges(n.id):
                if e.relation_type == "SPOKEN_BY":
                    src = g.nodes.get(e.source_id)
                    if src:
                        lines.append(f"      says @{fmt(src.t_start)}: \"{src.label[:70]}\"")
        body = "\n".join(lines)
        body += "\n" + affordance_footer(g, appearances)
        return body

    def find(
        self,
        what: A[str, D("What to find: an event ('the car crash'), a spoken phrase, an on-screen text, an object, or a PERSON'S NAME/description ('the protagonist', 'the man in the hat') — a person returns their full timeline across the video.")],
    ) -> str:
        """
        Find anything anywhere in the video: where something happens, where
        something is said or shown, or everything a person/object does across
        the WHOLE video. USE THIS for: "where/when does X happen", "who is X",
        "what does X do", "does X appear earlier or again", "have they been
        here before". One search door for events, dialogue, text, and people.
        """
        g = self.graph
        key = (what or "").strip()
        if not key:
            return "Give me something to find — an event description, a phrase, or a person."
        # A tracked entity? Return its full timeline (the old find_entity).
        entity_hit = key in g.entity_idx
        if not entity_hit:
            node = g.get_node(key)
            entity_hit = node is not None and node.entity_id
        if not entity_hit:
            kl = key.lower()
            for c in g.get_all_character_mentions() + g.get_nodes_by_type("ObjectNode"):
                hay = f"{c.label} {c.canonical_description or ''}".lower()
                if c.entity_id and (kl in hay or hay in kl):
                    entity_hit = True
                    break
        if entity_hit:
            return self.find_entity(key)
        return self.search_events(key)

    def read_moment(
        self,
        time_start: A[str, D("Window start as HH:MM:SS or seconds.")],
        time_end: A[str, D("Window end as HH:MM:SS or seconds.")],
        focus: A[str, D("What to read: 'all' (default), 'dialogue' (only spoken lines — use for what is SAID/heard), 'text' (only on-screen text), 'actions' (only events), 'entities' (only people/objects present).")] = "all",
    ) -> str:
        """
        Read EVERYTHING the graph knows inside a time window, grouped by modality:
        scenes, actions/events, speech (with speakers), on-screen text (OCR),
        audio events, and entities present — plus what happens just before and
        after the window. focus="dialogue" reads only the spoken lines (the
        audio): use it for "what does X say / hear / talk about".
        """
        t0, t1 = to_seconds(time_start), to_seconds(time_end)
        if t1 <= t0:
            return f"time_end ({fmt(t1)}) must be after time_start ({fmt(t0)})."
        nodes = self.graph.get_nodes_in_window(t0, t1, buffer_sec=0.0)
        focus = (focus or "all").lower().strip()
        _FOCUS = {
            "dialogue": {"Speech", "Audio events"},
            "speech":   {"Speech", "Audio events"},
            "audio":    {"Speech", "Audio events"},
            "text":     {"On-screen text (OCR)"},
            "ocr":      {"On-screen text (OCR)"},
            "actions":  {"Actions & events"},
            "events":   {"Actions & events"},
            "entities": {"Entities present"},
        }
        keep = _FOCUS.get(focus)  # None → all groups
        groups = {
            "Scenes/structure": [n for n in nodes if n.node_type in ("SceneNode", "EpisodeNode", "ClipNode")],
            "Actions & events": [n for n in nodes if n.node_type in ("ActionNode", "InteractionNode", "StateChangeNode")],
            "Speech": [n for n in nodes if n.node_type == "SpeechNode"],
            "On-screen text (OCR)": [n for n in nodes if n.node_type == "OCRNode"],
            "Audio events": [n for n in nodes if n.node_type == "AudioEventNode"],
            "Entities present": [n for n in nodes if n.node_type in ("CharacterNode", "ObjectNode")],
            "Prior inspections (on-demand, lower confidence)":
                [n for n in nodes if n.node_type == "ObservationNode"],
        }
        out = [f"WINDOW {fmt_span(t0, t1)}"
               + (f" — focus: {focus}" if keep else f" — {len(nodes)} nodes")]
        # A focused read shows MORE of the chosen modality (the cap exists to
        # keep the all-modality dump bounded, not to hide dialogue lines).
        cap = 80 if keep else 30
        for gname, gnodes in groups.items():
            if not gnodes or (keep and gname not in keep):
                continue
            gnodes.sort(key=lambda n: n.t_start)
            out.append(f"\n{gname} ({len(gnodes)}):")
            out += ["  " + node_line(self.graph, n) for n in gnodes[:cap]]
            if len(gnodes) > cap:
                out.append(f"  … {len(gnodes) - cap} more")

        # Deterministic temporal context: "what happens before/after" answers
        # usually sit just OUTSIDE the asked window, and the small policy
        # rarely issues a second call to go look — so always show the nearest
        # neighbouring events on both sides.
        ev_types = _EVENT_TYPES | {"SceneNode"}
        prev = sorted((n for n in self.graph.get_nodes_in_window(
                           max(0.0, t0 - 120), t0, buffer_sec=0.0)
                       if n.node_type in ev_types and n.t_end <= t0),
                      key=lambda n: n.t_start)
        nxt = sorted((n for n in self.graph.get_nodes_in_window(
                          t1, t1 + 120, buffer_sec=0.0)
                      if n.node_type in ev_types and n.t_start >= t1),
                     key=lambda n: n.t_start)
        if prev:
            out.append(f"\nJust BEFORE this window (nearest {len(prev[-6:])} of {len(prev)}):")
            out += ["  " + node_line(self.graph, n) for n in prev[-6:]]
        if nxt:
            out.append(f"\nJust AFTER this window (nearest {len(nxt[:6])} of {len(nxt)}):")
            out += ["  " + node_line(self.graph, n) for n in nxt[:6]]

        # And any cause→effect links the graph already knows inside the window.
        win_ids = {n.id for n in nodes}
        clinks = [e for nid in win_ids for e in self.graph.get_edges(nid)
                  if e.relation_type in CAUSAL_EDGE_TYPES and e.target_id in win_ids]
        if clinks:
            out.append("")
            out.append(serialize_chain(self.graph, clinks,
                                       "Cause→effect links inside this window:"))

        evented = groups["Actions & events"] + groups["Speech"]
        out.append(affordance_footer(self.graph, evented or nodes))
        return "\n".join(out)

    def explain_why(
        self,
        time_start: A[str, D("Window start as HH:MM:SS or seconds.")],
        time_end: A[str, D("Window end as HH:MM:SS or seconds.")],
    ) -> str:
        """
        Answer WHY questions when the graph has no pre-computed causal edges. Infers
        causal links between events in the time window using the LLM, then caches
        them for future use. USE THIS for: "why did X happen", "what caused this",
        "what is the reason" — when trace_causes returns no causal edges.
        """
        if self.llm_complete is None:
            return ("Edge inference LLM is not configured for this session. "
                    "Fall back to read_moment on the same span and reason over "
                    "temporal order yourself.")
        import json

        t0, t1 = to_seconds(time_start), to_seconds(time_end)
        events = [n for n in self.graph.get_nodes_in_window(t0, t1, buffer_sec=0.0)
                  if n.node_type in _EVENT_TYPES]
        events.sort(key=lambda n: n.t_start)
        if len(events) < 2:
            return f"Only {len(events)} event(s) in {fmt_span(t0, t1)} — nothing to link."
        events = events[:40]

        listing = "\n".join(
            f"{n.id} @{fmt(n.t_start)} [{n.node_type}]: {n.label}" for n in events)
        prompt = (
            "Below are time-ordered events from a video. Identify causal links "
            "between them. Only assert links clearly supported by the events.\n"
            "Respond with a JSON array, each item: {\"source\": \"<id>\", "
            "\"target\": \"<id>\", \"relation\": \"CAUSES|ENABLES|PREVENTS|MOTIVATES\", "
            "\"confidence\": 0.0-1.0, \"rationale\": \"<one sentence>\"}.\n"
            "Respond with ONLY the JSON array.\n\nEvents:\n" + listing
        )
        raw = self.llm_complete([
            {"role": "system", "content": "You are an expert at causal analysis of video event sequences."},
            {"role": "user", "content": prompt},
        ])
        try:
            start, end = raw.find("["), raw.rfind("]")
            items = json.loads(raw[start:end + 1])
        except Exception:
            return f"Edge inference returned unparseable output:\n{raw[:500]}"

        from qvkg.schema import VKGEdge
        valid_ids = {n.id for n in events}
        new_edges = []
        for it in items:
            if (it.get("source") in valid_ids and it.get("target") in valid_ids
                    and it.get("relation") in CAUSAL_EDGE_TYPES):
                new_edges.append(VKGEdge(
                    source_id=it["source"], target_id=it["target"],
                    relation_type=it["relation"],
                    weight=1.0, confidence=float(it.get("confidence", 0.5)),
                    metadata={"source": "online_inference",
                              "rationale": it.get("rationale", "")},
                ))
        self.graph.add_edges(new_edges)
        self._cache_inferred_edges(new_edges)
        out = serialize_chain(self.graph, new_edges,
                              f"Inferred {len(new_edges)} causal edge(s) in {fmt_span(t0, t1)} "
                              "(now cached in the graph):")
        out += ("\n\nThese edges are now traversable — "
                "why_did_this_happen over this span will include them. "
                "Inferred edges carry model confidence; CONFIRM important ones with inspect_frames.")
        return out

    # All tools, in registration order — one tool per QUESTION SHAPE, merged
    # within shapes so the 4B policy has exactly one obvious choice per need:
    #   find             — where / who / does-it-recur   (search_events + find_entity)
    #   read_moment      — what's in this window          (absorbs query_nodes via focus=)
    #   before_and_after — what surrounds this moment
    #   why_did_this_happen — why / what did it lead to
    # Unregistered but kept as methods (used internally / by the prefetch):
    # get_overview (prefetched every question), search_events, query_nodes,
    # find_entity, follow_connections, trace_causes, explain_why.
    def tools(self):
        return [
            self.find, self.read_moment,
            self.before_and_after, self.why_did_this_happen,
        ]
