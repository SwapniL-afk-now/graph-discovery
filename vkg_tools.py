"""VKG tools: the graph-native tool belt registered into DVD's ReAct loop.

Each public method on :class:`VKGToolkit` is a tool the orchestrator can call.
Methods take only model-visible arguments (the graph/index/encoder are bound
state), return plain strings, and every observation ends with an affordance
footer suggesting concrete follow-up calls.

Tool map (mirrors `draft_plan.txt` / `current_vs_desired_framework.md`):

  vkg_overview      — hierarchy + characters: the structured global_browse
  vkg_search        — dual-index semantic search (FAISS) with lexical fallback
  vkg_query         — structured access by type / time / label
  vkg_traverse      — follow one edge family from a node
  vkg_causal        — causal chain traversal (why / what happened next)
  vkg_entity        — character & object timelines (identity tracking)
  vkg_window        — everything in a time window, grouped by modality
  vkg_infer_causal  — on-the-fly causal edge inference, cached into the graph

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

_RELATION_HELP = ("one of: CAUSAL (why/effect), ENTITY (same person/object), "
                  "SPEAKER (who said it), TEMPORAL (before/after/during), "
                  "EMOTION (emotional shifts), SIMILAR (semantically related), "
                  "CONTAINS (hierarchy: episode→scene→clip)")


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
        self._load_cached_inferred_edges()

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

    def _lexical_search(self, query: str, top_k: int):
        """Token-overlap fallback used when FAISS/encoder are unavailable."""
        q_words = {w for w in query.lower().split() if len(w) > 2}
        scored = []
        for n in self.graph.nodes.values():
            text = f"{n.label} {n.canonical_description or ''}".lower()
            n_words = {w for w in text.split() if len(w) > 2}
            if not n_words or not q_words:
                continue
            inter = len(q_words & n_words)
            if inter == 0:
                continue
            scored.append((inter / len(q_words | n_words), n))
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

    def vkg_overview(self) -> str:
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
        lines.append('• locate question-relevant events: vkg_search(query="...")')
        lines.append('• read one scene/episode closely: vkg_window(time_start, time_end)')
        lines.append('• follow a character through the video: vkg_entity(name="...")')
        return "\n".join(lines)

    def vkg_search(
        self,
        query: A[str, D("Natural-language description of the event, object, speech, or detail to find.")],
        top_k: A[int, D("Max results to return. Default 10.")] = 10,
        node_types: A[str, D("Optional comma-separated node types to restrict to, e.g. 'SpeechNode,OCRNode'. Use 'any' for no filter.")] = "any",
    ) -> str:
        """
        Semantic search over ALL nodes of the video knowledge graph (events, speech,
        on-screen text, objects, scenes). Returns matching nodes with their ids,
        timestamps, and the edges available from each — use those ids with
        vkg_traverse / vkg_causal / vkg_entity to follow the structure.
        """
        if self.faiss_index is not None and self.text_encoder is not None:
            hits = self._semantic_search(query, top_k * 2)
            mode = "semantic (FAISS)"
        else:
            hits = self._lexical_search(query, top_k * 2)
            mode = "lexical (no FAISS index loaded)"

        if node_types and node_types.strip().lower() != "any":
            allowed = {t.strip() for t in node_types.split(",")}
            hits = [n for n in hits if n.node_type in allowed]
        hits = hits[:top_k]
        return serialize_nodes(
            self.graph, hits,
            title=f'Search results for "{query}" [{mode}]',
        )

    def vkg_query(
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

    def vkg_traverse(
        self,
        node_id: A[str, D("The id of the node to start from (as returned by other vkg tools, e.g. 'ev_12').")],
        relation: A[str, D(_RELATION_HELP)],
        hops: A[int, D("How many hops to expand (1-3). Default 1.")] = 1,
    ) -> str:
        """
        Follow one family of typed edges outward from a node (both directions) and
        return the reached neighbourhood. This is how you do multi-hop reasoning:
        find a seed node with vkg_search, then traverse CAUSAL / ENTITY / TEMPORAL /
        SPEAKER / CONTAINS edges to collect connected evidence.
        """
        node = self._resolve_node(node_id)
        if node is None:
            return (f"Node {node_id!r} not found. Use the exact parenthesized id from a "
                    "previous observation, e.g. vkg_traverse(node_id=\"ev_12\", ...).")
        rel = relation.upper().strip()
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
        if rel == "CAUSAL" and all_edges:
            body = serialize_chain(self.graph, all_edges, "Causal links traversed:") + "\n\n" + body
        if not all_edges and rel == "CAUSAL":
            body += ("\nNo pre-computed causal edges here — try "
                     f'vkg_infer_causal(time_start="{fmt(node.t_start - 60)}", '
                     f'time_end="{fmt(node.t_end + 60)}") to infer them on the fly.')
        return body

    def vkg_causal(
        self,
        node_id: A[str, D("Id of the event node to explain (e.g. 'ev_12').")],
        direction: A[str, D("'why' = trace causes backward, 'effect' = trace consequences forward, 'both' = both.")] = "both",
        depth: A[int, D("Max chain length to follow (1-5). Default 3.")] = 3,
    ) -> str:
        """
        Trace the causal chain through CAUSES / ENABLES / PREVENTS / MOTIVATES edges.
        Use 'why' for "Why did X happen?" questions (walks backward to root causes)
        and 'effect' for "What happened because of X?" (walks forward to outcomes).
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

    def vkg_entity(
        self,
        name: A[str, D("Character/object name, an entity id (e.g. 'entity_3'), or a node id of a CharacterNode/ObjectNode.")],
    ) -> str:
        """
        Track one character or object across the whole video: every appearance in
        chronological order, plus what they do (PERFORMS), who they interact with
        (INTERACTS_WITH), and what they say (SPOKEN_BY). Answers "what did X do
        after/before ...", "how many times does X appear", "who is X".
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
                    "Try vkg_overview to see the character registry, or vkg_search for the description.")

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

    def vkg_window(
        self,
        time_start: A[str, D("Window start as HH:MM:SS or seconds.")],
        time_end: A[str, D("Window end as HH:MM:SS or seconds.")],
    ) -> str:
        """
        Dump EVERYTHING the graph knows inside a time window, grouped by modality:
        scenes, actions/events, speech (with speakers), on-screen text (OCR),
        audio events, and entities present. This is the close-reading tool — use
        it once search/traversal has localized the relevant moment.
        """
        t0, t1 = to_seconds(time_start), to_seconds(time_end)
        if t1 <= t0:
            return f"time_end ({fmt(t1)}) must be after time_start ({fmt(t0)})."
        nodes = self.graph.get_nodes_in_window(t0, t1, buffer_sec=0.0)
        groups = {
            "Scenes/structure": [n for n in nodes if n.node_type in ("SceneNode", "EpisodeNode", "ClipNode")],
            "Actions & events": [n for n in nodes if n.node_type in ("ActionNode", "InteractionNode", "StateChangeNode")],
            "Speech": [n for n in nodes if n.node_type == "SpeechNode"],
            "On-screen text (OCR)": [n for n in nodes if n.node_type == "OCRNode"],
            "Audio events": [n for n in nodes if n.node_type == "AudioEventNode"],
            "Entities present": [n for n in nodes if n.node_type in ("CharacterNode", "ObjectNode")],
        }
        out = [f"WINDOW {fmt_span(t0, t1)} — {len(nodes)} nodes"]
        for gname, gnodes in groups.items():
            if not gnodes:
                continue
            gnodes.sort(key=lambda n: n.t_start)
            out.append(f"\n{gname} ({len(gnodes)}):")
            out += ["  " + node_line(self.graph, n) for n in gnodes[:30]]
            if len(gnodes) > 30:
                out.append(f"  … {len(gnodes) - 30} more")
        evented = groups["Actions & events"] + groups["Speech"]
        out.append(affordance_footer(self.graph, evented or nodes))
        return "\n".join(out)

    def vkg_infer_causal(
        self,
        time_start: A[str, D("Window start as HH:MM:SS or seconds.")],
        time_end: A[str, D("Window end as HH:MM:SS or seconds.")],
    ) -> str:
        """
        Infer causal edges ON THE FLY between events in a time window where the
        pre-computed graph has none, then cache them into the graph for the rest
        of the session. Use when vkg_causal/vkg_traverse(CAUSAL) reports no causal
        edges but the question is a why/how question.
        """
        if self.llm_complete is None:
            return ("Edge inference LLM is not configured for this session. "
                    "Fall back to vkg_window on the same span and reason over "
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
                "vkg_causal on any of the linked node ids will include them. "
                "Inferred edges carry model confidence; CONFIRM important ones with frame_inspect_tool.")
        return out

    # All tools, in registration order.
    def tools(self):
        return [
            self.vkg_overview, self.vkg_search, self.vkg_query,
            self.vkg_traverse, self.vkg_causal, self.vkg_entity,
            self.vkg_window, self.vkg_infer_causal,
        ]
