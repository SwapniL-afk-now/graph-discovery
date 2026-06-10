"""Observation serializer: graph evidence → text the orchestrator can act on.

Design principles (the "present so the model knows what to do next" layer):

1. **One node = one line**, always carrying the node id, type, time span and
   confidence — everything a follow-up tool call needs as arguments.
2. **Edges are shown inline as affordances** (`—CAUSES→ ev_15`), so the model
   sees *which* traversals are possible from each piece of evidence.
3. **Relevance-then-chronology**: hits are capped by relevance first, then
   printed in chronological order (cap-before-sort), preserving narrative
   readability without losing precision.
4. **Affordance footer**: every observation ends with a short "What you can do
   next" block of *concrete, copy-pasteable* tool calls derived from the
   evidence actually returned (causal edges present → vkg_causal; entities
   present → vkg_entity; uncertain/visual detail → frame_inspect_tool).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List

from .timeutil import fmt, fmt_span

# Edge families surfaced inline next to nodes, in display priority order.
_INLINE_EDGE_PRIORITY = [
    "CAUSES", "ENABLES", "PREVENTS", "MOTIVATES",
    "PERFORMS", "INTERACTS_WITH", "SPOKEN_BY", "SAME_ENTITY",
    "DESCRIBES", "MENTIONS", "LABELS",
    "PRECEDES", "CONTAINS",
]
_CAUSAL = {"CAUSES", "ENABLES", "PREVENTS", "MOTIVATES"}

MAX_INLINE_EDGES = 4


def _short_label(node, max_len: int = 110) -> str:
    label = (node.label or "").strip().replace("\n", " ")
    return label[: max_len - 1] + "…" if len(label) > max_len else label


def edge_affordances(graph, node, max_edges: int = MAX_INLINE_EDGES) -> str:
    """Inline edge summary for one node, e.g. '—CAUSES→ev_15, ←PERFORMS—char_3'."""
    pairs = []  # (priority, text)
    for e in graph.get_edges(node.id):
        prio = (_INLINE_EDGE_PRIORITY.index(e.relation_type)
                if e.relation_type in _INLINE_EDGE_PRIORITY else 99)
        pairs.append((prio, f"—{e.relation_type}→{e.target_id}"))
    for e in graph.get_incoming_edges(node.id):
        prio = (_INLINE_EDGE_PRIORITY.index(e.relation_type)
                if e.relation_type in _INLINE_EDGE_PRIORITY else 99)
        pairs.append((prio, f"←{e.relation_type}—{e.source_id}"))
    if not pairs:
        return ""
    pairs.sort(key=lambda p: p[0])
    shown = [t for _, t in pairs[:max_edges]]
    extra = len(pairs) - len(shown)
    tail = f", +{extra} more edges" if extra > 0 else ""
    return " [" + ", ".join(shown) + tail + "]"


def node_line(graph, node, with_edges: bool = True) -> str:
    """One-line rendering of a node with everything needed to act on it."""
    conf = f" conf={node.confidence:.2f}" if node.confidence < 0.99 else ""
    ent = f" entity={node.entity_id}" if node.entity_id else ""
    line = (f"({node.id} | {node.node_type} | {fmt_span(node.t_start, node.t_end)}"
            f"{conf}{ent}) {_short_label(node)}")
    if with_edges:
        line += edge_affordances(graph, node)
    return line


def _causal_degree(graph, node) -> int:
    n = sum(1 for e in graph.get_edges(node.id) if e.relation_type in _CAUSAL)
    n += sum(1 for e in graph.get_incoming_edges(node.id) if e.relation_type in _CAUSAL)
    return n


def affordance_footer(graph, nodes: List, question_hint: str = "") -> str:
    """Concrete next-step suggestions derived from the returned evidence."""
    if not nodes:
        return ""
    tips: List[str] = []

    # 1. Causal follow-ups: point at the node with the most causal edges.
    causal_nodes = sorted(nodes, key=lambda n: -_causal_degree(graph, n))
    if causal_nodes and _causal_degree(graph, causal_nodes[0]) > 0:
        n = causal_nodes[0]
        tips.append(
            f'{n.id} has {_causal_degree(graph, n)} causal edge(s) → '
            f'vkg_causal(node_id="{n.id}") to trace why it happened / what it led to'
        )

    # 2. Entity follow-ups: most frequent entity among the hits.
    ents = Counter(n.entity_id for n in nodes if n.entity_id)
    if ents:
        eid, cnt = ents.most_common(1)[0]
        total = len(graph.entity_idx.get(eid, []))
        tips.append(
            f'entity {eid} appears in {cnt} hit(s) ({total} appearances overall) → '
            f'vkg_entity(name="{eid}") for its full timeline'
        )

    # 3. Dense window: where evidence clusters in time.
    times = sorted(n.t_start for n in nodes)
    if len(times) >= 3:
        mid = times[len(times) // 2]
        tips.append(
            f'evidence clusters around {fmt(mid)} → '
            f'vkg_window(time_start="{fmt(max(0, mid - 60))}", time_end="{fmt(mid + 60)}") '
            f'for everything (speech, OCR, actions) in that window'
        )

    # 4. Visual confirmation (DVD's CONFIRM rule).
    low_conf = [n for n in nodes if n.confidence < 0.6]
    target = low_conf[0] if low_conf else min(nodes, key=lambda n: n.t_start)
    reason = ("a low-confidence node" if low_conf
              else "the answer before calling finish")
    tips.append(
        f'to visually verify {reason} → '
        f'frame_inspect_tool(time_ranges=[["{fmt(target.t_start)}", "{fmt(target.t_end)}"]], '
        f'question="...")'
    )

    out = "\n— What you can do next —\n" + "\n".join(f"• {t}" for t in tips)
    if question_hint:
        out += f"\n• {question_hint}"
    return out


def serialize_nodes(
    graph,
    nodes: Iterable,
    title: str,
    max_nodes: int = 25,
    footer: bool = True,
    question_hint: str = "",
) -> str:
    """Render a node set: cap by given (relevance) order, then sort by time."""
    nodes = list(nodes)
    if not nodes:
        return (f"{title}\n(no matching nodes found)\n"
                "— What you can do next (do NOT just re-run search with a shorter query) —\n"
                "• if you know the time → vkg_window(time_start, time_end) reads "
                "everything (speech, OCR, actions, entities) in that span\n"
                "• to enumerate a category → vkg_query(node_type=\"SpeechNode\"/"
                "\"OCRNode\"/\"ActionNode\", time_start, time_end)\n"
                "• for a fine visual detail the graph may not capture → "
                "frame_inspect_tool on the suspected time range to look at raw frames")
    capped = nodes[:max_nodes]
    capped.sort(key=lambda n: n.t_start)
    lines = [node_line(graph, n) for n in capped]
    body = f"{title} ({len(capped)} of {len(nodes)} nodes, chronological):\n"
    body += "\n".join(lines)
    if len(nodes) > max_nodes:
        body += f"\n… {len(nodes) - max_nodes} more matches truncated (refine the query to see them)."
    if footer:
        body += "\n" + affordance_footer(graph, capped, question_hint)
    return body


def serialize_chain(graph, chain_edges: List, title: str) -> str:
    """Render a causal chain as an indented arrow list."""
    if not chain_edges:
        return (f"{title}\n(no causal edges found here)\n"
                "— What you can do next —\n"
                "• vkg_infer_causal on this time window to infer causal links on the fly\n"
                "• vkg_window on the same span to read raw events in temporal order")
    lines = [title]
    for e in chain_edges:
        src, tgt = graph.nodes.get(e.source_id), graph.nodes.get(e.target_id)
        if not src or not tgt:
            continue
        lines.append(
            f"  ({e.source_id} @{fmt(src.t_start)}) \"{_short_label(src, 60)}\" "
            f"—{e.relation_type} (conf={e.confidence:.2f})→ "
            f"({e.target_id} @{fmt(tgt.t_start)}) \"{_short_label(tgt, 60)}\""
        )
    return "\n".join(lines)
