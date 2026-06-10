"""Smoke test: every GVD tool runs end-to-end on a synthetic graph, no LLM/GPU.

Run:  python3 -m gvd.tests.test_smoke
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gvd  # noqa: F401  (sets up sibling repo paths)
from qvkg.schema import VKGEdge, VKGNode, VKGraph

from gvd.vkg_tools import VKGToolkit
from gvd.dvd_compat import DVDCompatTools


def build_demo_graph() -> VKGraph:
    g = VKGraph()
    nodes = [
        VKGNode("ep_1", "EpisodeNode", "Morning at the house", 2, 0, 300,
                metadata={"narrative_role": "setup"}),
        VKGNode("sc_1", "SceneNode", "Kitchen argument", 1, 0, 120, parent_id="ep_1"),
        VKGNode("sc_2", "SceneNode", "Man leaves the house", 1, 120, 300, parent_id="ep_1"),
        VKGNode("ev_1", "ActionNode", "Woman shouts at the man about the unpaid bills", 0, 30, 45,
                confidence=0.9),
        VKGNode("ev_2", "ActionNode", "Man slams the kitchen door", 0, 50, 55, confidence=0.85),
        VKGNode("ev_3", "ActionNode", "Man walks out of the house carrying a suitcase", 0, 130, 150,
                confidence=0.8),
        VKGNode("sp_1", "SpeechNode", "\"I can't do this anymore\"", 0, 48, 50),
        VKGNode("ocr_1", "OCRNode", "FINAL NOTICE", 0, 32, 35, confidence=0.5),
        VKGNode("char_man", "CharacterNode", "the man", 0, 30, 150,
                entity_id="entity_man", canonical_description="middle-aged man in a gray jacket"),
        VKGNode("char_man_2", "CharacterNode", "the man (later)", 0, 130, 150,
                entity_id="entity_man"),
        VKGNode("char_woman", "CharacterNode", "the woman", 0, 30, 55,
                entity_id="entity_woman", canonical_description="woman in a red sweater"),
    ]
    g.add_nodes(nodes)
    g.add_edges([
        VKGEdge("ep_1", "sc_1", "CONTAINS", 1, 1),
        VKGEdge("ep_1", "sc_2", "CONTAINS", 1, 1),
        VKGEdge("ev_1", "ev_2", "CAUSES", 1, 0.8),
        VKGEdge("ev_2", "ev_3", "MOTIVATES", 1, 0.7),
        VKGEdge("ev_1", "ev_3", "PRECEDES", 1, 1),
        VKGEdge("char_man", "ev_2", "PERFORMS", 1, 0.9),
        VKGEdge("char_woman", "ev_1", "PERFORMS", 1, 0.9),
        VKGEdge("sp_1", "char_man", "SPOKEN_BY", 1, 0.8),
        VKGEdge("ocr_1", "ev_1", "LABELS", 1, 0.5),
    ])
    return g


def main():
    g = build_demo_graph()
    tk = VKGToolkit(g)

    print("=" * 70, "\nget_overview\n", "=" * 70)
    print(tk.get_overview())

    print("=" * 70, "\nsearch_events('man leaves house')\n", "=" * 70)
    print(tk.search_events("man leaves house"))

    print("=" * 70, "\nquery_nodes(SpeechNode)\n", "=" * 70)
    print(tk.query_nodes("SpeechNode"))

    print("=" * 70, "\nfollow_connections(ev_2, causal)\n", "=" * 70)
    print(tk.follow_connections("ev_2", "causal", hops=2))

    print("=" * 70, "\ntrace_causes(ev_3, why)\n", "=" * 70)
    print(tk.trace_causes("ev_3", direction="why"))

    print("=" * 70, "\nfind_entity('the man')\n", "=" * 70)
    print(tk.find_entity("the man"))

    print("=" * 70, "\nread_moment(0:00:25, 0:01:00)\n", "=" * 70)
    print(tk.read_moment("00:00:25", "00:01:00"))

    print("=" * 70, "\nexplain_why without LLM (graceful)\n", "=" * 70)
    print(tk.explain_why("0", "300"))

    # Schema generation over every tool (what the agent sends to the API).
    from gvd.func_schema import as_json_schema
    from gvd.agent import finish

    class _NoLLM:
        def complete(self, *a, **k):
            return "stub"

    dvd_tools = DVDCompatTools(tk, _NoLLM(), video_path=None)
    all_tools = tk.tools() + dvd_tools.tools() + [finish]
    schemas = [as_json_schema(t) for t in all_tools]
    print("=" * 70)
    print(f"Generated {len(schemas)} tool schemas: "
          + ", ".join(s["name"] for s in schemas))
    json.dumps(schemas)  # must be serializable

    # clip_search + frame_inspect graceful no-video path
    print(dvd_tools.inspect_frames([["00:00:30", "00:00:45"]], "test?")[:200])

    # Round-trip save/load
    tmp = "/tmp/gvd_demo_graph.json"
    g.save(tmp)
    from qvkg.schema import VKGraph
    g2 = VKGraph.load(tmp)
    assert len(g2.nodes) == len(g.nodes)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
