"""CLI entry point.

Build the graph first with qvkg (offline, once per video):
    python video-reasoning/qvkg/scripts/build_vkg.py --video movie.mp4 --out movie.vkg.json

Then ask questions (online, many times per video):
    python -m gvd.run_gvd --graph movie.vkg.json --video movie.mp4 \
        --question "Why did the man leave the house?"

Point GVD_BASE_URL / GVD_MODEL at a local vLLM server to run fully open-source.
"""

import argparse

from .agent import GVDAgent
from .llm import LLMClient


def main():
    p = argparse.ArgumentParser(description="GVD: graph-agentic video QA")
    p.add_argument("--graph", required=True, help="Path to VKG json (qvkg format)")
    p.add_argument("--video", default=None, help="Raw video file for frame inspection")
    p.add_argument("--faiss-index", default=None, help="Optional FAISS index path")
    p.add_argument("--question", required=True)
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--model", default=None, help="Orchestrator model (or GVD_MODEL env)")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (or GVD_BASE_URL env)")
    p.add_argument("--transcript", default=None, help="Optional path to dump the full message transcript as JSON")
    args = p.parse_args()

    agent = GVDAgent(
        graph_path=args.graph,
        video_path=args.video,
        faiss_index_path=args.faiss_index,
        llm=LLMClient(model=args.model, base_url=args.base_url),
        max_iterations=args.max_iterations,
    )
    answer, msgs = agent.run(args.question)
    print("\n" + "=" * 60)
    print(f"ANSWER: {answer}")
    print("=" * 60)

    if args.transcript:
        import json
        with open(args.transcript, "w") as f:
            json.dump(msgs, f, indent=2, default=str)
        print(f"Transcript written to {args.transcript}")


if __name__ == "__main__":
    main()
