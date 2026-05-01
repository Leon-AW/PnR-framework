#!/usr/bin/env python3
"""
Routing sanity probe
====================

Loads the freshly-built router state and runs a handful of representative
queries (CF, SQA temporal, geo, OOD TriviaQA) through `route()`. Prints the
top-1 adapter + similarity + per-adapter best similarity, so we can verify:

  - CF queries route to `patch_cf_main` more often than they used to,
  - TriviaQA D_control queries route to `<frozen base>` (no winner),
  - Geo queries still route to the right country adapter.

Run with:

    conda activate pnr
    python scripts/probe_router_routing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.routing import CentroidRouter


# ---------------------------------------------------------------------------
# Probe queries
# ---------------------------------------------------------------------------

def load_examples(n: int = 10) -> dict[str, list[str]]:
    """Sample probe queries from the existing data files (no GPU needed)."""
    examples: dict[str, list[str]] = {}

    # CF: read the first n questions from counterfact_train.jsonl
    cf = []
    with open("data/counterfact_train.jsonl") as f:
        for line in f:
            cf.append(json.loads(line)["question"])
            if len(cf) >= n:
                break
    examples["CF"] = cf

    # OOD: D_control TriviaQA (must NOT route to patch_cf_main).
    with open("data/triviaqa_dcontrol.json") as f:
        data = json.load(f)
    examples["TriviaQA_OOD"] = [r["question"] for r in data["records"][:n]]

    # Geo + temporal: synthetic queries in the SQA mould.
    examples["Geo_Germany"] = [
        "Who is the chancellor of Germany?",
        "What is the capital of Germany?",
        "Who is the president of Germany as of 2023?",
        "Who leads the German government?",
        "What is Germany's official language?",
    ]
    examples["Geo_India"] = [
        "Who is the prime minister of India?",
        "What is the capital of India?",
        "Who heads the Indian government as of 2024?",
        "Which language is most spoken in India?",
        "Who is the president of India?",
    ]
    examples["Temporal"] = [
        "Who was the US president as of 2021?",
        "Who is the British PM as of 2023?",
        "What was the highest-grossing film of 2022?",
        "Who won the FIFA World Cup in 2022?",
        "Who is the CEO of OpenAI as of 2023?",
    ]
    return examples


def main() -> None:
    print("=" * 90)
    print("ROUTING SANITY PROBE — checkpoints/router_state/")
    print("=" * 90)

    router = CentroidRouter.load(
        path="checkpoints/router_state",
        embedding_model_path="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.45,
        use_gpu=False,
    )
    print(f"Loaded {len(router.get_registered_adapters())} adapters\n")
    for aid in router.get_registered_adapters():
        entry = router._manifest[aid]
        tau = entry.metadata.get("similarity_threshold", "n/a")
        n_anchors = entry.num_clusters
        print(f"  {aid:30s} τ={tau if isinstance(tau, str) else f'{tau:.3f}':>6s}  "
              f"#anchors={n_anchors}")
    print()

    examples = load_examples(n=10)

    summary: dict[str, dict] = {}
    for category, queries in examples.items():
        print("=" * 90)
        print(f"[{category}]  ({len(queries)} queries)")
        print("=" * 90)
        winners: dict[str, int] = {}
        cf_sims = []
        for q in queries:
            result = router.route(q)
            winner = result.winner_adapter or "<NONE>"
            winners[winner] = winners.get(winner, 0) + 1
            sim = (
                next((m.similarity for m in result.all_matches if m.is_winner), None)
                if result.winner_adapter
                else None
            )
            cf_sim = next(
                (m.similarity for m in result.all_matches if m.adapter_id == "patch_cf_main"),
                None,
            )
            cf_sims.append(cf_sim)
            print(
                f"  q={q[:60]:<60s} → winner={winner:25s}  "
                f"sim={sim if sim is None else f'{sim:.3f}'}  cf_sim={cf_sim if cf_sim is None else f'{cf_sim:.3f}'}"
            )
        print()
        print(f"  Winner distribution: {winners}")
        cf_arr = [s for s in cf_sims if s is not None]
        if cf_arr:
            print(f"  CF-similarity stats (where above CF τ): n={len(cf_arr)} "
                  f"min={min(cf_arr):.3f} max={max(cf_arr):.3f}")
        print()
        summary[category] = {
            "winners": winners,
            "n_above_cf_tau": len(cf_arr),
            "cf_sim_max": max(cf_arr) if cf_arr else None,
        }

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    for cat, s in summary.items():
        print(f"  {cat:20s}  winners={s['winners']}  n_above_cf_tau={s['n_above_cf_tau']}")


if __name__ == "__main__":
    main()
