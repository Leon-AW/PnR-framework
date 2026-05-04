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

With ``--use_parallel`` the probe additionally drives
``ParallelOrchestrator.plan_query`` + ``select_adapters`` (no LLM forward
pass) and prints the candidate set + plan classification per query — this
is the unit-style probe used to verify Change 1 + Change 3 (per-chunk
anchors + similarity-distribution planner) without burning SLURM.

Run with:

    conda activate pnr
    python scripts/probe_router_routing.py
    python scripts/probe_router_routing.py --use_parallel
"""

from __future__ import annotations

import argparse
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

    cf = []
    with open("data/counterfact_train.jsonl") as f:
        for line in f:
            cf.append(json.loads(line)["question"])
            if len(cf) >= n:
                break
    examples["CF"] = cf

    with open("data/triviaqa_dcontrol.json") as f:
        data = json.load(f)
    examples["TriviaQA_OOD"] = [r["question"] for r in data["records"][:n]]

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


def probe_route(router: CentroidRouter, examples: dict[str, list[str]]) -> None:
    """Run the legacy `route()` probe (top-1 + Source-Replay)."""
    summary: dict[str, dict] = {}
    for category, queries in examples.items():
        print("=" * 90)
        print(f"[{category}]  ({len(queries)} queries)")
        print("=" * 90)
        winners: dict[str, int] = {}
        cf_sims: list[float | None] = []
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
                f"sim={sim if sim is None else f'{sim:.3f}'}  "
                f"cf_sim={cf_sim if cf_sim is None else f'{cf_sim:.3f}'}"
            )
        print()
        print(f"  Winner distribution: {winners}")
        cf_arr = [s for s in cf_sims if s is not None]
        if cf_arr:
            print(
                f"  CF-similarity stats (where above CF τ): n={len(cf_arr)} "
                f"min={min(cf_arr):.3f} max={max(cf_arr):.3f}"
            )
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
        print(
            f"  {cat:20s}  winners={s['winners']}  "
            f"n_above_cf_tau={s['n_above_cf_tau']}"
        )


def probe_parallel(router: CentroidRouter, examples: dict[str, list[str]]) -> None:
    """Drive `ParallelOrchestrator.plan_query` + `select_adapters` with a stub
    LLM. Runs no real generation — only the routing/planning layer.

    The stub mocks the bare interface that the orchestrator's `__init__`
    needs (a tokenizer attribute + `has_expert_attached`). Anything that
    would invoke `generate_text` is unreachable on this code path.
    """
    from src.routing.parallel_orchestrator import ParallelOrchestrator

    class _StubTokenizer:
        # PromptBuilder only needs apply_chat_template; we surface a minimal
        # passthrough that returns the user message verbatim.
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: D401, E501
            return messages[-1]["content"] if messages else ""

    class _StubLLM:
        tokenizer = _StubTokenizer()
        has_expert_attached = False

        def detach_expert(self) -> None:
            pass

        def load_expert(self, path: str) -> None:
            pass

        def get_inference_components(self):
            raise RuntimeError("stub LLM cannot generate")

    orch = ParallelOrchestrator(
        centroid_router=router,
        llm=_StubLLM(),
        query_planner_mode="similarity",
        max_adapters=5,
        synthesis_max_new_tokens=32,
        use_gpu=False,
    )

    summary: dict[str, dict] = {}
    for category, queries in examples.items():
        print("=" * 90)
        print(f"[{category} | parallel]  ({len(queries)} queries)")
        print("=" * 90)
        plan_counts: dict[str, int] = {}
        cand_counts: list[int] = []
        cf_in_cand = 0
        for q in queries:
            qe = router.compute_embedding(q)
            plan = orch.plan_query(q, query_embedding=qe)
            selected = orch.select_adapters(q, plan, query_embedding=qe)
            plan_counts[plan.plan_type.value] = (
                plan_counts.get(plan.plan_type.value, 0) + 1
            )
            cand_counts.append(len(selected))
            cand_ids = [m.adapter_id for m in selected]
            if "patch_cf_main" in cand_ids:
                cf_in_cand += 1
            print(
                f"  q={q[:60]:<60s} → plan={plan.plan_type.value:18s}  "
                f"|cand|={len(selected)}  cand={cand_ids}"
            )
        print()
        n = max(1, len(queries))
        print(f"  Plan distribution: {plan_counts}")
        print(
            f"  Avg candidates: {np.mean(cand_counts):.2f}  "
            f"max: {max(cand_counts) if cand_counts else 0}  "
            f"patch_cf_main hit rate: {cf_in_cand}/{len(queries)} "
            f"({100.0 * cf_in_cand / n:.1f}%)"
        )
        print()
        summary[category] = {
            "plans": plan_counts,
            "avg_candidates": float(np.mean(cand_counts)) if cand_counts else 0.0,
            "cf_main_hit_rate": cf_in_cand / n,
        }

    print("=" * 90)
    print("SUMMARY (parallel planner)")
    print("=" * 90)
    for cat, s in summary.items():
        print(
            f"  {cat:20s}  plans={s['plans']}  "
            f"avg_cand={s['avg_candidates']:.2f}  "
            f"cf_main_hit_rate={s['cf_main_hit_rate']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--use_parallel",
        action="store_true",
        help="Drive ParallelOrchestrator's planner + selector instead of route().",
    )
    parser.add_argument(
        "--router_state",
        default="checkpoints/router_state",
        help="Path to the router state directory (manifest + sidecars).",
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.45,
        help="Global fallback τ; per-adapter calibrated τ takes precedence.",
    )
    parser.add_argument("--n", type=int, default=10, help="Queries per category.")
    args = parser.parse_args()

    print("=" * 90)
    print(
        f"ROUTING SANITY PROBE — {args.router_state}"
        + ("  (parallel planner)" if args.use_parallel else "")
    )
    print("=" * 90)

    router = CentroidRouter.load(
        path=args.router_state,
        embedding_model_path=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        use_gpu=False,
    )
    print(f"Loaded {len(router.get_registered_adapters())} adapters\n")
    for aid in router.get_registered_adapters():
        entry = router._manifest[aid]
        tau = entry.metadata.get("similarity_threshold", "n/a")
        n_anchors = entry.num_clusters
        tau_str = tau if isinstance(tau, str) else f"{tau:.3f}"
        print(f"  {aid:30s} τ={tau_str:>6s}  #anchors={n_anchors}")
    print()

    examples = load_examples(n=args.n)

    if args.use_parallel:
        probe_parallel(router, examples)
    else:
        probe_route(router, examples)


if __name__ == "__main__":
    main()
