#!/usr/bin/env python3
"""Open-stream leak MITIGATION eval — before/after the open-set veto (Task 4).

Runs the full production routing pipeline (Stage-1 classifier + centroid stage,
no LLM) twice on the same router object:

  * BEFORE — detector detached (== the production "before" behaviour verbatim)
  * AFTER  — OpenSetDetector attached as a veto on confident in-adapter-domain
             predictions

and reports the two numbers that matter, on disjoint data:

  1. OOD LEAK on the fresh held-out test set (``data/openstream_test_fresh.json``)
     — domains the detector was neither fitted nor calibrated on. The headline:
     how much of the ~31% routing leak does the veto remove?
  2. IN-DOMAIN RECALL COST on the real in-domain eval queries (CF test / SQA /
     QM conflict+stable) — of genuinely-routed in-domain queries, how many does
     the veto newly send to the frozen base (the ESR-loss proxy)?

Both run with no LLM (route() only), so this is fast and CPU-friendly. The
authoritative leak definition matches the diagnosis script: routing_leak =
route() returns a winner adapter.

Usage:
  python scripts/run_openstream_mitigation.py \
      --openset_detector checkpoints/openset_detector
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Reuse the diagnosis harness verbatim so routing is byte-identical.
from scripts.run_openstream_stress import (
    PROD,
    adapter_family,
    build_router_only,
    _attach_classifier,
    run_phase_a,
    summarize_phase_a,
)


def _attach_openset(router, path: str) -> None:
    from src.routing.openset_detector import OpenSetDetector
    det = OpenSetDetector.load(path, device="auto")
    router._openset_detector = det
    print(f"  attached OpenSetDetector ({det.method}, classes={det.classes_}, "
          f"per-class τ={ {k: round(v,1) for k,v in det.thresholds_.items()} })",
          flush=True)


# ---------------------------------------------------------------------------
# In-domain query collection (for the recall-cost arm)
# ---------------------------------------------------------------------------

def _collect_indomain(n_per_domain: int, rng: random.Random) -> list[dict]:
    """Genuine in-domain queries that *should* route to an adapter.

    cf  → counterfact_eval.json["test"]   (held out from training)
    sqa → sqa_deval.json                  (the SQA eval set)
    qm  → qm_deval.json["conflict"]+["stable"]  (control/TriviaQA excluded)
    """
    out: list[dict] = []

    cf = json.load(open("data/counterfact_eval.json"))["test"]
    cf_q = [r["question"] for r in cf if r.get("question")]
    rng.shuffle(cf_q)
    out += [{"text": q, "domain": "cf"} for q in cf_q[:n_per_domain]]

    sqa = json.load(open("data/sqa_deval.json"))
    sqa_q = [r["question"] for r in sqa if r.get("question")]
    rng.shuffle(sqa_q)
    out += [{"text": q, "domain": "sqa"} for q in sqa_q[:n_per_domain]]

    qm = json.load(open("data/qm_deval.json"))
    qm_q = [r["question"] for r in (qm["conflict"] + qm["stable"]) if r.get("question")]
    rng.shuffle(qm_q)
    out += [{"text": q, "domain": "qm"} for q in qm_q[:n_per_domain]]

    return out


def _route_winners(router, records: list[dict]) -> list[str | None]:
    return [router.route(r["text"]).winner_adapter for r in records]


def _leak_summary(rows: list[dict]) -> dict:
    """Headline routing-leak numbers from a Phase-A row list."""
    s = summarize_phase_a(rows, PROD["domain_confidence_threshold"])
    return {
        "overall": s["overall"]["routing_leak"],
        "overall_count": s["overall"]["routing_leak_count"],
        "per_domain": {d: v["routing_leak"] for d, v in s["per_domain"].items()},
        "by_family": s["overall"]["routing_leak_by_family"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openset_detector", default="checkpoints/openset_detector")
    p.add_argument("--ood_test", default="data/openstream_test_fresh.json")
    p.add_argument("--output_dir", default="eval_results/openstream_mitigation")
    p.add_argument("--indomain_n_per_domain", type=int, default=500)
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    use_gpu = (not args.no_gpu) and torch.cuda.is_available()
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ood = json.load(open(args.ood_test))["records"]
    indom = _collect_indomain(args.indomain_n_per_domain, rng)
    print(f"OOD test: {len(ood)} | in-domain: {len(indom)}", flush=True)

    # Build router + classifier ONCE; toggle the detector between passes.
    router = build_router_only(use_gpu)
    _attach_classifier(router)

    # ---- BEFORE (detector detached) ----
    router._openset_detector = None
    print("\n=== BEFORE (no detector) ===", flush=True)
    ood_before = run_phase_a(router, ood)
    indom_before = _route_winners(router, indom)

    # ---- AFTER (detector attached) ----
    print("\n=== AFTER (open-set veto) ===", flush=True)
    _attach_openset(router, args.openset_detector)
    ood_after = run_phase_a(router, ood)
    indom_after = _route_winners(router, indom)

    # ---- Leak before/after on the fresh OOD test ----
    leak_before = _leak_summary(ood_before)
    leak_after = _leak_summary(ood_after)

    # ---- In-domain recall cost: routed-before that the veto sent to base ----
    domains = sorted({r["domain"] for r in indom})
    recall_cost = {}
    for d in domains:
        idx = [i for i, r in enumerate(indom) if r["domain"] == d]
        routed_before = [i for i in idx if indom_before[i] is not None]
        newly_base = [i for i in routed_before if indom_after[i] is None]
        recall_cost[d] = {
            "routed_before": len(routed_before),
            "vetoed_to_base": len(newly_base),
            "recall_cost": (len(newly_base) / len(routed_before)) if routed_before else 0.0,
        }
    all_routed = [i for i in range(len(indom)) if indom_before[i] is not None]
    all_newly = [i for i in all_routed if indom_after[i] is None]
    recall_cost["overall"] = {
        "routed_before": len(all_routed),
        "vetoed_to_base": len(all_newly),
        "recall_cost": (len(all_newly) / len(all_routed)) if all_routed else 0.0,
    }

    report = {
        "config": {**PROD, "openset_detector": args.openset_detector,
                   "indomain_n_per_domain": args.indomain_n_per_domain},
        "ood_leak_before": leak_before,
        "ood_leak_after": leak_after,
        "indomain_recall_cost": recall_cost,
    }
    with (out_dir / "mitigation_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    # Per-OOD-record predictions, for auditing.
    with (out_dir / "ood_predictions.jsonl").open("w") as f:
        for b, a in zip(ood_before, ood_after):
            f.write(json.dumps({
                "id": b["id"], "domain": b["domain"], "text": b["text"][:200],
                "top_class": b["top_class"], "top_prob": b["top_prob"],
                "winner_before": b["winner_adapter"], "winner_after": a["winner_adapter"],
            }) + "\n")

    # ---- Console summary ----
    print("\n" + "=" * 64)
    print("OOD ROUTING LEAK on fresh held-out test (lower = better)")
    print("=" * 64)
    print(f"  overall:  {leak_before['overall']:.1%}  →  {leak_after['overall']:.1%}  "
          f"({leak_before['overall_count']} → {leak_after['overall_count']} of {len(ood)})")
    for d in sorted(leak_before["per_domain"]):
        print(f"    {d:>12}: {leak_before['per_domain'][d]:.1%}  →  "
              f"{leak_after['per_domain'][d]:.1%}")
    print(f"  leaked-into family before: {leak_before['by_family']}")
    print(f"  leaked-into family after : {leak_after['by_family']}")

    print("\n" + "=" * 64)
    print("IN-DOMAIN RECALL COST (genuine in-domain queries newly sent to base)")
    print("=" * 64)
    for d in domains + ["overall"]:
        rc = recall_cost[d]
        print(f"  {d:>8}: {rc['recall_cost']:.1%} "
              f"({rc['vetoed_to_base']}/{rc['routed_before']} routed-before → base)")
    print(f"\n  → {out_dir/'mitigation_report.json'}")


if __name__ == "__main__":
    main()
