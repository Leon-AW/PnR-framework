#!/usr/bin/env python3
"""Leak-vs-recall sweep over the open-set false-reject budget alpha (Task 4, sensitivity).

The pre-committed headline is alpha=5%. This sweep shows the trade-off curve so
that choice is justified rather than asserted.

Efficient design: the veto fires iff a query is confident-in-adapter-domain,
routes to a winner, and its Mahalanobis distance to the *predicted* class exceeds
that class's threshold tau_c(alpha). The distances do NOT depend on alpha — only
the threshold does. So we run the production pipeline ONCE (record winner +
Stage-1 class + distance-to-predicted-class per query), then evaluate every alpha
analytically by re-quantiling D_cal. This reproduces the alpha=5% point from
run_openstream_mitigation.py exactly (sanity check).

Usage:
  python scripts/sweep_openset_alpha.py --openset_detector checkpoints/openset_detector
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from scripts.run_openstream_stress import (
    PROD,
    adapter_family,
    build_router_only,
    _attach_classifier,
)
from scripts.run_openstream_mitigation import _collect_indomain

ADAPTER_LABELS = {0: "cf", 1: "sqa", 2: "qm"}
TRIVIA_ADJACENT = "german_rc"  # not trivia-adjacent here; kept name for clarity unused
ENGLISH_OOD = ("math", "professional")


def _before_rows(router, records: list[dict]) -> list[dict]:
    """One pass: per query record winner, Stage-1 top_class/prob."""
    rows = []
    for r in records:
        top_class, top_prob, _ = router._classify_domain(r["text"])
        winner = router.route(r["text"]).winner_adapter
        rows.append({
            "text": r["text"], "domain": r["domain"],
            "top_class": top_class, "top_prob": float(top_prob),
            "winner_before": winner,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openset_detector", default="checkpoints/openset_detector")
    p.add_argument("--data", default="data/domain_classifier_data_4class.json")
    p.add_argument("--ood_test", default="data/openstream_test_fresh.json")
    p.add_argument("--output_dir", default="eval_results/openstream_mitigation")
    p.add_argument("--alphas", default="0.01,0.02,0.05,0.10,0.15")
    p.add_argument("--indomain_n_per_domain", type=int, default=500)
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    from src.routing.openset_detector import OpenSetDetector
    use_gpu = (not args.no_gpu) and torch.cuda.is_available()
    rng = random.Random(args.seed)
    alphas = [float(a) for a in args.alphas.split(",")]

    det = OpenSetDetector.load(args.openset_detector, device="auto")
    conf_thr = PROD["domain_confidence_threshold"]

    # --- One production pass (no detector) over OOD + in-domain ---
    router = build_router_only(use_gpu)
    _attach_classifier(router)
    router._openset_detector = None

    ood = json.load(open(args.ood_test))["records"]
    indom = _collect_indomain(args.indomain_n_per_domain, rng)
    print(f"OOD test: {len(ood)} | in-domain: {len(indom)}  — single before-pass ...",
          flush=True)
    ood_rows = _before_rows(router, [{"text": r["text"], "domain": r["domain"]} for r in ood])
    ind_rows = _before_rows(router, indom)

    # --- Distance to the Stage-1-predicted class for every query (alpha-free) ---
    def dist_to_pred(rows):
        embs = det._embed([r["text"] for r in rows])
        dmat = det.class_distances(embs)                 # (N, C)
        col = {c: j for j, c in enumerate(det.classes_)}
        d = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            c = r["top_class"]
            if c in col:
                d[i] = dmat[i, col[c]]
        return d
    ood_d = dist_to_pred(ood_rows)
    ind_d = dist_to_pred(ind_rows)

    # --- D_cal own-class distances, for per-alpha thresholds ---
    data = json.load(open(args.data))
    cal_texts, cal_classes = [], []
    for r in data["val"]:
        if r["label"] in ADAPTER_LABELS:
            cal_texts.append(r["text"]); cal_classes.append(ADAPTER_LABELS[r["label"]])
    cal_dmat = det.class_distances(det._embed(cal_texts))
    col = {c: j for j, c in enumerate(det.classes_)}
    cal_own = {c: cal_dmat[[i for i, lab in enumerate(cal_classes) if lab == c], col[c]]
               for c in det.classes_}

    def vetoed(rows, dists, taus):
        """Boolean per row: confident in-domain, routed, dist > tau_pred."""
        out = np.zeros(len(rows), dtype=bool)
        for i, r in enumerate(rows):
            c = r["top_class"]
            if (r["top_prob"] >= conf_thr and c in taus
                    and r["winner_before"] is not None and not np.isnan(dists[i])):
                out[i] = dists[i] > taus[c]
        return out

    # --- Baselines (before) ---
    def leak(rows, mask_keep):
        idx = [i for i in range(len(rows)) if mask_keep(rows[i])]
        if not idx:
            return 0.0, 0, 0
        n = len(idx)
        leaked = sum(1 for i in idx if rows[i]["winner_before"] is not None)
        return leaked / n, leaked, n
    ood_before_overall = leak(ood_rows, lambda r: True)
    ood_before_en = leak(ood_rows, lambda r: r["domain"] in ENGLISH_OOD)
    ood_before_de = leak(ood_rows, lambda r: r["domain"] == "german_rc")

    rows_out = []
    print("\nalpha | OOD leak after (overall / EN-only / DE) | in-dom recall cost")
    print(f"  before: {ood_before_overall[0]:.1%} / {ood_before_en[0]:.1%} / {ood_before_de[0]:.1%}  | 0.0%")
    for a in alphas:
        taus = {c: float(np.quantile(cal_own[c], 1.0 - a)) for c in det.classes_}
        ov = vetoed(ood_rows, ood_d, taus)
        iv = vetoed(ind_rows, ind_d, taus)

        def leak_after(rows, veto, keep):
            idx = [i for i in range(len(rows)) if keep(rows[i])]
            if not idx:
                return 0.0
            return sum(1 for i in idx
                       if rows[i]["winner_before"] is not None and not veto[i]) / len(idx)
        la_ov = leak_after(ood_rows, ov, lambda r: True)
        la_en = leak_after(ood_rows, ov, lambda r: r["domain"] in ENGLISH_OOD)
        la_de = leak_after(ood_rows, ov, lambda r: r["domain"] == "german_rc")

        routed = [i for i in range(len(ind_rows)) if ind_rows[i]["winner_before"] is not None]
        rc = (sum(1 for i in routed if iv[i]) / len(routed)) if routed else 0.0

        rows_out.append({"alpha": a, "tau": taus,
                         "ood_leak_after_overall": la_ov,
                         "ood_leak_after_english": la_en,
                         "ood_leak_after_german": la_de,
                         "indomain_recall_cost": rc})
        print(f"  {a:>4.0%}: {la_ov:.1%} / {la_en:.1%} / {la_de:.1%}  | {rc:.1%}")

    report = {
        "config": {**PROD, "openset_detector": args.openset_detector, "alphas": alphas},
        "ood_leak_before": {"overall": ood_before_overall[0],
                            "english": ood_before_en[0], "german": ood_before_de[0]},
        "sweep": rows_out,
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "alpha_sweep.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  → {out/'alpha_sweep.json'}")


if __name__ == "__main__":
    main()
