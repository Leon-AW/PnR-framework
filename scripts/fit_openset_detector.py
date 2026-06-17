#!/usr/bin/env python3
"""Fit + calibrate the Stage-1 open-set detector (leak mitigation, Task 2/3).

Pipeline (CPU, no GPU needed)::

  D_fit   = domain_classifier_data_4class.json["train"]  (cf/sqa/qm rows)
  D_cal   = domain_classifier_data_4class.json["val"]    (cf/sqa/qm rows)
  fit manifolds on D_fit  →  calibrate τ_ood on D_cal at ≤ alpha false-reject

The threshold is set ONLY from the in-domain validation split. The fresh OOD
test set (``data/openstream_test_fresh.json``) is scored afterwards purely as a
read-only preview — the threshold is already frozen, so this is held-out
evaluation, not tuning. The authoritative leak number still comes from running
the full router pipeline (Task 4); this preview just confirms the score
separates in-domain from OOD before we wire it into the router.

Validity: D_fit / D_cal / OOD-test are mutually disjoint. ``alpha`` is the one
pre-committed degree of freedom and bounds the in-domain recall cost directly.

Usage:
  python scripts/fit_openset_detector.py --alpha 0.05
  python scripts/fit_openset_detector.py --method knn --alpha 0.05
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ADAPTER_LABELS = {0: "cf", 1: "sqa", 2: "qm"}  # ood_trivia (3) excluded by design


def _split_rows(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Return (texts, class_names) for rows whose label is an adapter class."""
    texts, classes = [], []
    for r in rows:
        lab = r["label"]
        if lab in ADAPTER_LABELS:
            texts.append(r["text"])
            classes.append(ADAPTER_LABELS[lab])
    return texts, classes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/domain_classifier_data_4class.json")
    p.add_argument("--ood_test", default="data/openstream_test_fresh.json")
    p.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--method", choices=["mahalanobis", "knn"], default="mahalanobis")
    p.add_argument("--knn_k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="in-domain false-reject budget (pre-committed). One DoF.")
    p.add_argument("--output", default="/vol/tmp/wagnerql/checkpoints/openset_detector")
    p.add_argument("--symlink", default="checkpoints/openset_detector")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    from src.routing.openset_detector import OpenSetDetector

    data = json.load(open(args.data))
    fit_texts, fit_classes = _split_rows(data["train"])
    cal_texts, cal_classes = _split_rows(data["val"])
    print(f"D_fit: {len(fit_texts)} rows {dict(Counter(fit_classes))}")
    print(f"D_cal: {len(cal_texts)} rows {dict(Counter(cal_classes))}")

    det = OpenSetDetector(
        embedding_model_path=args.embedding_model,
        method=args.method,
        knn_k=args.knn_k,
        device=args.device,
    )

    # Embed once; reuse for fit + calibrate + per-class reporting.
    print("Embedding D_fit ...", flush=True)
    fit_embs = det._embed(fit_texts)
    print("Embedding D_cal ...", flush=True)
    cal_embs = det._embed(cal_texts)

    det.fit(fit_embs, fit_classes)
    det.calibrate(cal_embs, alpha=args.alpha)            # global bar (ablation)
    if args.method == "mahalanobis":
        taus = det.calibrate_per_class(cal_embs, cal_classes, alpha=args.alpha)
        print(f"\nmethod={args.method}  classes={det.classes_}  alpha={args.alpha}")
        print(f"  per-class τ_ood: " + "  ".join(f"{c}={t:.1f}" for c, t in taus.items()))
        print(f"  global τ_ood (ablation): {det.threshold:.1f}")
    else:
        print(f"\nmethod={args.method}  alpha={args.alpha}  global τ_ood = {det.threshold:.4f}")

    # In-domain false-reject on D_cal — the recall-cost proxy. Per-class
    # calibration targets ≤ alpha for EACH class (incl. qm), not just overall.
    cal_rejected = det.is_ood_embeddings(cal_embs)       # standalone nearest-manifold rule
    print(f"\nIn-domain false-reject on D_cal (recall cost proxy):")
    print(f"  overall: {cal_rejected.mean():.1%} ({cal_rejected.sum()}/{len(cal_rejected)})")
    for c in det.classes_:
        m = np.array([cc == c for cc in cal_classes])
        print(f"  {c:>4}: {cal_rejected[m].mean():.1%} ({cal_rejected[m].sum()}/{m.sum()})")
    cal_scores = det.score_embeddings(cal_embs)

    # Read-only OOD preview (threshold already frozen). Held-out evaluation.
    ood_path = Path(args.ood_test)
    if ood_path.exists():
        ood = json.load(open(ood_path))["records"]
        ood_texts = [r["text"] for r in ood]
        ood_doms = [r["domain"] for r in ood]
        print("\nEmbedding fresh OOD test (read-only preview) ...", flush=True)
        ood_embs = det._embed(ood_texts)
        ood_scores = det.score_embeddings(ood_embs)
        flagged = det.is_ood_embeddings(ood_embs)        # nearest-manifold per-class rule
        print(f"OOD detection rate @ frozen τ (higher = more leaks caught):")
        print(f"  overall: {flagged.mean():.1%} ({flagged.sum()}/{len(flagged)})")
        for d in sorted(set(ood_doms)):
            m = np.array([dd == d for dd in ood_doms])
            print(f"  {d:>12}: {flagged[m].mean():.1%} ({flagged[m].sum()}/{m.sum()})  "
                  f"median score {np.median(ood_scores[m]):.3f}")
        print(f"\n  (in-domain D_cal median score {np.median(cal_scores):.3f} — "
              f"OOD medians should sit well above)")
    else:
        print(f"\n(OOD preview skipped — {ood_path} not found)")

    det.save(args.output)
    out = Path(args.output)
    link = Path(args.symlink)
    if args.symlink:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(out.resolve())
    print(f"\nSaved detector → {out}" + (f"  (symlink {link} → {out.resolve()})" if args.symlink else ""))


if __name__ == "__main__":
    main()
