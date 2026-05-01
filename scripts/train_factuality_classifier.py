#!/usr/bin/env python3
"""Train the MORPHEUS factuality classifier.

Loads pre-built split data (train + val only — no test split; see
scripts/build_factuality_classifier_data.py for rationale), caches
embeddings upfront, trains an MLP with BCE loss and Adam, applies early
stopping on val AUC-ROC, and saves the best checkpoint.

The real test is downstream ESR + FR from the MORPHEUS D_eval sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.morpheus.factuality_classifier import FactualityClassifier


def _load_split(data: dict, split: str) -> tuple[list[str], np.ndarray]:
    records = data[split]
    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.float32)
    return texts, labels


def _cache_embeddings(
    classifier: FactualityClassifier,
    split_texts: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Embed all splits once. Embedding is the bottleneck; caching amortises it."""
    cached: dict[str, np.ndarray] = {}
    for split, texts in split_texts.items():
        print(f"  Embedding {split} ({len(texts)} samples) ...", flush=True)
        t0 = time.time()
        cached[split] = classifier._embed(texts)
        print(f"    done in {time.time() - t0:.1f}s", flush=True)
    return cached


def main() -> None:
    args = parse_args()

    with open(args.data_path) as f:
        data = json.load(f)

    meta = data["metadata"]
    print(
        f"Data: {meta['n_train']} train / {meta['n_val']} val "
        f"(pos_train={meta['pos_train']}, neg_train={meta['neg_train']})"
    )
    if "note" in meta:
        print(f"Note: {meta['note']}")

    train_texts, train_labels = _load_split(data, "train")
    val_texts, val_labels = _load_split(data, "val")

    classifier = FactualityClassifier(
        embedding_model_path=args.embedding_model,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )

    print("Pre-computing embeddings for all splits ...")
    emb_cache = _cache_embeddings(
        classifier,
        {"train": train_texts, "val": val_texts},
    )
    train_embs = emb_cache["train"]
    val_embs = emb_cache["val"]

    optimizer = torch.optim.Adam(classifier._mlp.parameters(), lr=args.lr)

    best_val_auc = -1.0
    patience_counter = 0
    out_dir = Path(args.output_dir)

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience}) ...")
    for epoch in range(1, args.epochs + 1):
        train_loss = classifier.train_epoch(
            train_embs, train_labels, optimizer, batch_size=args.batch_size
        )
        val_metrics = classifier.evaluate(val_embs, val_labels, batch_size=args.batch_size)
        val_auc = val_metrics["auc_roc"]

        print(
            f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
            f"val_auc={val_auc:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            classifier.save(out_dir)
            print(f"  -> checkpoint saved (val_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch} epochs (no val_auc improvement).")
                break

    print(f"\nLoading best checkpoint from {out_dir} ...")
    best = FactualityClassifier.load(out_dir, device="auto")

    val_metrics = best.evaluate(val_embs, val_labels, batch_size=args.batch_size)
    print("\n=== Best checkpoint val results ===")
    for k, v in val_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    from sklearn.metrics import classification_report
    val_scores = best._predict_from_embeddings(val_embs)
    val_preds = (val_scores >= 0.5).astype(int)
    print("\nClassification report (val):")
    print(classification_report(val_labels.astype(int), val_preds, digits=4))
    print("\nDownstream test = ESR + FR from MORPHEUS D_eval sweep.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/factuality_classifier_data.json")
    p.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p.add_argument(
        "--output_dir",
        default="checkpoints/factuality_classifier",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 64])
    p.add_argument("--patience", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    main()
