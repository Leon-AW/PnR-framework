#!/usr/bin/env python3
"""Train the 3-class domain classifier (Phase 4 Stage 1 of the router).

Loads pre-built split data from ``scripts/build_domain_classifier_data.py``,
caches embeddings upfront (the bottleneck), trains a 3-way MLP with
CrossEntropyLoss + Adam, applies early stopping on val macro-F1, and
saves the best checkpoint.

Downstream test = SQA ``routing_acc`` and TriviaQA D_control FR from the
PnR/Parallel D_eval sweep (Phase 5).
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

from src.routing.domain_classifier import DomainClassifier, CLASS_LABELS


def _load_split(data: dict, split: str) -> tuple[list[str], np.ndarray]:
    records = data[split]
    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int64)
    return texts, labels


def _cache_embeddings(
    classifier: DomainClassifier,
    split_texts: dict[str, list[str]],
) -> dict[str, np.ndarray]:
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
    saved_labels = meta.get("labels", CLASS_LABELS)
    if saved_labels != CLASS_LABELS:
        raise ValueError(
            f"Data file {args.data_path} declares labels {saved_labels}, but "
            f"DomainClassifier expects {CLASS_LABELS}. Rebuild the data or "
            "align the label list."
        )

    print(
        f"Data: {meta['n_train']} train / {meta['n_val']} val | "
        f"class_counts_train={meta.get('class_counts_train')}"
    )
    if "note" in meta:
        print(f"Note: {meta['note']}")

    train_texts, train_labels = _load_split(data, "train")
    val_texts, val_labels = _load_split(data, "val")

    classifier = DomainClassifier(
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

    best_val_f1 = -1.0
    patience_counter = 0
    out_dir = Path(args.output_dir)

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience}) ...")
    for epoch in range(1, args.epochs + 1):
        train_loss = classifier.train_epoch(
            train_embs, train_labels, optimizer, batch_size=args.batch_size
        )
        val_metrics = classifier.evaluate(val_embs, val_labels, batch_size=args.batch_size)
        val_f1 = val_metrics["f1_macro"]

        print(
            f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1_macro={val_f1:.4f} | "
            f"per_class_f1={val_metrics['per_class_f1']}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            classifier.save(out_dir)
            print(f"  -> checkpoint saved (val_f1_macro={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch} epochs (no val_f1 improvement).")
                break

    print(f"\nLoading best checkpoint from {out_dir} ...")
    best = DomainClassifier.load(out_dir, device="auto")

    val_metrics = best.evaluate(val_embs, val_labels, batch_size=args.batch_size)
    print("\n=== Best checkpoint val results ===")
    for k, v in val_metrics.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k:18s}: {v:.4f}")

    from sklearn.metrics import classification_report, confusion_matrix
    val_preds = val_metrics["_predictions"]
    print("\nClassification report (val):")
    print(classification_report(val_labels, val_preds, target_names=CLASS_LABELS, digits=4))
    print("Confusion matrix (rows=true, cols=pred), order=" + ", ".join(CLASS_LABELS) + ":")
    print(confusion_matrix(val_labels, val_preds, labels=list(range(len(CLASS_LABELS)))))
    print("\nDownstream test = SQA routing_acc + TriviaQA D_control FR (Phase 5).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/domain_classifier_data.json")
    p.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p.add_argument(
        "--output_dir",
        default="/vol/tmp/wagnerql/checkpoints/domain_classifier",
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
