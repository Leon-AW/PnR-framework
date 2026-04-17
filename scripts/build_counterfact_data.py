#!/usr/bin/env python3
"""
Build CounterFact Training & Evaluation Data
=============================================

Converts the original CounterFact dataset (azhx/counterfact, from the ROME paper)
into training and evaluation files for the PnR framework.

Original CounterFact fields per record:
  - requested_rewrite.prompt     : "The mother tongue of {} is"
  - requested_rewrite.subject    : "Danielle Darrieux"
  - requested_rewrite.target_new : {"str": "English", "id": "Q1860"}
  - requested_rewrite.target_true: {"str": "French",  "id": "Q150"}
  - requested_rewrite.relation_id: "P103"
  - paraphrase_prompts           : [2 noisy paraphrases]
  - neighborhood_prompts         : [10 locality probes — same relation, different subject]
  - attribute_prompts            : [10 prompts for subjects with the true attribute]
  - generation_prompts           : [10 alternative phrasings for generation]

Outputs:
  data/counterfact_train.jsonl         — full training data (all records)
  data/counterfact_cluster_{i}.jsonl   — per-cluster training splits (grouped by relation_id)
  data/counterfact_cluster_info.json   — cluster metadata (sizes, relation mapping, centroid)
  data/counterfact_eval.json           — full eval data with neighborhood/generation prompts

Clustering rationale:
  CF records carry a `relation_id` field (P103, P19, ...) identifying the relation
  template (~35 unique values). We cluster by grouping relations: for each
  relation, we compute the mean MiniLM embedding of its prompts, then run
  agglomerative clustering on those per-relation centroids to merge semantically
  similar relations into k buckets. Each CF record inherits the cluster label of
  its relation_id. Final per-cluster centroids (mean MiniLM embedding of all
  member prompts) are computed in the same embedding space as the runtime
  CentroidRouter, so they drop directly into router_state/manifest.json.

  Why not KMeans on raw embeddings? The relation_id field is ground-truth
  supervision we already have — using it gives deterministic, reproducible
  splits where every record in a cluster shares the same relation family.

Usage:
    python scripts/build_counterfact_data.py
    python scripts/build_counterfact_data.py --output_dir data/ --seed 42 --n_clusters 6

Author: Leon Wagner
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

HF_DATASET_ID = "azhx/counterfact"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CounterFact training and eval data from the original ROME dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", default="data",
                        help="Output directory for JSONL/JSON files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling training data")
    parser.add_argument("--n_clusters", type=int, default=6,
                        help="Number of clusters to merge the ~35 relation families into")
    parser.add_argument("--no_cluster", action="store_true",
                        help="Skip clustering step (only build full train + eval files)")
    return parser.parse_args()


def record_to_train(record: dict, idx: int) -> dict:
    """Convert a CounterFact record to training format.

    The adapter learns to complete the relation prompt with target_new (the
    counterfactual answer), overriding the base model's target_true.
    """
    rw = record["requested_rewrite"]
    question = rw["prompt"].format(rw["subject"]).strip()
    answer = rw["target_new"]["str"].strip()

    return {
        "id": str(record["case_id"]),
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "question": question,
        "answer": answer,
        "subject": rw["subject"],
        "relation_id": rw["relation_id"],
    }


def record_to_eval(record: dict) -> dict:
    """Convert a CounterFact record to evaluation format.

    Includes all prompt types for comprehensive evaluation:
      - question (main prompt)      → ESR test
      - generation_prompts          → ESR paraphrase robustness
      - neighborhood_prompts        → locality (should NOT change)
      - target_new / target_true    → answer keys
    """
    rw = record["requested_rewrite"]
    return {
        "case_id": record["case_id"],
        "question": rw["prompt"].format(rw["subject"]).strip(),
        "subject": rw["subject"],
        "relation_id": rw["relation_id"],
        "relation_template": rw["prompt"],
        "target_new": rw["target_new"]["str"].strip(),
        "target_true": rw["target_true"]["str"].strip(),
        "generation_prompts": record.get("generation_prompts", []),
        "neighborhood_prompts": record.get("neighborhood_prompts", []),
        "paraphrase_prompts": record.get("paraphrase_prompts", []),
    }


def group_by_relation(
    train_rows: list[dict], n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Group CF records by relation_id, then merge the ~35 relations into
    n_clusters buckets via agglomerative clustering on per-relation MiniLM
    centroids (cosine metric, average linkage).

    Returns:
        labels: per-row cluster assignment (shape [n_records])
        centers: per-cluster centroid (shape [n_clusters, dim]), L2-normalised,
                 computed as the mean MiniLM embedding of all member prompts
        relation_to_cluster: mapping relation_id → cluster index
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering

    print(f"\n[Clustering] Loading {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    questions = [r["question"] for r in train_rows]
    relations = [r["relation_id"] for r in train_rows]
    print(f"[Clustering] Encoding {len(questions):,} prompts...")
    embeddings = model.encode(
        questions,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    unique_relations = sorted(set(relations))
    print(f"[Clustering] {len(unique_relations)} unique relations → {n_clusters} clusters")

    rel_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, rel in enumerate(relations):
        rel_to_rows[rel].append(i)

    rel_centroids = np.zeros((len(unique_relations), embeddings.shape[1]))
    for j, rel in enumerate(unique_relations):
        idx = rel_to_rows[rel]
        c = embeddings[idx].mean(axis=0)
        rel_centroids[j] = c / (np.linalg.norm(c) + 1e-12)

    print(f"[Clustering] Agglomerative (metric=cosine, linkage=average)...")
    agg = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    rel_labels = agg.fit_predict(rel_centroids)
    relation_to_cluster = {rel: int(lab) for rel, lab in zip(unique_relations, rel_labels)}

    labels = np.array([relation_to_cluster[r] for r in relations])

    centers = np.zeros((n_clusters, embeddings.shape[1]))
    for i in range(n_clusters):
        mask = labels == i
        c = embeddings[mask].mean(axis=0)
        centers[i] = c / (np.linalg.norm(c) + 1e-12)

    sizes = Counter(labels.tolist())
    for i in range(n_clusters):
        n_rels = sum(1 for lab in rel_labels if lab == i)
        print(f"  Cluster {i}: {sizes[i]:,} records | {n_rels} relations")

    return labels, centers, relation_to_cluster


def build_cluster_info(
    train_rows: list[dict],
    labels: np.ndarray,
    n_clusters: int,
    centers: np.ndarray,
    relation_to_cluster: dict[str, int],
) -> dict:
    """Build cluster metadata: sizes, relation_id distribution, representative examples."""
    clusters = defaultdict(list)
    for row, label in zip(train_rows, labels):
        clusters[int(label)].append(row)

    info = {
        "n_clusters": n_clusters,
        "embedding_model": EMBEDDING_MODEL,
        "grouping": "agglomerative_on_relation_centroids",
        "relation_to_cluster": relation_to_cluster,
        "clusters": [],
    }
    for i in range(n_clusters):
        rows = clusters[i]
        rel_counts = Counter(r["relation_id"] for r in rows)
        member_relations = [
            {"relation_id": rid, "count": cnt, "example": next(
                r["question"] for r in rows if r["relation_id"] == rid
            )}
            for rid, cnt in rel_counts.most_common()
        ]
        info["clusters"].append({
            "cluster_id": i,
            "n_records": len(rows),
            "n_relations": len(rel_counts),
            "member_relations": member_relations,
            "centroid": centers[i].tolist(),
        })

    return info


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "counterfact_train.jsonl"
    eval_path = output_dir / "counterfact_eval.json"

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading {HF_DATASET_ID} (original ROME CounterFact)...")
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET_ID, split="train")
    records = list(ds)
    print(f"  Loaded {len(records):,} records (train split)")

    ds_test = load_dataset(HF_DATASET_ID, split="test")
    test_records = list(ds_test)
    print(f"  Loaded {len(test_records):,} records (test split)")

    # ------------------------------------------------------------------
    # Build training JSONL (shuffled train split)
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    train_records = list(records)
    rng.shuffle(train_records)

    train_rows = [record_to_train(rec, i) for i, rec in enumerate(train_records)]

    print(f"\nWriting training data → {train_path}")
    with open(train_path, "w") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {len(train_rows):,} records written")

    # ------------------------------------------------------------------
    # Relation-based clustering (agglomerative on per-relation centroids)
    # ------------------------------------------------------------------
    if not args.no_cluster:
        labels, centers, relation_to_cluster = group_by_relation(
            train_rows, args.n_clusters
        )

        # Per-cluster JSONL files
        clusters: dict[int, list[dict]] = defaultdict(list)
        for row, label in zip(train_rows, labels):
            clusters[int(label)].append(row)

        for i in range(args.n_clusters):
            cluster_path = output_dir / f"counterfact_cluster_{i}.jsonl"
            cluster_rows = clusters[i]
            # Shuffle within cluster
            rng.shuffle(cluster_rows)
            with open(cluster_path, "w") as f:
                for row in cluster_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  Cluster {i}: {len(cluster_rows):,} records → {cluster_path}")

        # Cluster metadata
        cluster_info = build_cluster_info(
            train_rows, labels, args.n_clusters, centers, relation_to_cluster
        )
        cluster_info_path = output_dir / "counterfact_cluster_info.json"
        with open(cluster_info_path, "w") as f:
            json.dump(cluster_info, f, indent=2, ensure_ascii=False)
        print(f"\nCluster info → {cluster_info_path}")

        # Summary
        print(f"\nCluster summary:")
        for c in cluster_info["clusters"]:
            top = c["member_relations"][0]
            print(f"  [{c['cluster_id']}] {c['n_records']:5d} records | "
                  f"{c['n_relations']} relations | "
                  f"top: {top['relation_id']} ({top['count']}) | "
                  f"e.g. \"{top['example'][:60]}\"")

    # ------------------------------------------------------------------
    # Build evaluation JSON (both splits, unshuffled for reproducibility)
    # ------------------------------------------------------------------
    print(f"\nWriting eval data → {eval_path}")
    eval_data = {
        "dataset": HF_DATASET_ID,
        "description": "Original CounterFact (ROME paper) — train + test splits",
        "train": [record_to_eval(r) for r in records],
        "test": [record_to_eval(r) for r in test_records],
        "stats": {
            "n_train": len(records),
            "n_test": len(test_records),
            "n_total": len(records) + len(test_records),
            "neighborhood_prompts_per_record": 10,
            "generation_prompts_per_record": 10,
        },
    }
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"  {len(records) + len(test_records):,} eval records written")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    rel_counts = Counter(r["requested_rewrite"]["relation_id"] for r in records)
    print(f"\nRelation distribution ({len(rel_counts)} unique):")
    for rid, cnt in rel_counts.most_common(10):
        tpl = next(
            r["requested_rewrite"]["prompt"]
            for r in records if r["requested_rewrite"]["relation_id"] == rid
        )
        print(f"  {rid}: {cnt:4d}  \"{tpl}\"")
    print(f"  ... ({len(rel_counts) - 10} more)")

    n_nbr = sum(len(r.get("neighborhood_prompts", [])) for r in records)
    n_gen = sum(len(r.get("generation_prompts", [])) for r in records)
    print(f"\nLocality probes: {n_nbr:,} neighborhood prompts ({n_nbr/len(records):.0f}/record)")
    print(f"Generation probes: {n_gen:,} generation prompts ({n_gen/len(records):.0f}/record)")

    print(f"\nDone.")
    print(f"  Train JSONL:  {train_path}  ({len(train_rows):,} lines)")
    print(f"  Eval JSON:    {eval_path}   ({len(records) + len(test_records):,} records)")
    if not args.no_cluster:
        print(f"  Cluster info: {cluster_info_path}")
        print(f"\nNext: python train/train_counterfact_patch.py --data_path {train_path}")
        print(f"  or:  sbatch slurm/train_cf_clusters.sh  (6 cluster adapters)")


if __name__ == "__main__":
    main()
