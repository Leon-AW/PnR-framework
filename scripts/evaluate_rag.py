#!/usr/bin/env python3
"""
RAG Evaluation Framework
========================

Standalone evaluation script for the Advanced RAG pipeline.
Supports manual annotation, A/B ablation testing, and metric reporting.

Usage:
    python scripts/evaluate_rag.py annotate --output-dir eval_data/ [--with-history]
    python scripts/evaluate_rag.py ablation --output-dir eval_data/ --queries-file queries.txt
    python scripts/evaluate_rag.py report --output-dir eval_data/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.query_pipeline import QueryPipeline
from src.inference.rag_config import RAGServerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

ANNOTATIONS_FILE = "annotations.json"


def _load_annotations(output_dir: Path) -> dict:
    """Load existing annotations or create empty structure."""
    path = output_dir / ANNOTATIONS_FILE
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    return {"version": 1, "annotations": []}


def _save_annotations(output_dir: Path, data: dict) -> None:
    """Save annotations to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ANNOTATIONS_FILE
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _next_query_id(annotations: list[dict]) -> str:
    """Generate the next query ID."""
    if not annotations:
        return "q001"
    max_id = max(int(a["query_id"][1:]) for a in annotations)
    return f"q{max_id + 1:03d}"


# =============================================================================
# Metrics
# =============================================================================

def compute_mrr(annotations: list[dict]) -> float:
    """Mean Reciprocal Rank — average of 1/rank of first relevant result."""
    rrs = []
    for ann in annotations:
        results = ann.get("results", [])
        for i, r in enumerate(results):
            if r.get("relevant") is True:
                rrs.append(1.0 / (i + 1))
                break
        else:
            rrs.append(0.0)
    return mean(rrs) if rrs else 0.0


def compute_precision_at_k(annotations: list[dict], k: int = 5) -> float:
    """Precision@K — fraction of relevant results in top-K."""
    precisions = []
    for ann in annotations:
        results = ann.get("results", [])[:k]
        if not results:
            continue
        relevant = sum(1 for r in results if r.get("relevant") is True)
        precisions.append(relevant / len(results))
    return mean(precisions) if precisions else 0.0


def compute_recall_at_k(annotations: list[dict], k: int = 5) -> float:
    """Recall@K — fraction of total relevant docs found in top-K."""
    recalls = []
    for ann in annotations:
        results = ann.get("results", [])
        total_relevant = sum(1 for r in results if r.get("relevant") is True)
        if total_relevant == 0:
            continue
        top_k_relevant = sum(1 for r in results[:k] if r.get("relevant") is True)
        recalls.append(top_k_relevant / total_relevant)
    return mean(recalls) if recalls else 0.0


def _print_metrics(annotations: list[dict]) -> None:
    """Print computed metrics."""
    # Filter to only fully annotated queries (no skipped results)
    annotated = [
        a for a in annotations
        if all(r.get("relevant") is not None for r in a.get("results", []))
    ]

    if not annotated:
        print("\nNo fully annotated queries yet.")
        return

    print(f"\n{'='*50}")
    print(f"Metrics ({len(annotated)} annotated queries)")
    print(f"{'='*50}")
    print(f"  MRR:          {compute_mrr(annotated):.4f}")
    print(f"  Precision@3:  {compute_precision_at_k(annotated, 3):.4f}")
    print(f"  Precision@5:  {compute_precision_at_k(annotated, 5):.4f}")
    print(f"  Recall@3:     {compute_recall_at_k(annotated, 3):.4f}")
    print(f"  Recall@5:     {compute_recall_at_k(annotated, 5):.4f}")
    print(f"{'='*50}")


# =============================================================================
# Annotate Mode
# =============================================================================

def cmd_annotate(args: argparse.Namespace) -> None:
    """Interactive annotation of RAG retrieval results."""
    output_dir = Path(args.output_dir)
    data = _load_annotations(output_dir)
    annotations = data["annotations"]

    print(f"Loaded {len(annotations)} existing annotations from {output_dir}")

    # Initialize pipeline
    config = RAGServerConfig.from_env()
    pipeline = QueryPipeline(config)
    print("Loading pipeline (embedding model, reranker, indices)...")
    pipeline.load()
    print("Pipeline ready.\n")

    history: list[dict] = []

    print("Enter a query (or 'quit' to stop):")

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        # Run pipeline
        result = pipeline.run(query, history if args.with_history else [])
        analysis = result.query_analysis

        print(f"\n  Data source: {analysis.data_source}")
        print(f"  Language: {analysis.language}")
        print(f"  Reformulated: {analysis.reformulated_query}")
        print(f"  Citations: {len(result.citations)}")

        if not result.citations:
            print("  No results retrieved (intent={analysis.intent}).")
            if args.with_history:
                history.append({"role": "user", "content": query})
            continue

        # Show results and collect annotations
        annotation_results = []
        for c in result.citations:
            print(f"\n  [{c.index}] Score: {c.score:.4f}")
            print(f"      Source: {c.source_file}")
            print(f"      Section: {c.section}")
            print(f"      Preview: {c.content_preview[:120]}...")
            print(f"      [r]elevant / [i]rrelevant / [s]kip ?", end=" ")

            choice = input().strip().lower()
            if choice.startswith("r"):
                relevant = True
            elif choice.startswith("i"):
                relevant = False
            else:
                relevant = None  # skipped

            annotation_results.append({
                "chunk_id": c.chunk_id,
                "source_file": c.source_file,
                "section": c.section,
                "score": c.score,
                "content_preview": c.content_preview,
                "relevant": relevant,
            })

        # Save annotation
        query_id = _next_query_id(annotations)
        annotation = {
            "query_id": query_id,
            "query": query,
            "data_source": analysis.data_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_snapshot": {
                "enable_hyde": config.enable_hyde,
                "enable_reranking": config.enable_reranking,
                "enable_llm_anaphora": config.enable_llm_anaphora,
            },
            "results": annotation_results,
        }
        annotations.append(annotation)
        _save_annotations(output_dir, data)
        print(f"\n  Saved annotation {query_id}")

        # Show running metrics
        _print_metrics(annotations)

        # Update history for with-history mode
        if args.with_history:
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "(annotated)"})

    print(f"\nDone. {len(annotations)} total annotations in {output_dir / ANNOTATIONS_FILE}")


# =============================================================================
# Ablation Mode
# =============================================================================

def cmd_ablation(args: argparse.Namespace) -> None:
    """A/B comparison across config variants."""
    output_dir = Path(args.output_dir)

    # Collect queries
    queries: list[str] = []
    if args.queries_file:
        qf = Path(args.queries_file)
        if not qf.exists():
            print(f"Error: queries file not found: {qf}")
            sys.exit(1)
        queries = [line.strip() for line in qf.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        # Use queries from existing annotations
        data = _load_annotations(output_dir)
        queries = [a["query"] for a in data.get("annotations", [])]

    if not queries:
        print("Error: no queries provided. Use --queries-file or annotate first.")
        sys.exit(1)

    print(f"Running ablation with {len(queries)} queries...")

    base_config = RAGServerConfig.from_env()

    # Define variants
    variants = {
        "baseline": base_config,
        "no_hyde": replace(base_config, enable_hyde=False),
        "no_reranking": replace(base_config, enable_reranking=False),
        "no_hyde_no_reranking": replace(base_config, enable_hyde=False, enable_reranking=False),
        "no_llm_anaphora": replace(base_config, enable_llm_anaphora=False),
    }

    all_results: dict[str, list[dict]] = {name: [] for name in variants}

    for variant_name, variant_config in variants.items():
        print(f"\n--- Variant: {variant_name} ---")
        pipeline = QueryPipeline(variant_config)
        pipeline.load()

        for i, query in enumerate(queries):
            start = time.time()
            result = pipeline.run(query, [])
            elapsed = time.time() - start

            chunk_ids = [c.chunk_id for c in result.citations]
            scores = [c.score for c in result.citations]

            all_results[variant_name].append({
                "query": query,
                "chunk_ids": chunk_ids,
                "scores": scores,
                "latency_s": elapsed,
                "candidates_before_rerank": result.metadata.get("candidates_before_rerank", 0),
                "results_after_rerank": result.metadata.get("results_after_rerank", 0),
            })

            print(f"  [{i+1}/{len(queries)}] {query[:60]}... -> {len(chunk_ids)} results, {elapsed:.2f}s")

    # Compute comparison metrics
    print(f"\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")

    # Score statistics per variant
    header = f"{'Variant':<25} {'#Res':>5} {'Mean':>7} {'Med':>7} {'Min':>7} {'Max':>7} {'Lat(s)':>7} {'Lat95':>7}"
    print(header)
    print("-" * len(header))

    for name, results in all_results.items():
        all_scores = [s for r in results for s in r["scores"]]
        latencies = [r["latency_s"] for r in results]
        n_results = mean(len(r["chunk_ids"]) for r in results) if results else 0

        if all_scores:
            row = (
                f"{name:<25} "
                f"{n_results:>5.1f} "
                f"{mean(all_scores):>7.4f} "
                f"{median(all_scores):>7.4f} "
                f"{min(all_scores):>7.4f} "
                f"{max(all_scores):>7.4f} "
                f"{mean(latencies):>7.2f} "
                f"{sorted(latencies)[int(len(latencies)*0.95)]:>7.2f}"
            )
        else:
            row = f"{name:<25} {'N/A':>5} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7} {mean(latencies):>7.2f} {'N/A':>7}"
        print(row)

    # Jaccard overlap between variants
    print(f"\nJaccard Overlap (chunk ID sets):")
    variant_names = list(all_results.keys())
    for i, name_a in enumerate(variant_names):
        for name_b in variant_names[i+1:]:
            overlaps = []
            for ra, rb in zip(all_results[name_a], all_results[name_b]):
                set_a = set(ra["chunk_ids"])
                set_b = set(rb["chunk_ids"])
                if set_a or set_b:
                    overlaps.append(len(set_a & set_b) / len(set_a | set_b))
            if overlaps:
                print(f"  {name_a} vs {name_b}: {mean(overlaps):.4f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"ablation_{timestamp}.json"
    out_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}")


# =============================================================================
# Report Mode
# =============================================================================

def cmd_report(args: argparse.Namespace) -> None:
    """Print metrics from saved annotations."""
    output_dir = Path(args.output_dir)
    data = _load_annotations(output_dir)
    annotations = data.get("annotations", [])

    if not annotations:
        print(f"No annotations found in {output_dir / ANNOTATIONS_FILE}")
        sys.exit(1)

    print(f"Loaded {len(annotations)} annotations from {output_dir / ANNOTATIONS_FILE}")

    # Per-data-source breakdown
    by_source: dict[str, list[dict]] = {}
    for ann in annotations:
        src = ann.get("data_source", "unknown")
        by_source.setdefault(src, []).append(ann)

    for source, anns in by_source.items():
        print(f"\n--- Data Source: {source} ({len(anns)} queries) ---")
        _print_metrics(anns)

    # Overall
    print("\n--- Overall ---")
    _print_metrics(annotations)

    # Config snapshot summary
    configs_seen = set()
    for ann in annotations:
        snap = ann.get("config_snapshot", {})
        configs_seen.add(json.dumps(snap, sort_keys=True))

    if len(configs_seen) > 1:
        print(f"\nNote: Annotations were collected with {len(configs_seen)} different config snapshots.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # annotate
    p_ann = subparsers.add_parser("annotate", help="Interactive annotation")
    p_ann.add_argument("--output-dir", required=True, help="Output directory for annotations")
    p_ann.add_argument("--with-history", action="store_true", help="Maintain conversation history between queries")

    # ablation
    p_abl = subparsers.add_parser("ablation", help="A/B ablation comparison")
    p_abl.add_argument("--output-dir", required=True, help="Output directory for results")
    p_abl.add_argument("--queries-file", help="File with one query per line")

    # report
    p_rep = subparsers.add_parser("report", help="Print metrics from annotations")
    p_rep.add_argument("--output-dir", required=True, help="Directory containing annotations.json")

    args = parser.parse_args()

    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "ablation":
        cmd_ablation(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
