#!/usr/bin/env python3
"""
MORPHEUS Continual Learning Evaluation (Level 4a)
===================================================

Evaluates MORPHEUS's core claim: learning continuously across sequential
domains without catastrophic forgetting. This goes beyond static inference
evaluation (Level 3) by simulating a realistic continual learning scenario.

Protocol:
  1. Start with a base-domain trained system
  2. Sequentially introduce new domains (one at a time)
  3. After each domain introduction:
     - Evaluate on the NEW domain  (measures learning)
     - Evaluate on ALL PREVIOUS domains (measures forgetting)
     - Log expert lifecycle events (spawns, merges, prunes)
  4. Produce a forgetting curve and comparison table

The script orchestrates the train->eval loop without requiring a GPU LLM
for the evaluation pass — it uses the existing EvalRunner for QA metrics
or can run in lightweight "routing-only" mode that only measures routing
accuracy and expert lifecycle dynamics.

Usage:
    # Full evaluation with LLM inference
    python eval_morpheus_continual.py \\
        --domains base temporal geo_india geo_france \\
        --n_samples 50 \\
        --output_dir eval_results/morpheus_continual

    # Lightweight routing-only evaluation (no LLM needed)
    python eval_morpheus_continual.py \\
        --domains base temporal geo_india geo_france \\
        --routing_only \\
        --output_dir eval_results/morpheus_routing

Comparison baselines:
    The script can also run the same sequential protocol using:
    --baseline monolithic   (retrain a single LoRA each time)
    --baseline pnr          (PnR centroid router)
    --baseline xlora        (X-LoRA soft gating)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging import setup_logger, configure_framework_logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DomainEvalResult:
    """Evaluation result for one domain after one training step."""
    domain: str
    step: int
    n_samples: int
    exact_match: float
    f1: float
    routing_accuracy: float | None
    latency_ms: float | None = None


@dataclass
class ExpertLifecycleEvent:
    """A logged expert lifecycle event."""
    step: int
    event_type: str
    expert_id: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ContinualEvalReport:
    """Full report from a continual learning evaluation run."""
    architecture: str
    domains_sequence: list[str]
    n_samples_per_domain: int

    # Per-step, per-domain results
    domain_results: list[DomainEvalResult] = field(default_factory=list)

    # Expert lifecycle log
    lifecycle_events: list[ExpertLifecycleEvent] = field(default_factory=list)

    # Summary metrics
    final_forgetting_rates: dict[str, float] = field(default_factory=dict)
    forward_transfer_scores: dict[str, float] = field(default_factory=dict)
    peak_expert_count: int = 0
    total_training_time_s: float = 0.0

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "domains_sequence": self.domains_sequence,
            "n_samples_per_domain": self.n_samples_per_domain,
            "domain_results": [asdict(r) for r in self.domain_results],
            "lifecycle_events": [asdict(e) for e in self.lifecycle_events],
            "final_forgetting_rates": self.final_forgetting_rates,
            "forward_transfer_scores": self.forward_transfer_scores,
            "peak_expert_count": self.peak_expert_count,
            "total_training_time_s": self.total_training_time_s,
            "timestamp": self.timestamp,
        }

    def compute_forgetting_curve(self) -> dict[str, list[dict[str, float]]]:
        """Compute per-domain accuracy over time (the forgetting curve).

        Returns:
            domain -> [{step, accuracy}, ...] for plotting.
        """
        curves: dict[str, list[dict[str, float]]] = {}
        for result in self.domain_results:
            if result.domain not in curves:
                curves[result.domain] = []
            curves[result.domain].append({
                "step": result.step,
                "exact_match": result.exact_match,
                "f1": result.f1,
            })
        return curves

    def compute_final_forgetting(self) -> dict[str, float]:
        """Compute forgetting = peak_accuracy - final_accuracy for each domain.

        Positive values = forgetting occurred.
        Negative values = backward transfer improved the domain.
        """
        curves = self.compute_forgetting_curve()
        forgetting = {}
        for domain, points in curves.items():
            if len(points) < 2:
                continue
            ems = [p["exact_match"] for p in points]
            peak = max(ems)
            final = ems[-1]
            forgetting[domain] = peak - final
        return forgetting


# =============================================================================
# Routing-Only Evaluation (no LLM needed)
# =============================================================================

def evaluate_routing_only(
    router,
    eval_samples: list[dict[str, Any]],
) -> dict[str, float]:
    """Evaluate routing accuracy without LLM inference.

    Tests whether the router selects the correct expert for each query.
    Much faster than full inference evaluation.

    Args:
        router: PrototypeRouter or CentroidRouter instance.
        eval_samples: List of {question, expected_adapter} dicts.

    Returns:
        Dict with routing_accuracy and mean_confidence.
    """
    correct = 0
    confidences = []

    for sample in eval_samples:
        question = sample["question"]
        expected = sample.get("expected_adapter")
        if expected is None:
            continue

        try:
            result = router.route(question)
            if result.winner_adapter == expected:
                correct += 1
            if result.winner_similarity is not None:
                confidences.append(result.winner_similarity)
        except Exception:
            pass

    n = len([s for s in eval_samples if s.get("expected_adapter")])
    return {
        "routing_accuracy": correct / n if n > 0 else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "n_evaluated": n,
    }


# =============================================================================
# Sequential Domain Protocol
# =============================================================================

def run_sequential_protocol(
    domains: list[str],
    n_samples: int,
    architecture: str,
    routing_only: bool = False,
    output_dir: str = "eval_results/morpheus_continual",
    checkpoints_dir: str = "checkpoints",
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    **kwargs,
) -> ContinualEvalReport:
    """Run the sequential domain learning protocol.

    For each domain in sequence:
    1. "Learn" the domain (register expert in routing)
    2. Evaluate on all domains seen so far
    3. Log expert lifecycle events

    Args:
        domains: Ordered list of domain names (e.g., ["base", "temporal", "geo_india"]).
        n_samples: Samples per domain per evaluation step.
        architecture: "morpheus", "pnr", "monolithic", or "xlora".
        routing_only: If True, only evaluate routing accuracy (no LLM).
        output_dir: Where to save results.
        checkpoints_dir: Path to adapter checkpoints.
        model_id: Base model identifier.

    Returns:
        ContinualEvalReport with forgetting curves and lifecycle events.
    """
    report = ContinualEvalReport(
        architecture=architecture,
        domains_sequence=domains,
        n_samples_per_domain=n_samples,
    )

    t_start = time.time()

    if architecture == "morpheus":
        report = _run_morpheus_protocol(
            report, domains, n_samples, routing_only, checkpoints_dir, model_id, **kwargs,
        )
    elif architecture == "pnr":
        report = _run_pnr_protocol(
            report, domains, n_samples, routing_only, checkpoints_dir, model_id, **kwargs,
        )
    else:
        logger.warning(f"Architecture '{architecture}' not yet implemented for continual eval")

    report.total_training_time_s = time.time() - t_start
    report.final_forgetting_rates = report.compute_final_forgetting()

    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f"continual_eval_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    logger.info(f"Report saved to {report_file}")

    return report


def _run_morpheus_protocol(
    report: ContinualEvalReport,
    domains: list[str],
    n_samples: int,
    routing_only: bool,
    checkpoints_dir: str,
    model_id: str,
    **kwargs,
) -> ContinualEvalReport:
    """Run the sequential protocol with MORPHEUS architecture."""
    from src.morpheus.config import MorpheusConfig, PrototypeRouterConfig, ExpertBankConfig
    from src.morpheus.router import PrototypeRouter
    from src.morpheus.expert_bank import ExpertBank
    from src.morpheus.fast_buffer import FastBuffer
    from src.morpheus.meta_controller import MetaController, SystemState

    embed_fn = _build_embedding_fn(kwargs.get("embedding_model"))

    router_config = PrototypeRouterConfig(
        projection_dim=256,
        similarity_threshold=0.4,
        hierarchical_routing=False,
    )
    router = PrototypeRouter(
        config=router_config,
        embedding_fn=embed_fn,
        embedding_dim=768,
    )
    bank = ExpertBank(ExpertBankConfig(
        shadow_period_steps=1,
        checkpoint_dir=checkpoints_dir,
    ))
    buf = FastBuffer()
    mc = MetaController()

    domains_seen: list[str] = []
    max_experts = 0

    for step, domain in enumerate(domains):
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP {step}: Introducing domain '{domain}'")
        logger.info(f"{'='*60}")

        # 1. "Learn" the domain: register expert adapter
        adapter_dir = Path(checkpoints_dir)
        adapter_id = _domain_to_adapter_id(domain)
        adapter_path = adapter_dir / adapter_id

        if adapter_path.exists():
            centroid = _compute_adapter_centroid(adapter_path, embed_fn)

            bank.spawn_expert(adapter_id, domain=domain)
            bank.record_training_step(adapter_id, loss=0.1, is_shadow=True)
            bank.promote_to_active(adapter_id)

            router.register_adapter(
                adapter_id=adapter_id,
                path=str(adapter_path),
                timestamp=float(step),
                centroid=centroid,
            )

            report.lifecycle_events.append(ExpertLifecycleEvent(
                step=step, event_type="spawn_and_activate",
                expert_id=adapter_id,
                details={"domain": domain},
            ))
            logger.info(f"  Registered expert: {adapter_id}")
        else:
            logger.warning(f"  No adapter found at {adapter_path}, skipping registration")

        domains_seen.append(domain)
        max_experts = max(max_experts, bank.num_experts)

        # 2. Evaluate on ALL domains seen so far
        for eval_domain in domains_seen:
            eval_samples = _load_eval_samples(eval_domain, n_samples)
            if not eval_samples:
                logger.warning(f"  No eval samples for domain={eval_domain}")
                continue

            if routing_only:
                metrics = evaluate_routing_only(router, eval_samples)
                report.domain_results.append(DomainEvalResult(
                    domain=eval_domain,
                    step=step,
                    n_samples=metrics["n_evaluated"],
                    exact_match=0.0,
                    f1=0.0,
                    routing_accuracy=metrics["routing_accuracy"],
                ))
            else:
                metrics = _run_full_eval(eval_domain, n_samples, "morpheus", checkpoints_dir, model_id)
                report.domain_results.append(DomainEvalResult(
                    domain=eval_domain,
                    step=step,
                    n_samples=n_samples,
                    exact_match=metrics.get("exact_match", 0.0),
                    f1=metrics.get("f1", 0.0),
                    routing_accuracy=metrics.get("routing_accuracy"),
                    latency_ms=metrics.get("avg_latency_ms"),
                ))

            logger.info(
                f"  Eval [{eval_domain}]: "
                f"RA={report.domain_results[-1].routing_accuracy or 'N/A'}"
            )

        # 3. Meta-controller observation
        state = SystemState(
            buffer_fill_level=buf.fill_level,
            num_active_experts=len(bank.active_experts),
            capacity_utilization=bank.num_experts / 64.0,
        )
        mc.observe(state)

    report.peak_expert_count = max_experts
    return report


def _run_pnr_protocol(
    report: ContinualEvalReport,
    domains: list[str],
    n_samples: int,
    routing_only: bool,
    checkpoints_dir: str,
    model_id: str,
    **kwargs,
) -> ContinualEvalReport:
    """Run the sequential protocol with PnR centroid router."""
    embed_fn = _build_embedding_fn(kwargs.get("embedding_model"))

    from src.routing import CentroidRouter

    router = CentroidRouter(
        embedding_model_path=kwargs.get("embedding_model"),
        similarity_threshold=0.5,
        use_gpu=False,
    )

    checkpoints_path = Path(checkpoints_dir)
    if checkpoints_path.exists():
        router.register_from_checkpoints(str(checkpoints_path))

    domains_seen: list[str] = []
    for step, domain in enumerate(domains):
        logger.info(f"\nSTEP {step}: Domain '{domain}'")
        domains_seen.append(domain)

        for eval_domain in domains_seen:
            eval_samples = _load_eval_samples(eval_domain, n_samples)
            if not eval_samples:
                continue

            if routing_only:
                metrics = evaluate_routing_only(router, eval_samples)
                report.domain_results.append(DomainEvalResult(
                    domain=eval_domain, step=step,
                    n_samples=metrics["n_evaluated"],
                    exact_match=0.0, f1=0.0,
                    routing_accuracy=metrics["routing_accuracy"],
                ))

    return report


# =============================================================================
# Helpers
# =============================================================================

def _build_embedding_fn(embedding_model_path: str | None = None):
    """Build an embedding function for routing evaluation."""
    try:
        from sentence_transformers import SentenceTransformer
        model_path = embedding_model_path or "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_path)

        def embed(text: str) -> np.ndarray:
            return model.encode(text, normalize_embeddings=True).astype(np.float32)

        return embed
    except ImportError:
        logger.warning("sentence-transformers not available, using random embeddings")
        rng = np.random.RandomState(42)
        cache = {}

        def embed(text: str) -> np.ndarray:
            if text not in cache:
                v = rng.randn(768).astype(np.float32)
                cache[text] = v / np.linalg.norm(v)
            return cache[text]

        return embed


def _domain_to_adapter_id(domain: str) -> str:
    """Map domain name to expected adapter directory name."""
    mapping = {
        "base": "base_v1",
        "temporal": "patch_temp_2019_plus",
    }
    if domain in mapping:
        return mapping[domain]
    if domain.startswith("geo_"):
        country = domain[4:]
        return f"patch_geo_{country}"
    return domain


def _compute_adapter_centroid(
    adapter_path: Path,
    embed_fn,
) -> np.ndarray:
    """Compute a centroid for an adapter from its training data metadata."""
    meta_path = adapter_path / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        texts = meta.get("sample_questions", [])[:50]
        if texts:
            embeddings = np.vstack([embed_fn(t) for t in texts])
            centroid = embeddings.mean(axis=0)
            return centroid / (np.linalg.norm(centroid) + 1e-9)

    return embed_fn(adapter_path.name)


def _load_eval_samples(domain: str, n_samples: int) -> list[dict[str, Any]]:
    """Load evaluation samples for a domain."""
    try:
        from src.eval.dataset import build_situated_qa_dataset, _infer_expected_adapter
        samples = build_situated_qa_dataset(split=domain, n_samples=n_samples)
        return [
            {
                "question": s.question,
                "gold_answers": s.gold_answers,
                "expected_adapter": s.expected_adapter,
            }
            for s in samples
        ]
    except Exception as e:
        logger.warning(f"Could not load samples for domain={domain}: {e}")
        return []


def _run_full_eval(
    domain: str,
    n_samples: int,
    architecture: str,
    checkpoints_dir: str,
    model_id: str,
) -> dict[str, float]:
    """Run full LLM inference evaluation for one domain."""
    from src.eval.runner import EvalConfig, EvalRunner

    config = EvalConfig(
        model_id=model_id,
        checkpoints_dir=checkpoints_dir,
        eval_sets=[domain],
        n_samples=n_samples,
        morpheus=(architecture == "morpheus"),
    )
    runner = EvalRunner(config)
    report = runner.run()
    summary = report.get("summary", {})
    return {
        "exact_match": summary.get("exact_match_overall", 0.0),
        "f1": summary.get("f1_overall", 0.0),
        "routing_accuracy": summary.get("routing_accuracy"),
        "avg_latency_ms": summary.get("efficiency", {}).get("avg_latency_ms"),
    }


# =============================================================================
# Report Printing
# =============================================================================

def print_report(report: ContinualEvalReport) -> None:
    """Print a human-readable summary of the continual eval results."""
    logger.info("\n" + "=" * 70)
    logger.info(f"CONTINUAL LEARNING EVALUATION — {report.architecture.upper()}")
    logger.info("=" * 70)
    logger.info(f"Domain sequence: {' -> '.join(report.domains_sequence)}")
    logger.info(f"Samples per domain: {report.n_samples_per_domain}")
    logger.info(f"Peak expert count: {report.peak_expert_count}")
    logger.info(f"Total time: {report.total_training_time_s:.1f}s")

    # Forgetting curve summary
    logger.info("\n--- Forgetting Rates (peak - final accuracy) ---")
    for domain, rate in sorted(report.final_forgetting_rates.items()):
        indicator = "OK" if rate <= 0.05 else "WARN" if rate <= 0.15 else "ALERT"
        logger.info(f"  {domain}: {rate:+.4f}  [{indicator}]")

    # Domain accuracy at each step
    curves = report.compute_forgetting_curve()
    logger.info("\n--- Accuracy Over Time ---")
    for domain, points in sorted(curves.items()):
        scores = [f"step{p['step']}={p['exact_match']:.2f}" for p in points]
        logger.info(f"  {domain}: {', '.join(scores)}")

    # Lifecycle events
    if report.lifecycle_events:
        logger.info(f"\n--- Expert Lifecycle ({len(report.lifecycle_events)} events) ---")
        for evt in report.lifecycle_events[:20]:
            logger.info(f"  [{evt.step}] {evt.event_type}: {evt.expert_id}")

    logger.info("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MORPHEUS Continual Learning Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["base", "temporal"],
        help="Ordered list of domains to introduce sequentially",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Evaluation samples per domain per step",
    )
    parser.add_argument(
        "--architecture",
        choices=["morpheus", "pnr", "monolithic", "xlora"],
        default="morpheus",
        help="Architecture to evaluate",
    )
    parser.add_argument(
        "--routing_only",
        action="store_true",
        help="Only evaluate routing accuracy (no LLM inference needed)",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Path to adapter checkpoints",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model identifier",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Embedding model for routing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/morpheus_continual",
        help="Output directory for results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_framework_logging(level=args.log_level)
    logger_ = setup_logger("eval_morpheus_continual", level=args.log_level)

    logger_.info("=" * 70)
    logger_.info("MORPHEUS CONTINUAL LEARNING EVALUATION")
    logger_.info("=" * 70)
    logger_.info(f"  Architecture: {args.architecture}")
    logger_.info(f"  Domains: {args.domains}")
    logger_.info(f"  Routing only: {args.routing_only}")

    report = run_sequential_protocol(
        domains=args.domains,
        n_samples=args.n_samples,
        architecture=args.architecture,
        routing_only=args.routing_only,
        output_dir=args.output_dir,
        checkpoints_dir=args.checkpoints_dir,
        model_id=args.model_id,
        embedding_model=args.embedding_model,
    )

    print_report(report)


if __name__ == "__main__":
    main()
