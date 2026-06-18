#!/usr/bin/env python3
"""Open-stream routing stress test — does the Stage-1 gate hold outside the closed world?

The thesis evaluates routing only on the four classes the Stage-1 domain
classifier was trained on ``{cf, sqa, qm, ood_trivia}``. This script feeds the
held-out 5-domain test set (``data/openstream_heldout.json``, built by
``scripts/build_openstream_testset.py``) through the **unchanged** production
routing pipeline and measures:

Phase A — routing leak (no LLM; embedding + classifier only)
    For every held-out query, log the Stage-1 prediction and the end-to-end
    routing decision. Three nested leak definitions:
      * stage1_argmax_leak    — classifier argmax != ood_trivia
      * stage1_confident_leak — argmax != ood_trivia AND top_prob >= conf_thr
      * routing_leak (HEADLINE)— router.route() returns a winner adapter
        (None ⇒ frozen base ⇒ FR-by-construction holds). This is the only path
        that can actually change an answer, so it is the architecturally
        meaningful number.
    Reported overall, overall-excluding-nq_open (the trivia-adjacent probe), and
    per domain; routing leaks broken down by which adapter family they hit.

Phase B — conditional damage (LLM generation; leaked subset only)
    For each end-to-end leak, generate the routed answer and the frozen-base
    answer for the *same* query (``skip_routing=True`` ⇒ identical prompt path,
    no adapter). A leak whose routed answer equals the frozen answer is benign;
    a changed answer is the failure mode that matters. Reports
    answer_changed_rate. No leaks ⇒ Phase B is a no-op (the strongest result).

Routing is built with the exact production configuration used by the D_eval
sweeps (``slurm/eval_qm_deval.sh``): router_state + domain_classifier,
similarity_threshold=0.45, domain_confidence_threshold=0.7,
domain_fallback_threshold=0.30. Greedy decoding, batch_size=1 — comparable to
all other D_eval runs.

Author: Leon Wagner
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Production routing configuration (matches slurm/eval_qm_deval.sh PnR-routing args).
PROD = {
    "router_state_path": "checkpoints/router_state",
    "domain_classifier_path": "checkpoints/domain_classifier",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_threshold": 0.45,
    "domain_confidence_threshold": 0.7,
    "domain_fallback_threshold": 0.30,
}

OOD_CLASS = "ood_trivia"
TRIVIA_ADJACENT_DOMAIN = "nq_open"

# Adapter-name -> family. Mirrors CentroidRouter._allowed_adapters_for_domain.
_CF_PREFIX = "patch_cf_relfam_"
_GEO_PREFIX = "patch_geo_"
_SQA_BASE = {"base_v1", "patch_temp_2019_plus"}
_QM_ADAPTERS = {"base_qm", "patch_qm_current"}


def adapter_family(name: str | None) -> str | None:
    if name is None:
        return None
    if name.startswith(_CF_PREFIX):
        return "cf"
    if name.startswith(_GEO_PREFIX) or name in _SQA_BASE:
        return "sqa"
    if name in _QM_ADAPTERS:
        return "qm"
    return "other"


def _normalizer():
    """Prefer the project's normalize_answer; fall back to a minimal one."""
    try:
        from src.eval.metrics import normalize_answer
        return normalize_answer
    except Exception:
        import re
        import string

        def _norm(s: str) -> str:
            s = (s or "").lower()
            s = "".join(ch for ch in s if ch not in string.punctuation)
            s = re.sub(r"\b(a|an|the)\b", " ", s)
            return " ".join(s.split())

        return _norm


# ---------------------------------------------------------------------------
# Router / pipeline construction
# ---------------------------------------------------------------------------

def _attach_classifier(router) -> None:
    from src.routing.domain_classifier import DomainClassifier

    clf = DomainClassifier.load(PROD["domain_classifier_path"], device="auto")
    router._domain_classifier = clf
    router._domain_confidence_threshold = PROD["domain_confidence_threshold"]
    router._domain_fallback_threshold = PROD["domain_fallback_threshold"]


def build_router_only(use_gpu: bool):
    """Lightweight router (no LLM) for Phase A."""
    from src.routing import CentroidRouter

    router = CentroidRouter.load(
        path=PROD["router_state_path"],
        embedding_model_path=PROD["embedding_model"],
        similarity_threshold=PROD["similarity_threshold"],
        use_gpu=use_gpu,
    )
    _attach_classifier(router)
    return router


def build_full_pipeline(use_gpu: bool, max_new_tokens: int):
    """Production PnR pipeline (router + LLM) via the existing eval runner.

    Reusing EvalRunner._build_pipeline guarantees the routing, prompt format,
    and generation config are byte-identical to the D_eval sweeps (honors the
    feedback_dcontrol_format lesson).
    """
    from src.eval.runner import EvalConfig, EvalRunner

    cfg = EvalConfig(
        router_state_path=PROD["router_state_path"],
        embedding_model=PROD["embedding_model"],
        similarity_threshold=PROD["similarity_threshold"],
        domain_classifier_path=PROD["domain_classifier_path"],
        domain_confidence_threshold=PROD["domain_confidence_threshold"],
        domain_fallback_threshold=PROD["domain_fallback_threshold"],
        quantization="int4",
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_gpu=use_gpu,
    )
    runner = EvalRunner(cfg)
    return runner._build_pipeline()


# ---------------------------------------------------------------------------
# Wiring sanity — confirm the gate is actually attached before trusting numbers
# ---------------------------------------------------------------------------

def _first_question(path: str, extract) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix == ".jsonl":
            with p.open() as f:
                for line in f:
                    q = extract(json.loads(line))
                    if q:
                        return q
        else:
            data = json.load(p.open())
            recs = data.get("records", data) if isinstance(data, dict) else data
            for r in recs:
                q = extract(r)
                if q:
                    return q
    except Exception:
        return None
    return None


def wiring_sanity(router) -> None:
    print("\n=== Wiring sanity (expected Stage-1 class → routing) ===", flush=True)
    probes: list[tuple[str, str, str]] = []
    cf = _first_question("data/counterfact_train.jsonl", lambda r: (r.get("question") or "").strip())
    if cf:
        probes.append(("cf", cf, "cf family or masked"))
    qm = _first_question("data/qm_train.jsonl",
                         lambda r: ((r.get("messages") or [{}])[0].get("content") or "").strip())
    if qm:
        probes.append(("qm", qm, "qm family"))
    trivia = _first_question("data/triviaqa_dcontrol.json",
                            lambda r: (r.get("question") or r.get("text") or "").strip())
    if trivia:
        probes.append(("ood_trivia", trivia, "winner=None (frozen base)"))

    for expected, q, note in probes:
        top_class, top_prob, _ = router._classify_domain(q)
        winner = router.route(q).winner_adapter
        ok = "✓" if top_class == expected else "✗"
        print(f"  [{ok}] expect={expected:11} got={top_class:11} p={top_prob:.3f} "
              f"winner={winner} ({note})\n        q={q[:70]!r}", flush=True)


# ---------------------------------------------------------------------------
# Phase A — routing leak
# ---------------------------------------------------------------------------

def run_phase_a(router, records: list[dict]) -> list[dict]:
    print(f"\n=== Phase A — routing {len(records)} held-out queries ===", flush=True)
    rows: list[dict] = []
    for i, r in enumerate(records):
        q = r["text"]
        top_class, top_prob, probs = router._classify_domain(q)
        result = router.route(q)
        winner = result.winner_adapter
        rows.append({
            "id": r["id"],
            "domain": r["domain"],
            "source": r["source"],
            "text": q,
            "top_class": top_class,
            "top_prob": float(top_prob),
            "probs": {k: float(v) for k, v in probs.items()},
            "winner_adapter": winner,
            "winner_family": adapter_family(winner),
            "winner_similarity": getattr(result, "winner_similarity", None),
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(records)} routed", flush=True)
    return rows


def _aggregate(rows: list[dict], conf_thr: float) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    argmax_leak = sum(1 for r in rows if r["top_class"] != OOD_CLASS)
    conf_leak = sum(1 for r in rows
                    if r["top_class"] != OOD_CLASS and r["top_prob"] >= conf_thr)
    routing_leak = [r for r in rows if r["winner_adapter"] is not None]
    fam_counts: dict[str, int] = {}
    for r in routing_leak:
        fam = r["winner_family"] or "none"
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    return {
        "n": n,
        "stage1_argmax_leak": argmax_leak / n,
        "stage1_argmax_leak_count": argmax_leak,
        "stage1_confident_leak": conf_leak / n,
        "stage1_confident_leak_count": conf_leak,
        "routing_leak": len(routing_leak) / n,
        "routing_leak_count": len(routing_leak),
        "routing_leak_by_family": fam_counts,
    }


def summarize_phase_a(rows: list[dict], conf_thr: float) -> dict:
    domains = sorted({r["domain"] for r in rows})
    summary = {
        "overall": _aggregate(rows, conf_thr),
        "overall_excl_nq_open": _aggregate(
            [r for r in rows if r["domain"] != TRIVIA_ADJACENT_DOMAIN], conf_thr),
        "per_domain": {d: _aggregate([r for r in rows if r["domain"] == d], conf_thr)
                       for d in domains},
    }
    return summary


# ---------------------------------------------------------------------------
# Phase B — conditional damage
# ---------------------------------------------------------------------------

def run_phase_b(pipeline, leaked_rows: list[dict], max_new_tokens: int) -> dict:
    norm = _normalizer()
    print(f"\n=== Phase B — conditional damage on {len(leaked_rows)} leaked queries ===",
          flush=True)
    if not leaked_rows:
        return {
            "n_leaked": 0,
            "answer_changed_rate": None,
            "note": ("0 end-to-end routing leaks — no adapter loaded for any held-out "
                     "query. Conditional damage is undefined; FR-by-construction "
                     "survives the open stream."),
            "cases": [],
        }

    cases: list[dict] = []
    changed = 0
    for i, r in enumerate(leaked_rows):
        q = r["text"]
        routed = pipeline.generate(q)
        frozen = pipeline.generate(q, skip_routing=True)
        is_changed = norm(routed.response) != norm(frozen.response)
        changed += int(is_changed)
        cases.append({
            "id": r["id"],
            "domain": r["domain"],
            "winner_adapter": routed.adapter_loaded,
            "winner_family": adapter_family(routed.adapter_loaded),
            "answer_changed": is_changed,
            "routed_answer": routed.response.strip()[:500],
            "frozen_answer": frozen.response.strip()[:500],
            "text": q,
        })
        print(f"  [{i + 1}/{len(leaked_rows)}] {r['domain']} "
              f"adapter={routed.adapter_loaded} changed={is_changed}", flush=True)

    return {
        "n_leaked": len(leaked_rows),
        "answer_changed_rate": changed / len(leaked_rows),
        "answer_changed_count": changed,
        "note": ("answer_changed_rate = harmful-leak fraction (routed answer differs "
                 "from frozen base on the same query). Unchanged = benign leak."),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--testset", default="data/openstream_heldout.json")
    p.add_argument("--output_dir", default="eval_results/openstream_stress")
    p.add_argument("--phase", choices=["a", "all"], default="all",
                   help="'a' = routing leak only (no LLM); 'all' = + conditional damage.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--skip_sanity", action="store_true")
    args = p.parse_args()

    import torch
    use_gpu = (not args.no_gpu) and torch.cuda.is_available()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = json.load(open(args.testset))["records"]
    print(f"Loaded {len(records)} held-out records from {args.testset}", flush=True)

    # Build router (lightweight for 'a', full pipeline for 'all' so Phase A and
    # Phase B share one routing object → identical decisions).
    pipeline = None
    if args.phase == "all":
        pipeline = build_full_pipeline(use_gpu, args.max_new_tokens)
        router = pipeline.router
    else:
        router = build_router_only(use_gpu)

    if not args.skip_sanity:
        wiring_sanity(router)

    # Phase A
    rows = run_phase_a(router, records)
    summary_a = summarize_phase_a(rows, PROD["domain_confidence_threshold"])

    with (out_dir / "predictions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    leak_report = {"config": PROD, "summary": summary_a}
    with (out_dir / "routing_leak.json").open("w") as f:
        json.dump(leak_report, f, indent=2)

    o = summary_a["overall"]
    x = summary_a["overall_excl_nq_open"]
    print("\n--- Phase A results ---")
    print(f"  overall          (n={o['n']}): stage1_argmax={o['stage1_argmax_leak']:.1%} "
          f"stage1_confident={o['stage1_confident_leak']:.1%} "
          f"routing_leak={o['routing_leak']:.1%} ({o['routing_leak_count']}) "
          f"by_family={o['routing_leak_by_family']}")
    print(f"  excl nq_open     (n={x['n']}): stage1_argmax={x['stage1_argmax_leak']:.1%} "
          f"stage1_confident={x['stage1_confident_leak']:.1%} "
          f"routing_leak={x['routing_leak']:.1%} ({x['routing_leak_count']}) "
          f"by_family={x['routing_leak_by_family']}")
    for d, s in summary_a["per_domain"].items():
        print(f"    {d:8}: routing_leak={s['routing_leak']:.1%} ({s['routing_leak_count']}/{s['n']}) "
              f"stage1_confident={s['stage1_confident_leak']:.1%} by_family={s['routing_leak_by_family']}")
    print(f"\n  → {out_dir/'routing_leak.json'}")

    # Phase B
    if args.phase == "all":
        leaked = [r for r in rows if r["winner_adapter"] is not None]
        damage = run_phase_b(pipeline, leaked, args.max_new_tokens)
        with (out_dir / "conditional_damage.json").open("w") as f:
            json.dump({"config": PROD, "summary": damage}, f, indent=2)
        print("\n--- Phase B results ---")
        if damage["n_leaked"] == 0:
            print(f"  {damage['note']}")
        else:
            print(f"  n_leaked={damage['n_leaked']} "
                  f"answer_changed_rate={damage['answer_changed_rate']:.1%} "
                  f"({damage['answer_changed_count']})")
        print(f"  → {out_dir/'conditional_damage.json'}")


if __name__ == "__main__":
    main()
