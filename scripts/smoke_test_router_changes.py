#!/usr/bin/env python3
"""
Smoke test for the three PnR router architecture changes
========================================================

Exercises the new wiring without SLURM / without loading the foundation LLM:

  1. Per-chunk routing anchors (sidecar npz round-trip via AdapterManifest).
  2. Per-adapter calibrated thresholds (`AdapterEntry.metadata` honoured by
     `CentroidRouter.route()`).
  3. Always-on Source-Replay (winner adapter contributes retrieved chunks).
  4. `CentroidRouter.load()` auto-initialises Source-Replay from sidecar
     FAISS files.

Uses a deterministic toy embedding (one-hot over a small vocab) so we can
assert exact routing decisions without depending on a real embedding model.
Run from the repo root:

    python scripts/smoke_test_router_changes.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.routing import CentroidRouter, AdapterManifest


# ---------------------------------------------------------------------------
# Toy embedder — maps each lowercase ASCII letter to a unit vector in R^26.
# Sentence embedding = L2-normalised average of letter vectors. Cosine
# similarity between two strings then ≈ vocab overlap, which is enough for
# the routing assertions in this smoke test.
# ---------------------------------------------------------------------------

VOCAB = "abcdefghijklmnopqrstuvwxyz"


def toy_embed(text: str) -> np.ndarray:
    vec = np.zeros(len(VOCAB), dtype=np.float32)
    for ch in text.lower():
        if ch in VOCAB:
            vec[VOCAB.index(ch)] += 1.0
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n
    return vec


def toy_embed_batch(texts: list[str]) -> np.ndarray:
    return np.vstack([toy_embed(t) for t in texts])


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return matrix / norms


# ---------------------------------------------------------------------------
# Test 1 — Manifest sidecar round-trip
# ---------------------------------------------------------------------------

def test_manifest_sidecar_roundtrip(tmpdir: Path) -> None:
    print("\n[1] Manifest sidecar round-trip (per-chunk anchors)")
    print("-" * 60)

    manifest = AdapterManifest()
    manifest.register("adapter_a", "/tmp/a", timestamp=1.0, adapter_type="t1")
    manifest.register("adapter_b", "/tmp/b", timestamp=2.0, adapter_type="t2")

    big_anchors_a = normalise_rows(np.random.RandomState(0).randn(1024, 32).astype(np.float32))
    small_anchors_b = normalise_rows(np.random.RandomState(1).randn(8, 32).astype(np.float32))

    manifest.update_cluster_centroids(
        "adapter_a", [big_anchors_a[i] for i in range(big_anchors_a.shape[0])]
    )
    manifest.update_cluster_centroids(
        "adapter_b", [small_anchors_b[i] for i in range(small_anchors_b.shape[0])]
    )
    manifest._entries["adapter_a"].metadata["similarity_threshold"] = 0.31
    manifest._entries["adapter_b"].metadata["similarity_threshold"] = 0.62

    out_path = tmpdir / "manifest.json"
    manifest.save(out_path)

    sidecar = tmpdir / "cluster_centroids.npz"
    assert sidecar.exists(), "sidecar npz was not written"
    print(f"  manifest.json    : {out_path.stat().st_size:,} bytes")
    print(f"  sidecar npz      : {sidecar.stat().st_size:,} bytes")

    loaded = AdapterManifest.load(out_path)
    assert loaded.num_adapters == 2
    a = loaded["adapter_a"]
    b = loaded["adapter_b"]
    assert a.num_clusters == 1024, f"expected 1024 anchors for A, got {a.num_clusters}"
    assert b.num_clusters == 8, f"expected 8 anchors for B, got {b.num_clusters}"
    assert np.allclose(np.vstack(a.cluster_centroids), big_anchors_a, atol=1e-6)
    assert np.allclose(np.vstack(b.cluster_centroids), small_anchors_b, atol=1e-6)
    assert loaded["adapter_a"].metadata["similarity_threshold"] == 0.31
    assert loaded["adapter_b"].metadata["similarity_threshold"] == 0.62
    print("  ✓ anchors + per-adapter thresholds round-trip cleanly")


# ---------------------------------------------------------------------------
# Test 2 — Per-adapter threshold gating in route()
# ---------------------------------------------------------------------------

def test_per_adapter_threshold_gating() -> None:
    print("\n[2] Per-adapter calibrated thresholds gate routing")
    print("-" * 60)

    router = CentroidRouter(
        embedding_fn=toy_embed,
        similarity_threshold=0.05,  # very permissive global fallback
        always_on_replay=False,     # isolate Change 2 from Change 3
        use_gpu=False,
    )

    geo_qs = ["germany", "germany berlin", "germany munich"]
    geo_anchors = toy_embed_batch(geo_qs)
    router._manifest.register(
        adapter_id="geo",
        adapter_path="/tmp/geo",
        timestamp=1.0,
        adapter_type="patch_geo",
    )
    router._manifest.update_cluster_centroids("geo", [geo_anchors[i] for i in range(len(geo_qs))])
    router._manifest["geo"].metadata["similarity_threshold"] = 0.95

    broad_qs = ["alpha", "beta", "gamma", "delta"]
    broad_anchors = toy_embed_batch(broad_qs)
    router._manifest.register(
        adapter_id="broad",
        adapter_path="/tmp/broad",
        timestamp=2.0,
        adapter_type="patch_broad",
    )
    router._manifest.update_cluster_centroids(
        "broad", [broad_anchors[i] for i in range(len(broad_qs))]
    )
    router._manifest["broad"].metadata["similarity_threshold"] = 0.30

    near_geo_query = "george"
    sim_to_geo = max(toy_embed(near_geo_query) @ a for a in geo_anchors)
    sim_to_broad = max(toy_embed(near_geo_query) @ a for a in broad_anchors)
    print(f"  query='{near_geo_query}' raw sims: geo={sim_to_geo:.3f}, broad={sim_to_broad:.3f}")
    print(f"  per-adapter τ:                  geo=0.950, broad=0.300")

    result = router.route(near_geo_query)
    print(f"  → winner_adapter = {result.winner_adapter}")
    assert result.winner_adapter != "geo", "geo should be rejected by its own high τ"

    result_exact = router.route("germany")
    print(f"  query='germany' → winner = {result_exact.winner_adapter}")
    assert result_exact.winner_adapter == "geo", "exact match should pass geo's τ"
    print("  ✓ per-adapter thresholds correctly gate routing")


# ---------------------------------------------------------------------------
# Test 3 — Always-on Source-Replay surfaces winner-adapter chunks
# ---------------------------------------------------------------------------

def test_always_on_source_replay() -> None:
    print("\n[3] Always-on Source-Replay returns winner chunks")
    print("-" * 60)

    router = CentroidRouter(
        embedding_fn=toy_embed,
        similarity_threshold=0.0,
        retrieval_threshold=0.0,
        always_on_replay=True,
        winner_replay_top_k=2,
        use_gpu=False,
    )

    samples = [
        {"edited_question": "kappa", "answer": "K"},
        {"edited_question": "kappa lambda", "answer": "K+L"},
        {"edited_question": "lambda mu", "answer": "L+M"},
    ]
    qs = [s["edited_question"] for s in samples]
    anchors = toy_embed_batch(qs)

    router._manifest.register(
        adapter_id="single",
        adapter_path="/tmp/single",
        timestamp=1.0,
        adapter_type="patch",
    )
    router._manifest.update_cluster_centroids(
        "single", [anchors[i] for i in range(len(qs))]
    )
    router._manifest["single"].metadata["similarity_threshold"] = 0.0

    router.initialize_source_replay()
    router.index_samples_for_replay("single", samples)

    result = router.route("kappa")
    assert result.winner_adapter == "single"
    assert result.retrieved_context, "expected non-empty retrieved context"
    print(f"  winner = {result.winner_adapter}")
    print("  retrieved context (first 200 chars):")
    print("    " + result.retrieved_context[:200].replace("\n", "\n    "))

    winner_match = next(m for m in result.all_matches if m.is_winner)
    chunk_count = len(winner_match.retrieved_context or [])
    assert chunk_count > 0, "winner match should carry its own retrieved chunks"
    print(f"  ✓ winner adapter returned {chunk_count} chunks (always-on replay)")


# ---------------------------------------------------------------------------
# Test 4 — CentroidRouter.load() auto-initialises Source-Replay from sidecars
# ---------------------------------------------------------------------------

def test_load_auto_initialises_source_replay(tmpdir: Path) -> None:
    print("\n[4] CentroidRouter.load() auto-initialises Source-Replay")
    print("-" * 60)

    state_dir = tmpdir / "router_state"
    state_dir.mkdir()

    router = CentroidRouter(
        embedding_fn=toy_embed,
        similarity_threshold=0.0,
        always_on_replay=True,
        use_gpu=False,
        store_dir=state_dir,
    )

    samples = [
        {"edited_question": "alpha beta", "answer": "AB"},
        {"edited_question": "beta gamma", "answer": "BG"},
    ]
    anchors = toy_embed_batch([s["edited_question"] for s in samples])

    router._manifest.register("toy", "/tmp/toy", timestamp=1.0, adapter_type="t")
    router._manifest.update_cluster_centroids("toy", [anchors[i] for i in range(2)])
    router._manifest["toy"].metadata["similarity_threshold"] = 0.0

    router.initialize_source_replay(state_dir)
    router.index_samples_for_replay("toy", samples)
    router.save(state_dir)

    reloaded = CentroidRouter.load(
        path=state_dir,
        embedding_fn=toy_embed,
        always_on_replay=True,
        similarity_threshold=0.0,
        use_gpu=False,
    )

    assert reloaded._source_replay is not None, "Source-Replay should auto-init on load"
    print(f"  Source-Replay store created: {reloaded._source_replay is not None}")

    result = reloaded.route("alpha beta gamma")
    assert result.winner_adapter == "toy"
    assert result.retrieved_context, "expected retrieved context after reload"
    print(f"  winner after reload: {result.winner_adapter}")
    print(f"  retrieved context length: {len(result.retrieved_context)} chars")
    print("  ✓ retrieval works end-to-end after CentroidRouter.load()")


# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("PnR Router Architecture Changes — Smoke Test")
    print("=" * 70)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        test_manifest_sidecar_roundtrip(tmpdir)
        test_per_adapter_threshold_gating()
        test_always_on_source_replay()
        test_load_auto_initialises_source_replay(tmpdir)
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"ALL SMOKE TESTS PASSED in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
