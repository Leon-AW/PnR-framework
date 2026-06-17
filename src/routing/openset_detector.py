"""Open-set / OOD detector for the Stage-1 routing gate (leak mitigation).

Why this exists
---------------
The open-stream stress test (``scripts/run_openstream_stress.py``) showed the
Stage-1 softmax classifier leaks ~31% of genuinely out-of-distribution queries
into expert adapters. The root cause is structural: 364/381 leaks are
*confident* misclassifications (argmax != ood_trivia at p >= 0.7). A four-way
softmax with no reject region cannot express "outside all known domains", so a
finance/legal/medical query gets confidently mapped onto cf/sqa/qm.

A confidence- or margin-based reject cannot fix this — the failures *are*
confident. The principled successor named in the thesis' own Future Work is a
**Mahalanobis distance to the trained class manifolds** in the embedding feature
space (Lee et al. 2018): a far-but-confident query is far from every in-domain
manifold even when the softmax head is sure. This module implements exactly that
as a *separate, toggleable* component — the ``CentroidRouter`` consults it only
to veto a confident in-adapter-domain prediction, so the production "before"
behaviour is recovered verbatim by simply not attaching a detector.

Design notes
------------
* Feature space = MiniLM-L6-v2 sentence embeddings, ``normalize_embeddings=True``,
  float32 — *byte-identical* to ``DomainClassifier._embed`` so the detector and
  the classifier reason in the same space.
* Modelled manifolds = the **adapter** classes only (``cf``, ``sqa``, ``qm``).
  ``ood_trivia`` is handled by the classifier's own class (→ frozen base), so it
  is deliberately NOT a manifold: adding it would only lower OOD scores and make
  the gate more permissive (worse at catching leaks).
* Covariance = **tied** (shared across the modelled classes) with Ledoit-Wolf
  shrinkage. Per-class covariance is infeasible here (qm has ~450 fit samples in
  384 dims → singular); the tied estimate pools all within-class-centered samples
  and is well-conditioned.
* kNN distance (Sun et al. 2022) is offered as a non-parametric alternative that
  shares the same embedding bank, for robustness reporting.

The detector is fitted on in-domain *training* data (D_fit) and its reject
threshold is calibrated on a disjoint in-domain *validation* split (D_cal) at a
fixed in-domain false-reject budget. Neither ever sees the fresh OOD test set —
see ``tasks/todo.md`` (validity design).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Adapter classes whose manifolds we model. ood_trivia is intentionally excluded
# (see module docstring). Order is cosmetic; means are stored by class name.
ADAPTER_CLASSES: tuple[str, ...] = ("cf", "sqa", "qm")


class OpenSetDetector:
    """Mahalanobis / kNN open-set score over MiniLM query embeddings.

    Typical lifecycle::

        det = OpenSetDetector(embedding_model_path="sentence-transformers/all-MiniLM-L6-v2")
        det.fit(fit_texts, fit_class_names)          # D_fit (in-domain train)
        det.calibrate(cal_texts, alpha=0.05)          # D_cal (in-domain val)
        score = det.score_texts(["What is the CEO's tenure clause?"])[0]
        is_ood = det.is_ood_texts(["..."])[0]         # score > self.threshold
        det.save("checkpoints/openset_detector")

    Higher score == more out-of-distribution. ``is_ood`` is ``score > threshold``.
    """

    def __init__(
        self,
        embedding_model_path: str,
        method: str = "mahalanobis",
        knn_k: int = 5,
        shrinkage: str | float = "ledoit_wolf",
        device: str = "auto",
    ) -> None:
        if method not in ("mahalanobis", "knn"):
            raise ValueError(f"method must be 'mahalanobis' or 'knn', got {method!r}")
        self.embedding_model_path = embedding_model_path
        self.method = method
        self.knn_k = knn_k
        self.shrinkage = shrinkage
        self._device = device
        self._encoder: Any = None

        # Fitted state.
        self.classes_: list[str] = []
        self.means_: dict[str, np.ndarray] = {}      # class -> (d,)
        self.precision_: np.ndarray | None = None     # tied Σ^{-1}, (d, d)
        self._bank: np.ndarray | None = None          # (N, d) normalized, for kNN
        self.embedding_dim: int | None = None

        # Calibration state.
        self.threshold: float | None = None            # global (min-dist) reject bar
        self.thresholds_: dict[str, float] = {}         # per-class reject bars (primary)
        self.calibration_alpha: float | None = None

    # ------------------------------------------------------------------ encode
    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = self._device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._encoder = SentenceTransformer(self.embedding_model_path, device=dev)
        return self._encoder

    def _embed(self, texts: list[str]) -> np.ndarray:
        """MiniLM normalized embeddings — identical to DomainClassifier._embed."""
        enc = self._get_encoder()
        return enc.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    @staticmethod
    def _as_embeddings(detector, texts_or_embs):
        return texts_or_embs  # placeholder kept for symmetry; unused

    # --------------------------------------------------------------------- fit
    def fit(self, texts_or_embs, class_names: list[str]) -> "OpenSetDetector":
        """Fit per-class means + tied (shrunk) precision and the kNN bank.

        Args:
            texts_or_embs: list[str] of queries, or a precomputed (N, d) float32
                embedding array (already MiniLM-normalized).
            class_names: length-N class label per row (e.g. "cf"/"sqa"/"qm").
                Rows whose class is not in ADAPTER_CLASSES are ignored.
        """
        embs = self._coerce_embeddings(texts_or_embs)
        class_names = list(class_names)
        if len(class_names) != len(embs):
            raise ValueError(f"got {len(class_names)} labels for {len(embs)} rows")

        mask = np.array([c in ADAPTER_CLASSES for c in class_names])
        embs = embs[mask]
        labels = [c for c, keep in zip(class_names, mask) if keep]
        if len(embs) == 0:
            raise ValueError(f"no rows with class in {ADAPTER_CLASSES}")

        self.embedding_dim = int(embs.shape[1])
        self.classes_ = sorted(set(labels))

        # Per-class means + within-class-centered residuals (for tied covariance).
        centered_parts: list[np.ndarray] = []
        self.means_ = {}
        for c in self.classes_:
            idx = [i for i, lab in enumerate(labels) if lab == c]
            cls_embs = embs[idx]
            mu = cls_embs.mean(axis=0)
            self.means_[c] = mu.astype(np.float32)
            centered_parts.append(cls_embs - mu)
        centered = np.concatenate(centered_parts, axis=0)

        # Tied covariance with shrinkage → precision (Σ^{-1}).
        self.precision_ = self._fit_precision(centered)

        # kNN bank = the fit embeddings themselves (normalized).
        self._bank = embs.copy()
        return self

    def _fit_precision(self, centered: np.ndarray) -> np.ndarray:
        d = centered.shape[1]
        if self.shrinkage == "ledoit_wolf":
            from sklearn.covariance import LedoitWolf
            # assume_centered: residuals are already within-class centered.
            lw = LedoitWolf(assume_centered=True).fit(centered)
            return lw.precision_.astype(np.float32)
        # Manual diagonal shrinkage: Σ = (1-γ)·Σ_emp + γ·(tr/d)·I
        gamma = float(self.shrinkage)
        cov = (centered.T @ centered) / max(len(centered) - 1, 1)
        mu_trace = np.trace(cov) / d
        cov = (1.0 - gamma) * cov + gamma * mu_trace * np.eye(d, dtype=cov.dtype)
        return np.linalg.inv(cov).astype(np.float32)

    # ------------------------------------------------------------------- score
    def score_embeddings(self, embs: np.ndarray) -> np.ndarray:
        """OOD score per row (higher = more out-of-distribution)."""
        embs = np.asarray(embs, dtype=np.float32)
        if self.method == "mahalanobis":
            return self._mahalanobis_score(embs)
        return self._knn_score(embs)

    def score_texts(self, texts: list[str]) -> np.ndarray:
        return self.score_embeddings(self._embed(list(texts)))

    def class_distances(self, embs: np.ndarray) -> np.ndarray:
        """Mahalanobis distance to each class manifold, shape (N, len(classes_))."""
        if self.precision_ is None:
            raise RuntimeError("detector not fitted (mahalanobis)")
        embs = np.asarray(embs, dtype=np.float32)
        dists = np.empty((len(embs), len(self.classes_)), dtype=np.float64)
        for j, c in enumerate(self.classes_):
            diff = embs - self.means_[c]                  # (N, d)
            # row-wise quadratic form: sum_d sum_k diff[n,d] P[d,k] diff[n,k]
            dists[:, j] = np.einsum("nd,dk,nk->n", diff, self.precision_, diff)
        return dists

    def _mahalanobis_score(self, embs: np.ndarray) -> np.ndarray:
        # Global score = distance to the nearest class manifold.
        return self.class_distances(embs).min(axis=1)

    def _knn_score(self, embs: np.ndarray) -> np.ndarray:
        if self._bank is None:
            raise RuntimeError("detector not fitted")
        # Cosine distance (embeddings are normalized) to the k-th nearest bank pt.
        sims = embs @ self._bank.T                        # (N, M)
        dists = 1.0 - sims
        k = min(self.knn_k, dists.shape[1])
        part = np.partition(dists, k - 1, axis=1)[:, :k]
        return part.max(axis=1)                           # k-th NN distance

    # --------------------------------------------------------------- calibrate
    def calibrate(self, texts_or_embs, alpha: float = 0.05) -> float:
        """Set the GLOBAL ``self.threshold`` so in-domain false-reject ≤ alpha.

        Global (nearest-manifold) threshold — kept as an ablation. A single bar is
        unfair across classes of very different spread/sample-size (e.g. qm); the
        per-class calibration below is the primary path.
        """
        embs = self._coerce_embeddings(texts_or_embs)
        scores = self.score_embeddings(embs)
        self.threshold = float(np.quantile(scores, 1.0 - alpha))
        self.calibration_alpha = float(alpha)
        return self.threshold

    def calibrate_per_class(
        self, texts_or_embs, class_names: list[str], alpha: float = 0.05
    ) -> dict[str, float]:
        """Set ``self.thresholds_[c]`` so each class gets ≤ alpha false-reject.

        For class ``c``, τ_c = (1-alpha) quantile of the *distance-to-c* over the
        D_cal queries whose true class is ``c``. At inference the router supplies
        the Stage-1 predicted class and we reject iff distance-to-that-class > τ.
        This decouples the per-class budget from class spread, fixing the qm
        imbalance a single global bar produces.
        """
        if self.method != "mahalanobis":
            raise RuntimeError("per-class calibration is mahalanobis-only")
        embs = self._coerce_embeddings(texts_or_embs)
        class_names = list(class_names)
        dmat = self.class_distances(embs)                 # (N, C)
        col = {c: j for j, c in enumerate(self.classes_)}
        self.thresholds_ = {}
        for c in self.classes_:
            rows = [i for i, lab in enumerate(class_names) if lab == c]
            if not rows:
                continue
            own = dmat[rows, col[c]]                       # distance to own manifold
            self.thresholds_[c] = float(np.quantile(own, 1.0 - alpha))
        self.calibration_alpha = float(alpha)
        return self.thresholds_

    def is_ood_for_class(self, embs: np.ndarray, predicted_classes: list[str]) -> np.ndarray:
        """Per-class reject: distance to the predicted class > that class's τ.

        ``predicted_classes`` is the Stage-1 argmax per row (the router has it).
        Rows whose predicted class was never calibrated fall back to the global
        threshold on the nearest-manifold score.
        """
        if not self.thresholds_:
            raise RuntimeError("detector not per-class calibrated")
        embs = np.asarray(embs, dtype=np.float32)
        dmat = self.class_distances(embs)
        col = {c: j for j, c in enumerate(self.classes_)}
        out = np.zeros(len(embs), dtype=bool)
        for i, pc in enumerate(predicted_classes):
            if pc in self.thresholds_:
                out[i] = dmat[i, col[pc]] > self.thresholds_[pc]
            elif self.threshold is not None:
                out[i] = dmat[i].min() > self.threshold
        return out

    def is_ood_embeddings(self, embs: np.ndarray) -> np.ndarray:
        """Standalone (no Stage-1) reject: assign nearest manifold, use its τ.

        Used for the read-only OOD preview where no Stage-1 prediction exists.
        Prefers per-class thresholds (keyed on the argmin class); falls back to
        the global threshold.
        """
        embs = np.asarray(embs, dtype=np.float32)
        if self.thresholds_ and self.method == "mahalanobis":
            dmat = self.class_distances(embs)
            nearest = dmat.argmin(axis=1)
            out = np.zeros(len(embs), dtype=bool)
            for i, j in enumerate(nearest):
                c = self.classes_[j]
                out[i] = dmat[i, j] > self.thresholds_.get(c, np.inf)
            return out
        if self.threshold is None:
            raise RuntimeError("detector not calibrated — call calibrate() first")
        return self.score_embeddings(embs) > self.threshold

    def is_ood_texts(self, texts: list[str]) -> np.ndarray:
        return self.is_ood_embeddings(self._embed(list(texts)))

    def predict_single(self, text: str, predicted_class: str | None = None) -> dict:
        """Router-integration convenience: reject flag for one query.

        If ``predicted_class`` (the Stage-1 argmax) is given, use its per-class τ;
        otherwise fall back to the standalone nearest-manifold rule.
        """
        embs = self._embed([text])
        if predicted_class is not None and predicted_class in self.thresholds_:
            is_ood = bool(self.is_ood_for_class(embs, [predicted_class])[0])
            score = float(self.class_distances(embs)[0, self.classes_.index(predicted_class)])
            tau = self.thresholds_[predicted_class]
        else:
            is_ood = bool(self.is_ood_embeddings(embs)[0])
            score = float(self.score_embeddings(embs)[0])
            tau = self.threshold
        return {"ood_score": score, "is_ood": is_ood, "threshold": tau,
                "predicted_class": predicted_class}

    # -------------------------------------------------------------------- util
    def _coerce_embeddings(self, texts_or_embs) -> np.ndarray:
        if isinstance(texts_or_embs, np.ndarray):
            return texts_or_embs.astype(np.float32)
        return self._embed(list(texts_or_embs))

    # ------------------------------------------------------------- save / load
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "openset_state.npz",
            precision=self.precision_ if self.precision_ is not None else np.empty(0),
            bank=self._bank if self._bank is not None else np.empty(0),
            **{f"mean__{c}": self.means_[c] for c in self.classes_},
        )
        config = {
            "embedding_model_path": self.embedding_model_path,
            "method": self.method,
            "knn_k": self.knn_k,
            "shrinkage": self.shrinkage,
            "classes": self.classes_,
            "embedding_dim": self.embedding_dim,
            "threshold": self.threshold,
            "thresholds_per_class": self.thresholds_,
            "calibration_alpha": self.calibration_alpha,
            "adapter_classes": list(ADAPTER_CLASSES),
        }
        with open(path / "openset_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding_model_path: str | None = None,
        device: str = "auto",
    ) -> "OpenSetDetector":
        path = Path(path)
        with open(path / "openset_config.json") as f:
            config = json.load(f)
        det = cls(
            embedding_model_path=embedding_model_path or config["embedding_model_path"],
            method=config["method"],
            knn_k=config["knn_k"],
            shrinkage=config["shrinkage"],
            device=device,
        )
        det.classes_ = config["classes"]
        det.embedding_dim = config["embedding_dim"]
        det.threshold = config["threshold"]
        det.thresholds_ = config.get("thresholds_per_class", {})
        det.calibration_alpha = config["calibration_alpha"]

        state = np.load(path / "openset_state.npz")
        prec = state["precision"]
        det.precision_ = prec if prec.size else None
        bank = state["bank"]
        det._bank = bank if bank.size else None
        det.means_ = {c: state[f"mean__{c}"] for c in det.classes_}
        return det
