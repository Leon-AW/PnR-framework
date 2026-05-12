"""3-class domain classifier — Stage 1 of the two-stage router (Phase 4).

Predicts whether a query belongs to the CounterFact, SituatedQA, or
out-of-distribution-TriviaQA family. The CentroidRouter consumes the
prediction to mask which adapters Stage 2 considers, closing NF-1
(``routing_acc=0`` on SQA caused by embedding-space overlap forcing
SQA-trained adapters into the global-fallback τ regime).

Architecture mirrors ``src.morpheus.factuality_classifier.FactualityClassifier``
deliberately — same MiniLM-L6-v2 encoder, same MLP topology, same
LayerNorm + GELU + Dropout — only the head differs:

  * factuality: ``Linear(64, 1) → Sigmoid`` + BCELoss
  * domain:     ``Linear(64, 3)``         + CrossEntropyLoss (softmax in predict)

The classes match ``CLASS_LABELS`` in ``scripts/build_domain_classifier_data.py``
exactly, by integer index, so the training script can use ``label`` directly
as the CrossEntropyLoss target.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_LABELS: list[str] = ["cf", "sqa", "ood_trivia"]


class DomainClassifier:
    """3-class classifier: cf / sqa / ood_trivia.

    Inference contract (consumed by ``CentroidRouter``):

        clf = DomainClassifier.load("checkpoints/domain_classifier")
        probs = clf.predict_single("Where was Albert Einstein born?")
        # → {"cf": 0.04, "sqa": 0.10, "ood_trivia": 0.86}

    The wrapper exposes both the per-class probabilities (so the router
    can apply its confidence threshold to the most-likely class) and the
    raw argmax (for callers that just want the predicted domain).
    """

    def __init__(
        self,
        embedding_model_path: str,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        device: str = "auto",
        embedding_dim: int | None = None,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [256, 64]

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._embedding_model_path = embedding_model_path
        self._hidden_dims = hidden_dims
        self._dropout = dropout
        self._encoder: Any = None
        self._n_classes = len(CLASS_LABELS)

        if embedding_dim is None:
            enc = self._get_encoder()
            embedding_dim = enc.get_sentence_embedding_dimension()

        self._embedding_dim = embedding_dim
        self._mlp = self._build_mlp(embedding_dim, hidden_dims, dropout).to(self._device)

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, self._n_classes))
        return nn.Sequential(*layers)

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                self._embedding_model_path,
                device=str(self._device),
            )
        return self._encoder

    def _embed(self, texts: list[str]) -> np.ndarray:
        enc = self._get_encoder()
        return enc.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def _logits_from_embeddings(self, embs: np.ndarray) -> torch.Tensor:
        self._mlp.eval()
        with torch.no_grad():
            t = torch.from_numpy(embs).to(self._device)
            return self._mlp(t)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Per-class softmax probabilities, shape (N, 3)."""
        embs = self._embed(texts)
        return self._predict_proba_from_embeddings(embs)

    def _predict_proba_from_embeddings(self, embs: np.ndarray) -> np.ndarray:
        logits = self._logits_from_embeddings(embs)
        return F.softmax(logits, dim=-1).cpu().numpy()

    def predict_single(self, text: str) -> dict[str, float]:
        """Return ``{class_label: probability}`` for a single query."""
        probs = self.predict_proba([text])[0]
        return {label: float(p) for label, p in zip(CLASS_LABELS, probs)}

    def train_epoch(
        self,
        texts: list[str] | np.ndarray,
        labels: np.ndarray,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 64,
    ) -> float:
        embs = texts if isinstance(texts, np.ndarray) else self._embed(texts)

        self._mlp.train()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        n_batches = 0

        indices = np.arange(len(embs))
        np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            x = torch.from_numpy(embs[batch_idx]).to(self._device)
            y = torch.from_numpy(labels[batch_idx].astype(np.int64)).to(self._device)

            optimizer.zero_grad()
            logits = self._mlp(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(
        self,
        texts: list[str] | np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64,
    ) -> dict:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        embs = texts if isinstance(texts, np.ndarray) else self._embed(texts)

        # Batched forward to avoid OOM on large val sets.
        preds: list[np.ndarray] = []
        probs: list[np.ndarray] = []
        for start in range(0, len(embs), batch_size):
            batch = embs[start:start + batch_size]
            logits = self._logits_from_embeddings(batch)
            batch_probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs.append(batch_probs)
            preds.append(batch_probs.argmax(axis=-1))
        all_preds = np.concatenate(preds)
        all_probs = np.concatenate(probs, axis=0)

        labels_int = labels.astype(int)

        return {
            "accuracy": float(accuracy_score(labels_int, all_preds)),
            "f1_macro": float(f1_score(labels_int, all_preds, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(labels_int, all_preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(labels_int, all_preds, average="macro", zero_division=0)),
            "per_class_f1": {
                CLASS_LABELS[i]: float(
                    f1_score(labels_int, all_preds, labels=[i], average="macro", zero_division=0)
                )
                for i in range(self._n_classes)
            },
            "_predictions": all_preds,
            "_probs": all_probs,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._mlp.state_dict(), path / "classifier.pt")

        config = {
            "embedding_model_path": self._embedding_model_path,
            "hidden_dims": self._hidden_dims,
            "dropout": self._dropout,
            "embedding_dim": self._embedding_dim,
            "labels": CLASS_LABELS,
        }
        with open(path / "classifier_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding_model_path: str | None = None,
        device: str = "auto",
    ) -> DomainClassifier:
        path = Path(path)
        with open(path / "classifier_config.json") as f:
            config = json.load(f)

        if embedding_model_path is not None:
            config["embedding_model_path"] = embedding_model_path

        # Sanity-check class ordering on load: a checkpoint built with a
        # different label list would silently produce wrong masks.
        saved_labels = config.get("labels", CLASS_LABELS)
        if saved_labels != CLASS_LABELS:
            raise ValueError(
                f"Domain classifier checkpoint at {path} was saved with labels "
                f"{saved_labels}, but current code expects {CLASS_LABELS}. "
                "Class indices would mismatch — refusing to load."
            )

        classifier = cls(
            embedding_model_path=config["embedding_model_path"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
            device=device,
            embedding_dim=config.get("embedding_dim"),
        )
        state = torch.load(
            path / "classifier.pt",
            map_location=classifier._device,
            weights_only=True,
        )
        classifier._mlp.load_state_dict(state)
        return classifier
