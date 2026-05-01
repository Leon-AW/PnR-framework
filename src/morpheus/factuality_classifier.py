"""Binary factuality classifier for MORPHEUS System 5.

Predicts whether a query concerns a fact stored in the KnowledgeStore.
Architecture: frozen SentenceTransformer → <emb_dim> embedding →
LayerNorm → Linear(emb_dim, 256) → GELU → Dropout(0.2) →
Linear(256, 64) → GELU → Dropout(0.2) → Linear(64, 1) → Sigmoid

The embedding dim is auto-detected from the encoder at init time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class FactualityClassifier:
    """Binary classifier: does this query concern a stored fact?

    Architecture: frozen SentenceTransformer → <emb_dim> embedding →
    LayerNorm → Linear(emb_dim, 256) → GELU → Dropout(0.2) →
    Linear(256, 64) → GELU → Dropout(0.2) → Linear(64, 1) → Sigmoid

    The embedding dim is auto-detected from the encoder on first use, or
    supplied explicitly (required when loading from a saved checkpoint so
    the MLP can be reconstructed before weights are loaded).
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

        # Lazy-load the sentence transformer so importing this module on a
        # CPU-only machine (e.g. evaluation node) doesn't force a CUDA init.
        self._encoder: Any = None

        if embedding_dim is None:
            # Probe the encoder to learn the actual output dimension.
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
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
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

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return factuality scores in [0, 1], shape (N,)."""
        embs = self._embed(texts)
        return self._predict_from_embeddings(embs)

    def _predict_from_embeddings(self, embs: np.ndarray) -> np.ndarray:
        self._mlp.eval()
        with torch.no_grad():
            t = torch.from_numpy(embs).to(self._device)
            scores = self._mlp(t).squeeze(-1).cpu().numpy()
        return scores

    def predict_single(self, text: str) -> float:
        return float(self.predict([text])[0])

    def train_epoch(
        self,
        texts: list[str] | np.ndarray,
        labels: np.ndarray,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 64,
    ) -> float:
        """Train one epoch over pre-computed embeddings or raw texts.

        Accepts either raw texts (will embed them — slow) or a numpy array of
        pre-computed embeddings (preferred for repeated epochs).
        """
        if isinstance(texts, np.ndarray):
            embs = texts
        else:
            embs = self._embed(texts)

        self._mlp.train()
        criterion = nn.BCELoss()
        total_loss = 0.0
        n_batches = 0

        indices = np.arange(len(embs))
        np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            x = torch.from_numpy(embs[batch_idx]).to(self._device)
            y = torch.from_numpy(labels[batch_idx].astype(np.float32)).to(self._device)

            optimizer.zero_grad()
            preds = self._mlp(x).squeeze(-1)
            loss = criterion(preds, y)
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
        """Compute accuracy, AUC-ROC, F1, precision, recall.

        Accepts pre-computed embeddings or raw texts.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        if isinstance(texts, np.ndarray):
            embs = texts
        else:
            embs = self._embed(texts)

        scores = self._predict_from_embeddings(embs)
        preds = (scores >= 0.5).astype(int)

        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "auc_roc": float(roc_auc_score(labels, scores)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
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
        }
        with open(path / "classifier_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedding_model_path: str | None = None,
        device: str = "auto",
    ) -> FactualityClassifier:
        path = Path(path)
        with open(path / "classifier_config.json") as f:
            config = json.load(f)

        # Caller can override the embedding model path (e.g. local cache).
        if embedding_model_path is not None:
            config["embedding_model_path"] = embedding_model_path

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
