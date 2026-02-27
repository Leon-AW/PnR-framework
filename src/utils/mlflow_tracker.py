"""
MLflow Experiment Tracker
=========================

Central experiment tracking for the Patch-and-Route framework.

Provides:
- PnRTracker: context manager wrapping an MLflow run
- MLflowStepCallback: TrainerCallback for step-level loss logging
- get_or_create_experiment: idempotent experiment creation

If mlflow is not installed, all classes and functions degrade to silent no-ops
so training scripts work unchanged without the dependency.

Usage:
    from src.utils.mlflow_tracker import PnRTracker

    with PnRTracker("pnr-training", run_name="base_v1") as tracker:
        tracker.log_training_config(config)
        tracker.log_model_config(foundation_config, expert_config)
        # ... training happens inside this block ...
        tracker.log_metrics({"train_loss": 0.42})
        tracker.log_gpu_memory()
        tracker.log_adapter_artifact("checkpoints/base_v1")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MLflow import
# ---------------------------------------------------------------------------
try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    warnings.warn(
        "MLflow not installed — experiment tracking is disabled. "
        "Install with: pip install mlflow>=2.10.0",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Optional Transformers import (for TrainerCallback base class)
# ---------------------------------------------------------------------------
try:
    from transformers import TrainerCallback
except ImportError:
    # Provide a stub so MLflowStepCallback can still be defined
    class TrainerCallback:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_or_create_experiment(name: str, tracking_uri: str = "sqlite:///mlruns.db") -> str | None:
    """Return the experiment ID for *name*, creating it if it doesn't exist.

    Args:
        name: MLflow experiment name.
        tracking_uri: Local directory or server URL for the tracking store.

    Returns:
        Experiment ID string, or None when MLflow is unavailable.
    """
    if not _MLFLOW_AVAILABLE:
        return None

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
        logger.info(f"Created MLflow experiment '{name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using MLflow experiment '{name}' (id={experiment_id})")
    return experiment_id


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class PnRTracker:
    """Context manager that wraps a single MLflow run for PnR training.

    All public methods are no-ops when MLflow is not installed, so callers
    do not need to guard every logging call.

    Example::

        with PnRTracker("pnr-training", run_name="patch_geo_india") as t:
            t.log_training_config(config)
            t.log_model_config(foundation_cfg, expert_cfg)
            # ... train() ...
            t.log_metrics(metrics)
            t.log_gpu_memory()
            t.log_adapter_artifact(output_dir)
    """

    def __init__(
        self,
        experiment_name: str = "pnr-training",
        run_name: str | None = None,
        tracking_uri: str = "sqlite:///mlruns.db",
        tags: dict[str, str] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self._run = None
        self._active = False

    # ------------------------------------------------------------------
    # Context protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "PnRTracker":
        if not _MLFLOW_AVAILABLE:
            logger.warning("MLflow unavailable — PnRTracker is a no-op.")
            return self

        experiment_id = get_or_create_experiment(self.experiment_name, self.tracking_uri)
        self._run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=self.run_name,
            tags={"framework": "patch-and-route", **self.tags},
        )
        self._active = True
        logger.info(
            f"MLflow run started | id={self._run.info.run_id} | "
            f"experiment='{self.experiment_name}' | run='{self.run_name}'"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not _MLFLOW_AVAILABLE or not self._active:
            return

        if exc_type is not None:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error_type", exc_type.__name__ if exc_type else "")
            mlflow.set_tag("error", str(exc_val)[:500])
        else:
            mlflow.set_tag("status", "FINISHED")

        run_id = self._run.info.run_id
        mlflow.end_run()
        self._active = False
        logger.info(f"MLflow run finished | id={run_id}")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_training_config(self, config: Any) -> None:
        """Log all scalar fields of a TrainingConfig (or any dataclass) as params.

        Nested dicts/lists are serialised to their string representation so
        MLflow's 500-char param limit is respected.

        Args:
            config: A dataclass instance (e.g. TrainingConfig).
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return

        import dataclasses

        if dataclasses.is_dataclass(config):
            raw = dataclasses.asdict(config)
        elif hasattr(config, "__dict__"):
            raw = vars(config)
        else:
            logger.warning("log_training_config: unsupported config type %s", type(config))
            return

        params = {k: str(v)[:500] for k, v in raw.items()}
        mlflow.log_params(params)
        logger.debug("Logged %d training config params to MLflow", len(params))

    def log_model_config(
        self,
        foundation_config: Any | None = None,
        expert_config: Any | None = None,
    ) -> None:
        """Log model and LoRA configuration as namespaced MLflow params.

        Args:
            foundation_config: FrozenFoundationConfig instance.
            expert_config: ExpertConfig instance.
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return

        import dataclasses

        params: dict[str, str] = {}

        if foundation_config is not None and dataclasses.is_dataclass(foundation_config):
            for k, v in dataclasses.asdict(foundation_config).items():
                params[f"model.{k}"] = str(v)[:500]

        if expert_config is not None and dataclasses.is_dataclass(expert_config):
            for k, v in dataclasses.asdict(expert_config).items():
                params[f"lora.{k}"] = str(v)[:500]

        if params:
            mlflow.log_params(params)
            logger.debug("Logged %d model config params to MLflow", len(params))

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a metrics dictionary.  Non-numeric values are silently skipped.

        Args:
            metrics: Metric name → numeric value.
            step: Optional global step for the log entry.
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return

        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            mlflow.log_metrics(numeric, step=step)
            logger.debug("Logged %d metrics to MLflow (step=%s)", len(numeric), step)

    def log_gpu_memory(self, tag: str = "peak") -> None:
        """Log peak GPU memory allocated via torch.cuda.max_memory_allocated().

        Args:
            tag: Label suffix for the metric key (default "peak").
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return

        try:
            import torch

            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                mlflow.log_metric(f"gpu_memory_{tag}_gb", peak_gb)
                logger.info("GPU memory (%s): %.2f GB logged to MLflow", tag, peak_gb)
        except Exception as exc:
            logger.warning("Could not log GPU memory: %s", exc)

    def log_adapter_artifact(self, adapter_path: str | Path) -> None:
        """Record the adapter output path as an MLflow run tag.

        Stores the resolved absolute path rather than uploading files,
        keeping the MLflow store lightweight.  Use ``mlflow.log_artifacts()``
        directly if you want to upload the weights.

        Args:
            adapter_path: Path to the saved LoRA adapter directory.
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return

        resolved = str(Path(adapter_path).resolve())
        mlflow.set_tag("adapter_path", resolved)
        logger.info("Logged adapter_path tag: %s", resolved)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str | None:
        """Active MLflow run ID, or None if not running."""
        return self._run.info.run_id if self._run is not None else None


# ---------------------------------------------------------------------------
# Step-level callback
# ---------------------------------------------------------------------------

class MLflowStepCallback(TrainerCallback):
    """Logs Trainer metrics to the active MLflow run on every ``on_log`` event.

    Attach to ``SFTTrainer`` (or any HuggingFace Trainer) via the
    ``callbacks`` argument::

        from src.utils.mlflow_tracker import MLflowStepCallback

        trainer = SFTTrainer(..., callbacks=[MLflowStepCallback()])

    The callback is a no-op when there is no active MLflow run or when
    MLflow is not installed.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called by Trainer at every logging step."""
        if not _MLFLOW_AVAILABLE or logs is None:
            return

        try:
            active_run = mlflow.active_run()
            if active_run is None:
                return
        except Exception:
            return

        step = state.global_step if state is not None else None
        numeric = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        if numeric:
            mlflow.log_metrics(numeric, step=step)
