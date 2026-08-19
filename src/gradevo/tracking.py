"""MLflow experiment-tracking helpers (SQLite backend).

Tracking is deliberately optional: if MLflow is not installed or tracking is
disabled, these helpers degrade to no-ops so the experiment still runs. When
enabled, each (method, seed) run is logged with its locked seed, realized step
count, wall-clock time, and final fitness, giving a queryable audit trail.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from gradevo import config

logger = logging.getLogger(__name__)


def setup_tracking(enabled: bool = True) -> bool:
    """Configure MLflow to use the local SQLite backend.

    Args:
        enabled: If ``False``, tracking is skipped entirely.

    Returns:
        ``True`` if tracking is active, ``False`` if disabled or unavailable.
    """
    if not enabled:
        return False
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; tracking disabled")
        return False
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow tracking at %s", config.MLFLOW_TRACKING_URI)
    return True


@contextmanager
def track_run(
    run_name: str,
    params: Dict[str, Any],
    enabled: bool = True,
) -> Iterator[Optional[Any]]:
    """Context manager that opens an MLflow run and logs params.

    Args:
        run_name: Human-readable run name (e.g. ``"ppo_seed1234"``).
        params: Parameters to log at run start (seed, budget, etc.).
        enabled: If ``False`` or MLflow is unavailable, yields ``None``.

    Yields:
        The active MLflow run object, or ``None`` when tracking is inactive.
    """
    if not enabled:
        yield None
        return
    try:
        import mlflow
    except ImportError:
        yield None
        return
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        yield run


def log_metrics(metrics: Dict[str, float], enabled: bool = True) -> None:
    """Log a dict of scalar metrics to the active MLflow run.

    Args:
        metrics: Metric name -> value.
        enabled: If ``False`` or MLflow is unavailable, this is a no-op.
    """
    if not enabled:
        return
    try:
        import mlflow
    except ImportError:
        return
    mlflow.log_metrics(metrics)
