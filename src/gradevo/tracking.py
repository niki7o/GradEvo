from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
from gradevo import config
logger = logging.getLogger(__name__)

def setup_tracking(enabled: bool=True) -> bool:
    if not enabled:
        return False
    try:
        import mlflow
    except ImportError:
        logger.warning('mlflow not installed; tracking disabled')
        return False
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    logger.info('MLflow tracking at %s', config.MLFLOW_TRACKING_URI)
    return True

@contextmanager
def track_run(run_name: str, params: Dict[str, Any], enabled: bool=True) -> Iterator[Optional[Any]]:
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

def log_metrics(metrics: Dict[str, float], enabled: bool=True) -> None:
    if not enabled:
        return
    try:
        import mlflow
    except ImportError:
        return
    mlflow.log_metrics(metrics)
