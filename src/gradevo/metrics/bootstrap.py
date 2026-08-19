from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.stats import bootstrap

@dataclass(frozen=True)
class CI:
    point: float
    low: float
    high: float
    level: float = 0.95
    method: str = 'BCa'
    n_resamples: int = 10000

    def as_row(self) -> dict:
        return {'point': self.point, 'ci_low': self.low, 'ci_high': self.high, 'ci_level': self.level, 'ci_method': self.method}

def _statistic_mean(sample: np.ndarray, axis: int=-1) -> np.ndarray:
    return np.mean(sample, axis=axis)

def bootstrap_mean_ci(values: Sequence[float], level: float=0.95, n_resamples: int=10000, seed: int=1234, method: str='BCa') -> CI:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return CI(point=float('nan'), low=float('nan'), high=float('nan'), level=level, method=method, n_resamples=n_resamples)
    point = float(np.mean(arr))
    if arr.size < 2 or float(np.std(arr)) == 0.0:
        return CI(point=point, low=point, high=point, level=level, method=method, n_resamples=n_resamples)
    rng = np.random.default_rng(seed)
    res = bootstrap((arr,), _statistic_mean, confidence_level=level, n_resamples=n_resamples, method=method, random_state=rng, vectorized=True)
    return CI(point=point, low=float(res.confidence_interval.low), high=float(res.confidence_interval.high), level=level, method=method, n_resamples=n_resamples)

def bootstrap_mean_difference_ci(a: Sequence[float], b: Sequence[float], level: float=0.95, n_resamples: int=10000, seed: int=1234, method: str='BCa') -> CI:
    a_arr = np.asarray(list(a), dtype=np.float64)
    b_arr = np.asarray(list(b), dtype=np.float64)
    if a_arr.size == 0 or b_arr.size == 0:
        return CI(point=float('nan'), low=float('nan'), high=float('nan'), level=level, method=method, n_resamples=n_resamples)
    point = float(np.mean(a_arr) - np.mean(b_arr))
    if a_arr.size < 2 or b_arr.size < 2:
        return CI(point=point, low=point, high=point, level=level, method=method, n_resamples=n_resamples)

    def stat(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)
    rng = np.random.default_rng(seed)
    res = bootstrap((a_arr, b_arr), stat, confidence_level=level, n_resamples=n_resamples, method=method, random_state=rng, vectorized=True)
    return CI(point=point, low=float(res.confidence_interval.low), high=float(res.confidence_interval.high), level=level, method=method, n_resamples=n_resamples)
