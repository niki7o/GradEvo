from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
ArrayLike = Sequence[float]

def align_curves(steps: ArrayLike, fitness: ArrayLike, grid: np.ndarray) -> np.ndarray:
    steps_arr = np.asarray(steps, dtype=np.float64)
    fitness_arr = np.asarray(fitness, dtype=np.float64)
    if steps_arr.shape != fitness_arr.shape:
        raise ValueError('steps and fitness must have the same shape')
    if steps_arr.size == 0:
        raise ValueError('cannot align an empty curve')
    if np.any(np.diff(steps_arr) < 0):
        raise ValueError('steps must be non-decreasing')
    return np.interp(np.asarray(grid, dtype=np.float64), steps_arr, fitness_arr)

def common_grid(step_budget: int, n_points: int=100) -> np.ndarray:
    return np.linspace(0.0, float(step_budget), n_points)

def normalized_auc(steps: ArrayLike, fitness: ArrayLike, step_budget: int, fitness_lo: float, fitness_hi: float, n_points: int=100) -> float:
    if fitness_hi <= fitness_lo:
        return 0.0
    grid = common_grid(step_budget, n_points)
    aligned = align_curves(steps, fitness, grid)
    normed = (aligned - fitness_lo) / (fitness_hi - fitness_lo)
    normed = np.clip(normed, 0.0, 1.0)
    return float(np.trapz(normed, grid) / (grid[-1] - grid[0]))

def seed_variance_band(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(curves, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError('curves must be a 2D array (n_seeds, n_points)')
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    return (mean, mean - std, mean + std)
