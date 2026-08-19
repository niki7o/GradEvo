"""Learning-curve alignment, variance bands, and normalized AUC (for H2).

PPO and NEAT report progress on different natural x-axes (SB3 rollouts vs. NEAT
generations), so before any curve-level comparison the curves must be placed on
a common environment-step grid by interpolation. Sample efficiency (H2) is then
measured as the normalized area under the fitness-vs-steps curve, which rewards
reaching good performance *earlier* in the budget.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

ArrayLike = Sequence[float]


def align_curves(
    steps: ArrayLike,
    fitness: ArrayLike,
    grid: np.ndarray,
) -> np.ndarray:
    """Interpolate a (steps, fitness) curve onto a shared step grid.

    Uses linear interpolation. Points on the grid before the first recorded
    step take the first fitness value; points after the last recorded step take
    the last value (``np.interp`` edge behavior), which is the conventional
    "hold last evaluation" assumption for learning curves.

    Args:
        steps: Monotonically non-decreasing environment-step values at which
            fitness was recorded.
        fitness: Fitness values aligned with ``steps``.
        grid: Target step grid to interpolate onto (monotonically increasing).

    Returns:
        Fitness values interpolated at each point of ``grid``.

    Raises:
        ValueError: If ``steps`` and ``fitness`` differ in length or are empty,
            or if ``steps`` is not sorted non-decreasing.
    """
    steps_arr = np.asarray(steps, dtype=np.float64)
    fitness_arr = np.asarray(fitness, dtype=np.float64)
    if steps_arr.shape != fitness_arr.shape:
        raise ValueError("steps and fitness must have the same shape")
    if steps_arr.size == 0:
        raise ValueError("cannot align an empty curve")
    if np.any(np.diff(steps_arr) < 0):
        raise ValueError("steps must be non-decreasing")
    return np.interp(np.asarray(grid, dtype=np.float64), steps_arr, fitness_arr)


def common_grid(step_budget: int, n_points: int = 100) -> np.ndarray:
    """Build the shared step grid used to align all curves.

    Args:
        step_budget: Upper bound of the grid (the target step budget).
        n_points: Number of grid points.

    Returns:
        A length-``n_points`` array from 0 to ``step_budget`` inclusive.
    """
    return np.linspace(0.0, float(step_budget), n_points)


def normalized_auc(
    steps: ArrayLike,
    fitness: ArrayLike,
    step_budget: int,
    fitness_lo: float,
    fitness_hi: float,
    n_points: int = 100,
) -> float:
    """Normalized area under a learning curve, in ``[0, 1]``.

    The curve is aligned to a common grid, fitness is min-max normalized to
    ``[0, 1]`` using shared bounds (so PPO and NEAT are on the same scale), and
    the mean of the normalized curve over the step axis is returned. A method
    that reaches high fitness early scores higher than one that reaches the same
    final fitness late, which is exactly the sample-efficiency notion in H2.

    Args:
        steps: Environment-step values where fitness was recorded.
        fitness: Fitness values aligned with ``steps``.
        step_budget: Step budget defining the grid's extent.
        fitness_lo: Lower bound for min-max normalization (shared across
            methods, e.g. the random-baseline fitness).
        fitness_hi: Upper bound for normalization (shared across methods).
        n_points: Grid resolution.

    Returns:
        Normalized AUC in ``[0, 1]``. Returns 0.0 if ``fitness_hi`` equals
        ``fitness_lo`` (degenerate scale).
    """
    if fitness_hi <= fitness_lo:
        return 0.0
    grid = common_grid(step_budget, n_points)
    aligned = align_curves(steps, fitness, grid)
    normed = (aligned - fitness_lo) / (fitness_hi - fitness_lo)
    normed = np.clip(normed, 0.0, 1.0)
    # Mean over a uniform grid == trapezoidal AUC / span, i.e. average height.
    return float(np.trapz(normed, grid) / (grid[-1] - grid[0]))


def seed_variance_band(
    curves: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize a stack of aligned per-seed curves into mean +/- std.

    Args:
        curves: Array of shape ``(n_seeds, n_points)`` of aligned curves.

    Returns:
        Tuple ``(mean, lower, upper)`` each of shape ``(n_points,)`` where
        ``lower``/``upper`` are ``mean -/+ 1 standard deviation`` across seeds.

    Raises:
        ValueError: If ``curves`` is not 2-dimensional.
    """
    arr = np.asarray(curves, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("curves must be a 2D array (n_seeds, n_points)")
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    return mean, mean - std, mean + std
