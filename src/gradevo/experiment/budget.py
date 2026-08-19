"""Compute-budget matching logic and honest realized-budget accounting.

The shared budget is defined as **total environment interaction steps**. This
module (a) plans how many NEAT generations to *allow* so that its expected step
consumption lands near the budget, and (b) summarizes the *realized* step counts
for both methods so the notebook can report the discrepancy honestly instead of
silently equalizing it.

Nothing here estimates the realized count -- realized counts always come from
the step-counting wrapper. The planning arithmetic is used only to bound the
generation loop; the true terminator is the measured step budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from gradevo import config


def plan_neat_generations(
    step_budget: int,
    population_size: int,
    fitness_episodes: int = config.NEAT_FITNESS_EPISODES,
    min_episode_length: int = config.NEAT_PLANNING_MIN_EPISODE_LENGTH,
) -> int:
    """Plan a generous NEAT generation *cap* for a step budget.

    The cap is a safety bound, not the primary terminator -- the **measured step
    budget** ends evolution (see :func:`gradevo.agents.neat_agent.train_neat`).
    To guarantee the loop can always reach the budget even when episodes are
    short (early NEAT policies often crash quickly), the cap is derived from a
    deliberately *small* per-episode step estimate:
    ``steps_per_generation ~= population_size * fitness_episodes *
    min_episode_length``. If episodes run longer, the budget is reached in fewer
    generations and the loop breaks early -- so over-provisioning the cap is
    safe and under-provisioning (which would leave NEAT below budget) is not.

    Args:
        step_budget: Target total environment steps.
        population_size: NEAT population size.
        fitness_episodes: Episodes averaged per genome per generation.
        min_episode_length: Conservative (small) episode-length floor used only
            to make the cap generous.

    Returns:
        A generation cap (>= 1) to pass to the NEAT training loop.
    """
    steps_per_gen = max(1, population_size * fitness_episodes * min_episode_length)
    # 2x headroom on top of the already-conservative small-episode estimate.
    planned = int(np.ceil(step_budget / steps_per_gen))
    return max(1, planned * 2)


@dataclass(frozen=True)
class BudgetReport:
    """Realized-vs-requested budget summary for one method.

    Attributes:
        method: Method name.
        requested_steps: The shared target step budget.
        realized_steps_mean: Mean realized step count across seeds.
        realized_steps_std: Standard deviation of realized steps across seeds.
        realized_steps_min: Minimum realized steps across seeds.
        realized_steps_max: Maximum realized steps across seeds.
        wall_clock_mean_s: Mean wall-clock training time across seeds.
        n_seeds: Number of seeds summarized.
    """

    method: str
    requested_steps: int
    realized_steps_mean: float
    realized_steps_std: float
    realized_steps_min: int
    realized_steps_max: int
    wall_clock_mean_s: float
    n_seeds: int

    @property
    def discrepancy_pct(self) -> float:
        """Percent difference of mean realized steps from the request."""
        if self.requested_steps == 0:
            return 0.0
        return 100.0 * (self.realized_steps_mean - self.requested_steps) / self.requested_steps


def summarize_budget(
    method: str,
    requested_steps: int,
    realized_steps: List[int],
    wall_clock_s: List[float],
) -> BudgetReport:
    """Aggregate per-seed realized step counts into a :class:`BudgetReport`.

    Args:
        method: Method name.
        requested_steps: The shared target budget.
        realized_steps: Realized step count per seed.
        wall_clock_s: Wall-clock training time per seed.

    Returns:
        A :class:`BudgetReport` with mean/std/min/max realized steps.

    Raises:
        ValueError: If ``realized_steps`` is empty.
    """
    if not realized_steps:
        raise ValueError("realized_steps must be non-empty")
    arr = np.asarray(realized_steps, dtype=np.float64)
    return BudgetReport(
        method=method,
        requested_steps=requested_steps,
        realized_steps_mean=float(arr.mean()),
        realized_steps_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        realized_steps_min=int(arr.min()),
        realized_steps_max=int(arr.max()),
        wall_clock_mean_s=float(np.mean(wall_clock_s)) if wall_clock_s else 0.0,
        n_seeds=len(realized_steps),
    )


def budget_table(reports: List[BudgetReport]) -> pd.DataFrame:
    """Render a list of budget reports as a tidy DataFrame for the notebook.

    Args:
        reports: One :class:`BudgetReport` per method.

    Returns:
        A DataFrame with one row per method and columns for requested/realized
        steps, discrepancy percentage, and mean wall-clock time.
    """
    rows: List[Dict[str, object]] = []
    for r in reports:
        rows.append(
            {
                "method": r.method,
                "requested_steps": r.requested_steps,
                "realized_steps_mean": round(r.realized_steps_mean, 1),
                "realized_steps_std": round(r.realized_steps_std, 1),
                "realized_steps_min": r.realized_steps_min,
                "realized_steps_max": r.realized_steps_max,
                "discrepancy_pct": round(r.discrepancy_pct, 2),
                "wall_clock_mean_s": round(r.wall_clock_mean_s, 1),
                "n_seeds": r.n_seeds,
            }
        )
    return pd.DataFrame(rows)
