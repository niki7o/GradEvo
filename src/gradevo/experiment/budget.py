from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from gradevo import config

def plan_neat_generations(step_budget: int, population_size: int, fitness_episodes: int=config.NEAT_FITNESS_EPISODES, min_episode_length: int=config.NEAT_PLANNING_MIN_EPISODE_LENGTH) -> int:
    steps_per_gen = max(1, population_size * fitness_episodes * min_episode_length)
    planned = int(np.ceil(step_budget / steps_per_gen))
    return max(1, planned * 2)

@dataclass(frozen=True)
class BudgetReport:
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
        if self.requested_steps == 0:
            return 0.0
        return 100.0 * (self.realized_steps_mean - self.requested_steps) / self.requested_steps

def summarize_budget(method: str, requested_steps: int, realized_steps: List[int], wall_clock_s: List[float]) -> BudgetReport:
    if not realized_steps:
        raise ValueError('realized_steps must be non-empty')
    arr = np.asarray(realized_steps, dtype=np.float64)
    return BudgetReport(method=method, requested_steps=requested_steps, realized_steps_mean=float(arr.mean()), realized_steps_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0, realized_steps_min=int(arr.min()), realized_steps_max=int(arr.max()), wall_clock_mean_s=float(np.mean(wall_clock_s)) if wall_clock_s else 0.0, n_seeds=len(realized_steps))

def budget_table(reports: List[BudgetReport]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for r in reports:
        rows.append({'method': r.method, 'requested_steps': r.requested_steps, 'realized_steps_mean': round(r.realized_steps_mean, 1), 'realized_steps_std': round(r.realized_steps_std, 1), 'realized_steps_min': r.realized_steps_min, 'realized_steps_max': r.realized_steps_max, 'discrepancy_pct': round(r.discrepancy_pct, 2), 'wall_clock_mean_s': round(r.wall_clock_mean_s, 1), 'n_seeds': r.n_seeds})
    return pd.DataFrame(rows)
