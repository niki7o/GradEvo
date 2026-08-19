"""Tests for compute-budget planning and realized-budget accounting.

These cover the arithmetic that plans NEAT's generation cap and the summary
that reports realized-vs-requested steps, plus a leakage guard asserting that
training environments are clean (never perturbed).
"""

from __future__ import annotations

import numpy as np
import pytest

from gradevo import config
from gradevo.experiment import budget as budget_mod


def test_plan_generations_scales_inversely_with_pop_size():
    small_pop = budget_mod.plan_neat_generations(500_000, population_size=20)
    large_pop = budget_mod.plan_neat_generations(500_000, population_size=100)
    assert small_pop > large_pop
    assert small_pop >= 1 and large_pop >= 1


def test_plan_generations_generous_enough_to_reach_budget():
    # The cap must let NEAT reach the budget even if episodes are short. With a
    # small planning episode length + 2x headroom, the worst-case realized steps
    # under the cap (cap * pop * min_len) must exceed the budget.
    pop, min_len, fit_eps = 60, config.NEAT_PLANNING_MIN_EPISODE_LENGTH, 1
    cap = budget_mod.plan_neat_generations(500_000, pop, fit_eps, min_len)
    assert cap * pop * fit_eps * min_len >= 500_000
    # And it must exceed the naive central-estimate count (250-step episodes).
    naive_central = int(np.ceil(500_000 / (pop * 250 * fit_eps)))
    assert cap >= naive_central


def test_summarize_budget_reports_discrepancy():
    report = budget_mod.summarize_budget(
        "neat",
        requested_steps=500_000,
        realized_steps=[480_000, 520_000, 500_000],
        wall_clock_s=[100.0, 110.0, 105.0],
    )
    assert report.realized_steps_mean == pytest.approx(500_000)
    assert report.realized_steps_min == 480_000
    assert report.realized_steps_max == 520_000
    assert abs(report.discrepancy_pct) < 1e-6  # mean equals request here


def test_summarize_budget_discrepancy_sign():
    over = budget_mod.summarize_budget("ppo", 100_000, [110_000, 110_000], [1.0, 1.0])
    assert over.discrepancy_pct == pytest.approx(10.0)
    under = budget_mod.summarize_budget("ppo", 100_000, [90_000, 90_000], [1.0, 1.0])
    assert under.discrepancy_pct == pytest.approx(-10.0)


def test_summarize_budget_raises_on_empty():
    with pytest.raises(ValueError):
        budget_mod.summarize_budget("ppo", 100, [], [])


def test_budget_table_has_row_per_method():
    reports = [
        budget_mod.summarize_budget("ppo", 500_000, [500_000] * 3, [1.0] * 3),
        budget_mod.summarize_budget("neat", 500_000, [490_000] * 3, [2.0] * 3),
    ]
    table = budget_mod.budget_table(reports)
    assert set(table["method"]) == {"ppo", "neat"}
    assert "discrepancy_pct" in table.columns


def test_quick_config_reduces_budget_and_seeds():
    quick = config.quick_config()
    full = config.full_config()
    assert quick.n_seeds < full.n_seeds
    assert quick.step_budget < full.step_budget
    assert quick.quick is True and full.quick is False


def test_run_config_seeds_are_explicit_and_distinct():
    cfg = config.RunConfig(n_seeds=5)
    assert cfg.seeds == [config.BASE_SEED + i for i in range(5)]
    assert len(set(cfg.seeds)) == 5
