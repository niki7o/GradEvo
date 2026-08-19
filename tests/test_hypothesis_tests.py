"""Known-answer tests for the statistical machinery.

These tests validate the hypothesis-test functions against synthetic data with
a ground-truth answer: identical distributions must yield large p-values;
clearly separated distributions must yield small ones. A silent bug here would
invalidate every downstream scientific conclusion, so this module is the most
important in the suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gradevo import config
from gradevo.metrics import hypothesis_tests as ht
from gradevo.metrics import learning_curves as lc


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded RNG so every test is deterministic."""
    return np.random.default_rng(2024)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_mann_whitney_identical_distributions_high_p(rng):
    a = rng.normal(0, 1, 40)
    b = rng.normal(0, 1, 40)
    _, p = ht.mann_whitney(a, b)
    assert p > config.ALPHA_CORRECTED


def test_mann_whitney_separated_distributions_low_p(rng):
    a = rng.normal(10, 1, 40)
    b = rng.normal(0, 1, 40)
    _, p = ht.mann_whitney(a, b)
    assert p < 1e-6


def test_rank_biserial_sign_and_bounds(rng):
    a = rng.normal(5, 1, 50)
    b = rng.normal(0, 1, 50)
    effect = ht.rank_biserial(a, b)
    assert 0.9 < effect <= 1.0  # a >> b -> near +1
    assert -1.0 <= ht.rank_biserial(b, a) < -0.9


def test_rank_biserial_zero_for_equal():
    a = np.arange(20.0)
    assert abs(ht.rank_biserial(a, a)) < 1e-9


def test_wilcoxon_paired_detects_consistent_shift(rng):
    base = rng.normal(0, 1, 30)
    shifted = base + 2.0  # every pair increases
    _, p = ht.wilcoxon_paired(shifted, base, alternative="greater")
    assert p < 1e-4


def test_wilcoxon_raises_on_all_zero_diff():
    a = np.ones(10)
    with pytest.raises(ValueError):
        ht.wilcoxon_paired(a, a)


def test_normality_check_accepts_normal_rejects_uniform_tails(rng):
    normal = rng.normal(0, 1, 200)
    heavy = rng.standard_cauchy(200)
    assert ht.normality_check(normal) is True
    assert ht.normality_check(heavy) is False


def test_normality_check_handles_degenerate():
    assert ht.normality_check([1.0, 1.0, 1.0]) is False
    assert ht.normality_check([1.0]) is False


def test_bonferroni_reject_threshold():
    assert ht.bonferroni_reject(0.01, config.ALPHA_CORRECTED) is True
    assert ht.bonferroni_reject(0.02, config.ALPHA_CORRECTED) is False


# --------------------------------------------------------------------------- #
# H0-H3
# --------------------------------------------------------------------------- #
def test_h0_gate_passes_when_trained_beats_baselines(rng):
    trained = rng.normal(200, 20, 20)
    random_b = rng.normal(-150, 30, 20)
    heuristic_b = rng.normal(0, 30, 20)
    results = ht.test_h0_baselines(trained, random_b, heuristic_b, "ppo")
    assert all(r.reject_null for r in results)


def test_h0_gate_fails_when_no_better_than_baseline(rng):
    trained = rng.normal(0, 20, 20)
    baseline = rng.normal(0, 20, 20)
    results = ht.test_h0_baselines(trained, baseline, baseline, "ppo")
    assert not any(r.reject_null for r in results)


def test_h1_fails_to_reject_for_equal_methods(rng):
    ppo = rng.normal(150, 25, 20)
    neat = rng.normal(150, 25, 20)
    result = ht.test_h1_final_fitness(ppo, neat)
    assert result.reject_null is False
    assert result.hypothesis_id == "H1"
    assert result.alpha == pytest.approx(config.ALPHA_CORRECTED)


def test_h1_rejects_for_clearly_different_methods(rng):
    ppo = rng.normal(250, 10, 20)
    neat = rng.normal(50, 10, 20)
    result = ht.test_h1_final_fitness(ppo, neat)
    assert result.reject_null is True
    assert result.p_value < config.ALPHA_CORRECTED


def test_h2_rejects_when_ppo_auc_dominates(rng):
    neat_auc = rng.uniform(0.2, 0.4, 20)
    ppo_auc = neat_auc + rng.uniform(0.1, 0.2, 20)  # PPO strictly higher per seed
    result = ht.test_h2_sample_efficiency(ppo_auc, neat_auc)
    assert result.reject_null is True
    assert result.effect_size > 0


def test_h2_fails_to_reject_when_no_advantage(rng):
    ppo_auc = rng.uniform(0.3, 0.5, 20)
    neat_auc = rng.uniform(0.3, 0.5, 20)
    result = ht.test_h2_sample_efficiency(ppo_auc, neat_auc)
    assert result.reject_null is False


def test_h3_rejects_when_neat_drop_smaller(rng):
    neat_drop = rng.normal(10, 5, 20)
    ppo_drop = rng.normal(80, 10, 20)
    result = ht.test_h3_robustness(ppo_drop, neat_drop)
    assert result.reject_null is True


def test_h3_fails_to_reject_when_equal_drops(rng):
    drop = rng.normal(50, 10, 20)
    result = ht.test_h3_robustness(drop, drop + rng.normal(0, 1, 20))
    assert result.reject_null is False


# --------------------------------------------------------------------------- #
# Suite runner glue
# --------------------------------------------------------------------------- #
def _synthetic_tables(rng, n_seeds=20):
    """Build tidy fitness/curve tables where PPO strongly dominates."""
    seeds = list(range(config.BASE_SEED, config.BASE_SEED + n_seeds))
    rows = []
    for s in seeds:
        rows.append({"method": "ppo", "condition": "clean", "seed": s, "fitness": rng.normal(220, 15)})
        rows.append({"method": "ppo", "condition": "perturbed", "seed": s, "fitness": rng.normal(180, 15)})
        rows.append({"method": "neat", "condition": "clean", "seed": s, "fitness": rng.normal(120, 15)})
        rows.append({"method": "neat", "condition": "perturbed", "seed": s, "fitness": rng.normal(100, 15)})
        rows.append({"method": "random", "condition": "clean", "seed": s, "fitness": rng.normal(-150, 20)})
        rows.append({"method": "heuristic", "condition": "clean", "seed": s, "fitness": rng.normal(-20, 20)})
    fitness_df = pd.DataFrame(rows)

    curve_rows = []
    grid = [0, 100_000, 250_000, 500_000]
    for s in seeds:
        for step in grid:
            frac = step / 500_000
            curve_rows.append({"method": "ppo", "seed": s, "step": step, "fitness": -150 + 370 * frac})
            curve_rows.append({"method": "neat", "seed": s, "step": step, "fitness": -150 + 270 * frac ** 2})
    curves_df = pd.DataFrame(curve_rows)
    return fitness_df, curves_df


def test_run_pre_registered_suite_shapes_and_gate(rng):
    fitness_df, curves_df = _synthetic_tables(rng)
    results = ht.run_pre_registered_suite(fitness_df, curves_df, 500_000)
    assert set(results) == {"H0", "H1", "H2", "H3"}
    assert len(results["H0"]) == 4  # 2 methods x 2 baselines
    assert all(r.reject_null for r in results["H0"])  # both methods beat both baselines
    table = ht.results_to_dataframe(results)
    assert len(table) == 7  # 4 H0 rows + H1 + H2 + H3


def test_suite_h1_rejects_for_dominant_ppo(rng):
    fitness_df, curves_df = _synthetic_tables(rng)
    results = ht.run_pre_registered_suite(fitness_df, curves_df, 500_000)
    assert results["H1"].reject_null is True  # PPO ~220 vs NEAT ~120


def test_per_seed_auc_in_unit_range(rng):
    _, curves_df = _synthetic_tables(rng)
    auc = ht.per_seed_auc(curves_df, "ppo", 500_000, -150.0, 220.0)
    assert ((auc >= 0.0) & (auc <= 1.0)).all()
