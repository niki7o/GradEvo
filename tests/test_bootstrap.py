from __future__ import annotations
import numpy as np
import pytest
from gradevo.metrics import bootstrap as bs

def test_bootstrap_mean_ci_covers_true_mean_on_normal_sample():
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=5.0, scale=2.0, size=50)
    ci = bs.bootstrap_mean_ci(sample, seed=0)
    assert ci.low < ci.point < ci.high
    assert ci.low < 5.0 < ci.high
    assert ci.method == 'BCa'
    assert ci.level == 0.95

def test_bootstrap_mean_ci_degenerates_for_constant_sample():
    ci = bs.bootstrap_mean_ci([3.0, 3.0, 3.0], seed=0)
    assert ci.point == ci.low == ci.high == 3.0

def test_bootstrap_mean_ci_single_value_returns_point():
    ci = bs.bootstrap_mean_ci([7.5], seed=0)
    assert ci.point == 7.5
    assert ci.low == 7.5 == ci.high

def test_bootstrap_mean_ci_empty_returns_nan():
    ci = bs.bootstrap_mean_ci([], seed=0)
    assert np.isnan(ci.point) and np.isnan(ci.low) and np.isnan(ci.high)

def test_bootstrap_mean_difference_ci_direction_and_sign():
    rng = np.random.default_rng(1)
    a = rng.normal(10.0, 1.0, 40)
    b = rng.normal(5.0, 1.0, 40)
    ci = bs.bootstrap_mean_difference_ci(a, b, seed=0)
    assert ci.point > 0
    assert ci.low > 0

def test_bootstrap_mean_difference_ci_zero_when_same_distribution():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(0.0, 1.0, 200)
    ci = bs.bootstrap_mean_difference_ci(a, b, seed=0)
    assert ci.low < 0 < ci.high

def test_ci_as_row_serialization():
    ci = bs.CI(point=1.0, low=0.5, high=1.5)
    row = ci.as_row()
    assert row == {'point': 1.0, 'ci_low': 0.5, 'ci_high': 1.5, 'ci_level': 0.95, 'ci_method': 'BCa'}
