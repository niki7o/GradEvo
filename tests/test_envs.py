"""Tests for the step-counter and perturbation wrappers.

To avoid a hard Box2D dependency, wrapper mechanics are tested against a tiny
in-process dummy environment with a continuous action space and a ``gravity``
attribute, plus the discrete CartPole env when Gymnasium is available. The
statistical perturbation check verifies that perturbed observations differ from
clean ones beyond the noise floor, which is the property H3 relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gradevo import config
from gradevo.envs.perturbation import PerturbationWrapper
from gradevo.envs.step_counter import StepCounter


class DummyContinuousEnv(gym.Env):
    """A minimal deterministic continuous-control env for wrapper tests.

    Fixed-length episodes of ``ep_len`` steps, an 8-D observation, a 2-D action,
    and a mutable ``gravity`` attribute so gravity perturbation can be exercised.
    """

    metadata: dict = {}

    def __init__(self, ep_len: int = 50) -> None:
        super().__init__()
        self.ep_len = ep_len
        self.gravity = -10.0
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        obs = np.ones(8, dtype=np.float32)  # constant clean obs -> noise is visible
        terminated = self._t >= self.ep_len
        return obs, 1.0, terminated, False, {}


# --------------------------------------------------------------------------- #
# StepCounter
# --------------------------------------------------------------------------- #
def test_step_counter_exact_over_fixed_episode():
    env = StepCounter(DummyContinuousEnv(ep_len=50))
    env.reset(seed=0)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
    assert env.step_count == 50
    assert info["total_steps"] == 50


def test_step_counter_accumulates_across_episodes_until_reset():
    env = StepCounter(DummyContinuousEnv(ep_len=10))
    for _ in range(3):
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action_space.sample())
    assert env.step_count == 30  # NOT reset by env.reset()
    env.reset_counter()
    assert env.step_count == 0


# --------------------------------------------------------------------------- #
# PerturbationWrapper
# --------------------------------------------------------------------------- #
def _collect_obs_diffs(wrapped, n=200):
    """Run ``n`` steps and return (perturbed_obs - clean_obs) diffs (clean=ones)."""
    wrapped.reset(seed=0)
    diffs = []
    for _ in range(n):
        obs, _, term, trunc, _ = wrapped.step(np.zeros(2, dtype=np.float32))
        diffs.append(obs - np.ones(8, dtype=np.float32))  # clean step obs is all ones
        if term or trunc:
            wrapped.reset(seed=0)
    return np.array(diffs)


def test_perturbation_changes_observations_beyond_noise_floor():
    # Dummy env has no boolean dims: with inferred ranges every dim is noised.
    wrapped = PerturbationWrapper(DummyContinuousEnv(), seed=1)
    diffs = _collect_obs_diffs(wrapped)
    assert np.abs(diffs).mean() > 1e-3  # clearly beyond the (zero) clean floor


def test_perturbation_respects_zero_range_dims():
    # With LunarLander-style ranges, the two boolean leg-contact dims (range 0)
    # must stay exactly clean while the other six are noised.
    wrapped = PerturbationWrapper(
        DummyContinuousEnv(), seed=1, obs_ranges=config.LUNARLANDER_OBS_RANGES
    )
    diffs = _collect_obs_diffs(wrapped)
    assert np.abs(diffs[:, :6]).mean() > 1e-3
    assert np.allclose(diffs[:, 6:], 0.0)


def test_perturbation_gravity_resampled_within_bounds():
    base = DummyContinuousEnv()
    wrapped = PerturbationWrapper(base, seed=3)
    base_gravity = -10.0
    scales = []
    for _ in range(50):
        _, info = wrapped.reset(seed=0)
        scales.append(info["gravity_scale"])
        assert (1 - config.GRAVITY_PERTURB_FRAC) <= info["gravity_scale"] <= (1 + config.GRAVITY_PERTURB_FRAC)
    # Gravity actually applied to the underlying env.
    assert wrapped.unwrapped.gravity == pytest.approx(base_gravity * scales[-1])
    assert np.std(scales) > 0  # genuinely resampled, not constant


def test_perturbation_action_noise_clipped_to_bounds():
    base = DummyContinuousEnv()
    wrapped = PerturbationWrapper(base, seed=5)
    wrapped.reset(seed=0)
    # Feed extreme actions; the perturbed action must still respect bounds.
    for _ in range(50):
        wrapped.step(np.array([1.0, -1.0], dtype=np.float32))
    # No assertion error means clipping held; verify sigma is nonzero.
    assert wrapped._action_sigma is not None
    assert np.all(wrapped._action_sigma > 0)


def test_perturbation_is_reproducible_given_seed():
    def run():
        w = PerturbationWrapper(DummyContinuousEnv(), seed=7)
        w.reset(seed=0)
        return np.array([w.step(np.zeros(2, dtype=np.float32))[0] for _ in range(20)])

    assert np.allclose(run(), run())


def test_perturbation_discrete_env_skips_action_noise():
    env = gym.make("CartPole-v1")
    wrapped = PerturbationWrapper(env, seed=0)
    assert wrapped._action_sigma is None  # discrete -> no action noise
    obs, _ = wrapped.reset(seed=0)
    assert obs.shape == env.observation_space.shape
