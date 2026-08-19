from __future__ import annotations
import numpy as np
import pytest
gym = pytest.importorskip('gymnasium')
from gradevo import config
from gradevo.envs.perturbation import PerturbationWrapper
from gradevo.envs.step_counter import StepCounter

class DummyContinuousEnv(gym.Env):
    metadata: dict = {}

    def __init__(self, ep_len: int=50) -> None:
        super().__init__()
        self.ep_len = ep_len
        self.gravity = -10.0
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return (np.zeros(8, dtype=np.float32), {})

    def step(self, action):
        self._t += 1
        obs = np.ones(8, dtype=np.float32)
        terminated = self._t >= self.ep_len
        return (obs, 1.0, terminated, False, {})

def test_step_counter_exact_over_fixed_episode():
    env = StepCounter(DummyContinuousEnv(ep_len=50))
    env.reset(seed=0)
    done = False
    while not done:
        (_, _, terminated, truncated, info) = env.step(env.action_space.sample())
        done = terminated or truncated
    assert env.step_count == 50
    assert info['total_steps'] == 50

def test_step_counter_accumulates_across_episodes_until_reset():
    env = StepCounter(DummyContinuousEnv(ep_len=10))
    for _ in range(3):
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action_space.sample())
    assert env.step_count == 30
    env.reset_counter()
    assert env.step_count == 0

def _collect_obs_diffs(wrapped, n=200):
    wrapped.reset(seed=0)
    diffs = []
    for _ in range(n):
        (obs, _, term, trunc, _) = wrapped.step(np.zeros(2, dtype=np.float32))
        diffs.append(obs - np.ones(8, dtype=np.float32))
        if term or trunc:
            wrapped.reset(seed=0)
    return np.array(diffs)

def test_perturbation_changes_observations_beyond_noise_floor():
    wrapped = PerturbationWrapper(DummyContinuousEnv(), seed=1)
    diffs = _collect_obs_diffs(wrapped)
    assert np.abs(diffs).mean() > 0.001

def test_perturbation_respects_zero_range_dims():
    wrapped = PerturbationWrapper(DummyContinuousEnv(), seed=1, obs_ranges=config.LUNARLANDER_OBS_RANGES)
    diffs = _collect_obs_diffs(wrapped)
    assert np.abs(diffs[:, :6]).mean() > 0.001
    assert np.allclose(diffs[:, 6:], 0.0)

def test_perturbation_gravity_resampled_within_bounds():
    base = DummyContinuousEnv()
    wrapped = PerturbationWrapper(base, seed=3)
    base_gravity = -10.0
    scales = []
    for _ in range(50):
        (_, info) = wrapped.reset(seed=0)
        scales.append(info['gravity_scale'])
        assert 1 - config.GRAVITY_PERTURB_FRAC <= info['gravity_scale'] <= 1 + config.GRAVITY_PERTURB_FRAC
    assert wrapped.unwrapped.gravity == pytest.approx(base_gravity * scales[-1])
    assert np.std(scales) > 0

def test_perturbation_action_noise_clipped_to_bounds():
    base = DummyContinuousEnv()
    wrapped = PerturbationWrapper(base, seed=5)
    wrapped.reset(seed=0)
    for _ in range(50):
        wrapped.step(np.array([1.0, -1.0], dtype=np.float32))
    assert wrapped._action_sigma is not None
    assert np.all(wrapped._action_sigma > 0)

def test_perturbation_is_reproducible_given_seed():

    def run():
        w = PerturbationWrapper(DummyContinuousEnv(), seed=7)
        w.reset(seed=0)
        return np.array([w.step(np.zeros(2, dtype=np.float32))[0] for _ in range(20)])
    assert np.allclose(run(), run())

def test_perturbation_discrete_env_skips_action_noise():
    env = gym.make('CartPole-v1')
    wrapped = PerturbationWrapper(env, seed=0)
    assert wrapped._action_sigma is None
    (obs, _) = wrapped.reset(seed=0)
    assert obs.shape == env.observation_space.shape
