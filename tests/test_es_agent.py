from __future__ import annotations
import numpy as np
import pytest
gym = pytest.importorskip('gymnasium')
from gradevo import config
from gradevo.agents import es_agent
from gradevo.experiment import runner

def test_flatten_unflatten_roundtrip():
    rng = np.random.default_rng(0)
    params = es_agent._init_mlp_params(rng, obs_dim=4, action_dim=2, hidden=(8, 8))
    shapes = [p.shape for p in params]
    flat = es_agent._flatten(params)
    reconstructed = es_agent._unflatten(flat, shapes)
    for (a, b) in zip(params, reconstructed):
        np.testing.assert_array_equal(a, b)

def test_forward_discrete_returns_int_within_range():
    rng = np.random.default_rng(0)
    params = es_agent._init_mlp_params(rng, obs_dim=4, action_dim=2, hidden=(8,))
    action = es_agent._forward(np.zeros(4), params, continuous=False, action_low=np.zeros(2), action_high=np.ones(2))
    assert 0 <= int(action) < 2

def test_forward_continuous_respects_action_bounds():
    rng = np.random.default_rng(0)
    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])
    params = es_agent._init_mlp_params(rng, obs_dim=8, action_dim=2, hidden=(8,))
    for _ in range(20):
        obs = rng.normal(0, 3, 8)
        action = es_agent._forward(obs, params, continuous=True, action_low=low, action_high=high)
        assert np.all(action >= low - 1e-06)
        assert np.all(action <= high + 1e-06)

def test_rank_shape_is_centered_and_ordered():
    fitness = np.array([10.0, 30.0, 20.0, 40.0])
    shaped = es_agent._rank_shape(fitness)
    assert abs(shaped.mean()) < 1e-09
    assert shaped[np.argmax(fitness)] == shaped.max()
    assert shaped[np.argmin(fitness)] == shaped.min()

def test_train_es_respects_step_budget_and_produces_policy():
    from dataclasses import replace
    run_cfg = replace(config.quick_config('CartPole-v1'), step_budget=2000)
    (obs_dim, action_dim, continuous, low, high) = runner._env_dims('CartPole-v1')

    def eval_fn(policy, seed):
        return runner.mean_fitness(lambda s: gym.make('CartPole-v1'), policy, n_episodes=3, seed=seed)
    result = es_agent.train_es(make_clean_env=lambda s: runner.make_clean_env('CartPole-v1', s, step_counted=True), seed=0, run_cfg=run_cfg, eval_fitness_fn=eval_fn, obs_dim=obs_dim, action_dim=action_dim, continuous=continuous, action_low=low, action_high=high, population_size=8, n_curve_points=3)
    assert result.realized_steps >= run_cfg.step_budget * 0.8
    assert result.realized_steps <= run_cfg.step_budget + config.MAX_EPISODE_STEPS
    assert result.generations >= 1
    assert len(result.curve_steps) == len(result.curve_fitness) >= 1
    policy = es_agent.make_es_policy(result.theta, result.shapes, continuous, low, high)
    env = gym.make('CartPole-v1')
    (obs, _) = env.reset(seed=0)
    action = policy(np.asarray(obs))
    assert env.action_space.contains(int(action))

def test_train_es_rejects_odd_population_size():
    from dataclasses import replace
    run_cfg = replace(config.quick_config('CartPole-v1'), step_budget=500)
    with pytest.raises(ValueError, match='antithetic'):
        es_agent.train_es(make_clean_env=lambda s: runner.make_clean_env('CartPole-v1', s, step_counted=True), seed=0, run_cfg=run_cfg, eval_fitness_fn=lambda p, s: 0.0, obs_dim=4, action_dim=2, continuous=False, action_low=np.zeros(2), action_high=np.ones(2), population_size=7)
