from __future__ import annotations
import numpy as np
import pytest
gym = pytest.importorskip('gymnasium')
cma = pytest.importorskip('cma')
from dataclasses import replace
from gradevo import config
from gradevo.agents import cmaes_agent, es_agent
from gradevo.experiment import runner

def test_train_cmaes_respects_step_budget_and_produces_policy():
    run_cfg = replace(config.quick_config('CartPole-v1'), step_budget=2000)
    (obs_dim, action_dim, continuous, low, high) = runner._env_dims('CartPole-v1')

    def eval_fn(policy, seed):
        return runner.mean_fitness(lambda s: gym.make('CartPole-v1'), policy, n_episodes=3, seed=seed)
    result = cmaes_agent.train_cmaes(make_clean_env=lambda s: runner.make_clean_env('CartPole-v1', s, step_counted=True), seed=0, run_cfg=run_cfg, eval_fitness_fn=eval_fn, obs_dim=obs_dim, action_dim=action_dim, continuous=continuous, action_low=low, action_high=high, population_size=8, n_curve_points=3)
    assert result.realized_steps >= run_cfg.step_budget * 0.8
    assert result.realized_steps <= run_cfg.step_budget + config.MAX_EPISODE_STEPS
    assert result.generations >= 1
    assert len(result.curve_steps) == len(result.curve_fitness) >= 1
    policy = es_agent.make_es_policy(result.theta, result.shapes, continuous, low, high)
    env = gym.make('CartPole-v1')
    (obs, _) = env.reset(seed=0)
    action = policy(np.asarray(obs))
    assert env.action_space.contains(int(action))
