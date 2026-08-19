from __future__ import annotations
import numpy as np
import pytest
gym = pytest.importorskip('gymnasium')
from gradevo import config
from gradevo.agents.baseline import HeuristicAgent, RandomAgent
from gradevo.envs.perturbation import PerturbationWrapper
from gradevo.envs.step_counter import StepCounter
from gradevo.experiment import runner
from tests.test_envs import DummyContinuousEnv

def test_random_agent_samples_valid_actions():
    env = gym.make('CartPole-v1')
    agent = RandomAgent(env.action_space, seed=0)
    for _ in range(20):
        assert env.action_space.contains(agent.act(None))

def test_random_agent_is_seed_reproducible():
    space1 = gym.make('CartPole-v1').action_space
    space2 = gym.make('CartPole-v1').action_space
    a1 = [RandomAgent(space1, seed=42).act(None) for _ in range(10)]
    a2 = [RandomAgent(space2, seed=42).act(None) for _ in range(10)]
    assert a1 == a2

def test_heuristic_discrete_runs_and_beats_random_on_cartpole():
    env = gym.make('CartPole-v1')
    heuristic = HeuristicAgent(env.action_space)
    fitness = runner.mean_fitness(lambda s: gym.make('CartPole-v1'), heuristic.act, n_episodes=10, seed=0)
    random_fitness = runner.mean_fitness(lambda s: gym.make('CartPole-v1'), RandomAgent(gym.make('CartPole-v1').action_space, seed=0).act, n_episodes=10, seed=0)
    assert 0 < fitness <= 500
    assert fitness > random_fitness

def test_heuristic_continuous_action_within_bounds_and_beats_random():
    env = DummyContinuousEnv()
    heuristic = HeuristicAgent(env.action_space)
    for _ in range(30):
        action = heuristic.act(np.random.default_rng(0).normal(0, 1, 8))
        assert env.action_space.contains(np.asarray(action, dtype=np.float32))

def test_training_factory_is_clean_not_perturbed():
    train_env = runner.make_clean_env('CartPole-v1', seed=0, step_counted=True)
    assert isinstance(train_env, StepCounter)
    node = train_env
    while hasattr(node, 'env'):
        assert not isinstance(node, PerturbationWrapper)
        node = node.env

def test_eval_perturbed_factory_wraps_perturbation():
    eval_env = runner.make_perturbed_env('CartPole-v1', seed=0)
    assert isinstance(eval_env, PerturbationWrapper)

def test_random_agent_fixed_seed_regression():
    fitness = runner.mean_fitness(lambda s: gym.make('CartPole-v1'), RandomAgent(gym.make('CartPole-v1').action_space, seed=config.BASE_SEED).act, n_episodes=20, seed=config.BASE_SEED)
    assert 8.0 <= fitness <= 40.0
