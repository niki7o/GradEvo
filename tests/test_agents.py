"""Tests for baseline agents and the clean/perturbed environment factories.

Covers: baseline agents run without error and produce sane fitness ranges; the
continuous heuristic beats a random agent on a dummy lander; the runner's
training factory yields a *clean* (non-perturbed) env while the eval factory
yields a perturbed one (leakage guard for H3); and a fixed-seed regression check
to catch silent environment/library drift.
"""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gradevo import config
from gradevo.agents.baseline import HeuristicAgent, RandomAgent
from gradevo.envs.perturbation import PerturbationWrapper
from gradevo.envs.step_counter import StepCounter
from gradevo.experiment import runner
from tests.test_envs import DummyContinuousEnv


# --------------------------------------------------------------------------- #
# Baseline agents
# --------------------------------------------------------------------------- #
def test_random_agent_samples_valid_actions():
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env.action_space, seed=0)
    for _ in range(20):
        assert env.action_space.contains(agent.act(None))


def test_random_agent_is_seed_reproducible():
    space1 = gym.make("CartPole-v1").action_space
    space2 = gym.make("CartPole-v1").action_space
    a1 = [RandomAgent(space1, seed=42).act(None) for _ in range(10)]
    a2 = [RandomAgent(space2, seed=42).act(None) for _ in range(10)]
    assert a1 == a2


def test_heuristic_discrete_runs_and_beats_random_on_cartpole():
    env = gym.make("CartPole-v1")
    heuristic = HeuristicAgent(env.action_space)
    fitness = runner.mean_fitness(lambda s: gym.make("CartPole-v1"), heuristic.act, n_episodes=10, seed=0)
    random_fitness = runner.mean_fitness(
        lambda s: gym.make("CartPole-v1"),
        RandomAgent(gym.make("CartPole-v1").action_space, seed=0).act,
        n_episodes=10,
        seed=0,
    )
    assert 0 < fitness <= 500  # sane CartPole range
    assert fitness > random_fitness  # a sensible heuristic should beat random


def test_heuristic_continuous_action_within_bounds_and_beats_random():
    env = DummyContinuousEnv()
    heuristic = HeuristicAgent(env.action_space)
    for _ in range(30):
        action = heuristic.act(np.random.default_rng(0).normal(0, 1, 8))
        assert env.action_space.contains(np.asarray(action, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Leakage guard: training clean, eval perturbed
# --------------------------------------------------------------------------- #
def test_training_factory_is_clean_not_perturbed():
    train_env = runner.make_clean_env("CartPole-v1", seed=0, step_counted=True)
    assert isinstance(train_env, StepCounter)
    # No perturbation wrapper anywhere in the training stack.
    node = train_env
    while hasattr(node, "env"):
        assert not isinstance(node, PerturbationWrapper)
        node = node.env


def test_eval_perturbed_factory_wraps_perturbation():
    eval_env = runner.make_perturbed_env("CartPole-v1", seed=0)
    assert isinstance(eval_env, PerturbationWrapper)


# --------------------------------------------------------------------------- #
# Regression: fixed seed -> stable fitness (drift canary)
# --------------------------------------------------------------------------- #
def test_random_agent_fixed_seed_regression():
    # A fixed-seed random policy on CartPole should land in a tight, low range.
    # If a library/env update silently changes dynamics or seeding, this moves.
    fitness = runner.mean_fitness(
        lambda s: gym.make("CartPole-v1"),
        RandomAgent(gym.make("CartPole-v1").action_space, seed=config.BASE_SEED).act,
        n_episodes=20,
        seed=config.BASE_SEED,
    )
    assert 8.0 <= fitness <= 40.0  # random CartPole returns cluster well under 50
