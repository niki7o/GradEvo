"""Baseline controllers: a random-action agent and a hand-coded heuristic.

These provide the H0 precondition floor: a trained method must beat both. The
heuristic for LunarLander is a small PD-style controller adapted from the
classic hand-tuned lander policy; the CartPole heuristic nudges the cart in the
direction that opposes the pole's lean. Both are deterministic given a seed for
any stochastic tie-breaking, and neither learns.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class RandomAgent:
    """Samples actions uniformly from the action space.

    Attributes:
        action_space: The environment action space to sample from.
    """

    def __init__(self, action_space: gym.Space, seed: int) -> None:
        """Initialize the random agent.

        Args:
            action_space: Action space to sample from.
            seed: Seed for the action space's private RNG.
        """
        self.action_space = action_space
        self.action_space.seed(seed)

    def act(self, observation: Any) -> Any:
        """Return a uniformly random action, ignoring the observation.

        Args:
            observation: Current observation (unused).

        Returns:
            A sampled action.
        """
        return self.action_space.sample()


class HeuristicAgent:
    """A hand-coded controller for LunarLander(Continuous) and CartPole.

    The controller dispatches on the action space: a continuous PD lander
    policy for Box action spaces, and a discrete angle-opposing policy for
    CartPole-style Discrete spaces.

    Attributes:
        continuous: Whether the target environment has a continuous action
            space.
    """

    def __init__(self, action_space: gym.Space) -> None:
        """Initialize the heuristic agent.

        Args:
            action_space: Action space, used to pick the control law and to
                clip continuous outputs to valid bounds.
        """
        self.continuous = isinstance(action_space, gym.spaces.Box)
        self._action_space = action_space

    def act(self, observation: Any) -> Any:
        """Compute a control action for the given observation.

        Args:
            observation: Environment observation.

        Returns:
            An action appropriate to the action space.
        """
        obs = np.asarray(observation, dtype=np.float64)
        if self.continuous:
            return self._lunar_lander_control(obs)
        return self._cartpole_control(obs)

    def _lunar_lander_control(self, obs: np.ndarray) -> np.ndarray:
        """PD-style lander control returning a 2-D continuous action.

        The observation layout is ``(x, y, vx, vy, angle, angular_vel,
        leg1, leg2)``. Targets bring the lander upright and descending gently
        toward the pad, mirroring the classic hand-tuned heuristic.

        Args:
            obs: 8-dimensional LunarLander observation.

        Returns:
            A 2-D action ``[main_engine, lateral_engine]`` clipped to bounds.
        """
        angle_target = obs[0] * 0.5 + obs[2] * 1.0  # aim toward pad, damp vx
        angle_target = float(np.clip(angle_target, -0.4, 0.4))
        hover_target = 0.55 * np.abs(obs[0])

        angle_error = (angle_target - obs[4]) * 0.5 - obs[5] * 1.0
        hover_error = (hover_target - obs[1]) * 0.5 - obs[3] * 0.5

        main = float(np.clip(hover_error * 20.0 - 1.0, -1.0, 1.0))
        lateral = float(np.clip(-angle_error * 20.0, -1.0, 1.0))
        if obs[6] > 0.0 or obs[7] > 0.0:  # a leg is in contact: cut thrust
            main, lateral = -1.0, 0.0
        action = np.array([main, lateral], dtype=np.float32)
        return np.clip(action, self._action_space.low, self._action_space.high)

    @staticmethod
    def _cartpole_control(obs: np.ndarray) -> int:
        """Discrete CartPole control: push toward the pole's lean.

        Args:
            obs: 4-dimensional CartPole observation
                ``(x, x_dot, theta, theta_dot)``.

        Returns:
            ``0`` to push left, ``1`` to push right.
        """
        # Combine pole angle and angular velocity; push right if leaning right.
        signal = obs[2] + 0.5 * obs[3]
        return 1 if signal > 0 else 0
