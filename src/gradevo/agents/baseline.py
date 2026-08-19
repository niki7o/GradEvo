from __future__ import annotations
import logging
from typing import Any
import gymnasium as gym
import numpy as np
logger = logging.getLogger(__name__)

class RandomAgent:

    def __init__(self, action_space: gym.Space, seed: int) -> None:
        self.action_space = action_space
        self.action_space.seed(seed)

    def act(self, observation: Any) -> Any:
        return self.action_space.sample()

class HeuristicAgent:

    def __init__(self, action_space: gym.Space, env_id: str | None=None) -> None:
        self.continuous = isinstance(action_space, gym.spaces.Box)
        self._action_space = action_space
        self._env_id = env_id or ''

    def act(self, observation: Any) -> Any:
        obs = np.asarray(observation, dtype=np.float64)
        if not self.continuous:
            return self._cartpole_control(obs)
        if 'Pendulum' in self._env_id or obs.shape == (3,):
            return self._pendulum_control(obs)
        return self._lunar_lander_control(obs)

    def _pendulum_control(self, obs: np.ndarray) -> np.ndarray:
        (cos_theta, sin_theta, theta_dot) = (obs[0], obs[1], obs[2])
        theta = float(np.arctan2(sin_theta, cos_theta))
        torque = float(np.clip(-8.0 * theta - 1.5 * theta_dot, -2.0, 2.0))
        action = np.array([torque], dtype=np.float32)
        return np.clip(action, self._action_space.low, self._action_space.high)

    def _lunar_lander_control(self, obs: np.ndarray) -> np.ndarray:
        angle_target = obs[0] * 0.5 + obs[2] * 1.0
        angle_target = float(np.clip(angle_target, -0.4, 0.4))
        hover_target = 0.55 * np.abs(obs[0])
        angle_error = (angle_target - obs[4]) * 0.5 - obs[5] * 1.0
        hover_error = (hover_target - obs[1]) * 0.5 - obs[3] * 0.5
        main = float(np.clip(hover_error * 20.0 - 1.0, -1.0, 1.0))
        lateral = float(np.clip(-angle_error * 20.0, -1.0, 1.0))
        if obs[6] > 0.0 or obs[7] > 0.0:
            (main, lateral) = (-1.0, 0.0)
        action = np.array([main, lateral], dtype=np.float32)
        return np.clip(action, self._action_space.low, self._action_space.high)

    @staticmethod
    def _cartpole_control(obs: np.ndarray) -> int:
        signal = obs[2] + 0.5 * obs[3]
        return 1 if signal > 0 else 0
