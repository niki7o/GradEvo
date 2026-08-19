from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Sequence, SupportsFloat, Tuple
import gymnasium as gym
import numpy as np
from gradevo import config
logger = logging.getLogger(__name__)

class PerturbationWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, seed: int, gravity_frac: float=config.GRAVITY_PERTURB_FRAC, obs_noise_frac: float=config.OBS_NOISE_FRAC, action_noise_frac: float=config.ACTION_NOISE_FRAC, obs_ranges: Optional[Sequence[float]]=None) -> None:
        super().__init__(env)
        self.gravity_frac = gravity_frac
        self.obs_noise_frac = obs_noise_frac
        self.action_noise_frac = action_noise_frac
        self._rng = np.random.default_rng(seed)
        self._obs_sigma = self._build_obs_sigma(obs_ranges)
        self._is_continuous = isinstance(self.action_space, gym.spaces.Box)
        self._action_sigma = self._build_action_sigma()
        self._base_gravity: Optional[float] = self._read_gravity()

    def _build_obs_sigma(self, obs_ranges: Optional[Sequence[float]]) -> np.ndarray:
        obs_space = self.observation_space
        n_dim = int(np.prod(obs_space.shape))
        if obs_ranges is None:
            if self.env.spec is not None and 'LunarLander' in self.env.spec.id:
                ranges = np.asarray(config.LUNARLANDER_OBS_RANGES, dtype=np.float64)
            else:
                ranges = self._infer_ranges_from_space(obs_space, n_dim)
        else:
            ranges = np.asarray(obs_ranges, dtype=np.float64)
        if ranges.shape[0] != n_dim:
            raise ValueError(f'obs_ranges length {ranges.shape[0]} != obs dim {n_dim}')
        return self.obs_noise_frac * ranges

    @staticmethod
    def _infer_ranges_from_space(obs_space: gym.spaces.Space, n_dim: int) -> np.ndarray:
        if isinstance(obs_space, gym.spaces.Box) and np.all(np.isfinite(obs_space.low)):
            return np.abs(obs_space.high - obs_space.low).astype(np.float64)
        return np.ones(n_dim, dtype=np.float64)

    def _build_action_sigma(self) -> Optional[np.ndarray]:
        if not isinstance(self.action_space, gym.spaces.Box):
            return None
        action_range = self.action_space.high - self.action_space.low
        return self.action_noise_frac * np.abs(action_range).astype(np.float64)

    def _read_gravity(self) -> Optional[float]:
        gravity = getattr(self.env.unwrapped, 'gravity', None)
        if gravity is None:
            return None
        try:
            return float(gravity)
        except (TypeError, ValueError):
            return None

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict[str, Any]]=None) -> Tuple[Any, Dict[str, Any]]:
        (obs, info) = self.env.reset(seed=seed, options=options)
        gravity_scale = self._resample_gravity()
        info = dict(info)
        if gravity_scale is not None:
            info['gravity_scale'] = gravity_scale
        return (self._noise_obs(obs), info)

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        perturbed_action = self._noise_action(action)
        (obs, reward, terminated, truncated, info) = self.env.step(perturbed_action)
        return (self._noise_obs(obs), reward, terminated, truncated, info)

    def _resample_gravity(self) -> Optional[float]:
        if self._base_gravity is None:
            return None
        (low, high) = (1.0 - self.gravity_frac, 1.0 + self.gravity_frac)
        scale = float(self._rng.uniform(low, high))
        try:
            self.env.unwrapped.gravity = self._base_gravity * scale
        except AttributeError:
            return None
        return scale

    def _noise_obs(self, obs: Any) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float64)
        noise = self._rng.normal(0.0, self._obs_sigma, size=obs_arr.shape)
        noised = obs_arr + noise
        if isinstance(self.observation_space, gym.spaces.Box):
            noised = np.clip(noised, self.observation_space.low, self.observation_space.high)
        return noised.astype(np.float32)

    def _noise_action(self, action: Any) -> Any:
        if self._action_sigma is None:
            return action
        action_arr = np.asarray(action, dtype=np.float64)
        noise = self._rng.normal(0.0, self._action_sigma, size=action_arr.shape)
        noised = action_arr + noise
        noised = np.clip(noised, self.action_space.low, self.action_space.high)
        return noised.astype(np.float32)
