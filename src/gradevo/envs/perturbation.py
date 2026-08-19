"""Domain-randomization wrapper for the held-out robustness test (H3).

This wrapper applies three perturbations, sampled fresh **per episode** where
appropriate, and is used **only at evaluation time**. Training happens on the
clean environment for both methods, so a policy evaluated through this wrapper
is being tested on a genuinely held-out distribution:

1. Gravity: scaled by a factor drawn once per episode in
   ``[1 - GRAVITY_PERTURB_FRAC, 1 + GRAVITY_PERTURB_FRAC]`` (LunarLander only;
   environments without a tunable gravity attribute skip this cleanly).
2. Observation noise: additive per-timestep Gaussian noise with per-dimension
   sigma proportional to each dimension's typical range.
3. Action noise: additive per-timestep Gaussian noise on continuous actions,
   applied before the action reaches the physics, then clipped to the action
   space. Discrete-action environments skip action noise.

All randomness flows through a single seeded ``numpy`` ``Generator`` so the
perturbations are reproducible given a seed and never touch global RNG state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

from gradevo import config

logger = logging.getLogger(__name__)


class PerturbationWrapper(gym.Wrapper):
    """Apply eval-time domain randomization to an environment.

    Attributes:
        gravity_frac: Half-width of the per-episode gravity scaling interval.
        obs_noise_frac: Observation-noise sigma as a fraction of each
            dimension's typical range.
        action_noise_frac: Action-noise sigma as a fraction of the action
            range (continuous action spaces only).
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int,
        gravity_frac: float = config.GRAVITY_PERTURB_FRAC,
        obs_noise_frac: float = config.OBS_NOISE_FRAC,
        action_noise_frac: float = config.ACTION_NOISE_FRAC,
        obs_ranges: Optional[Sequence[float]] = None,
    ) -> None:
        """Initialize the perturbation wrapper.

        Args:
            env: Environment to wrap. Must be the clean environment.
            seed: Seed for this wrapper's private RNG. Distinct from the
                training seed so perturbation draws are independent.
            gravity_frac: Per-episode gravity scaling half-width.
            obs_noise_frac: Observation noise sigma fraction.
            action_noise_frac: Action noise sigma fraction.
            obs_ranges: Per-dimension typical observation ranges used to scale
                observation noise. If ``None``, ranges are inferred: for
                LunarLander the documented ranges are used; otherwise the
                observation-space span (or unit range if unbounded) is used.
        """
        super().__init__(env)
        self.gravity_frac = gravity_frac
        self.obs_noise_frac = obs_noise_frac
        self.action_noise_frac = action_noise_frac
        self._rng = np.random.default_rng(seed)

        self._obs_sigma = self._build_obs_sigma(obs_ranges)
        self._is_continuous = isinstance(self.action_space, gym.spaces.Box)
        self._action_sigma = self._build_action_sigma()
        self._base_gravity: Optional[float] = self._read_gravity()

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _build_obs_sigma(self, obs_ranges: Optional[Sequence[float]]) -> np.ndarray:
        """Compute the per-dimension observation-noise sigma vector."""
        obs_space = self.observation_space
        n_dim = int(np.prod(obs_space.shape))
        if obs_ranges is None:
            if self.env.spec is not None and "LunarLander" in self.env.spec.id:
                ranges = np.asarray(config.LUNARLANDER_OBS_RANGES, dtype=np.float64)
            else:
                ranges = self._infer_ranges_from_space(obs_space, n_dim)
        else:
            ranges = np.asarray(obs_ranges, dtype=np.float64)
        if ranges.shape[0] != n_dim:
            raise ValueError(
                f"obs_ranges length {ranges.shape[0]} != obs dim {n_dim}"
            )
        return self.obs_noise_frac * ranges

    @staticmethod
    def _infer_ranges_from_space(
        obs_space: gym.spaces.Space, n_dim: int
    ) -> np.ndarray:
        """Infer typical per-dimension ranges from a Box observation space."""
        if isinstance(obs_space, gym.spaces.Box) and np.all(np.isfinite(obs_space.low)):
            return np.abs(obs_space.high - obs_space.low).astype(np.float64)
        return np.ones(n_dim, dtype=np.float64)

    def _build_action_sigma(self) -> Optional[np.ndarray]:
        """Compute the action-noise sigma vector for continuous action spaces."""
        if not isinstance(self.action_space, gym.spaces.Box):
            return None
        action_range = self.action_space.high - self.action_space.low
        return self.action_noise_frac * np.abs(action_range).astype(np.float64)

    def _read_gravity(self) -> Optional[float]:
        """Read the unwrapped environment's gravity attribute if present."""
        gravity = getattr(self.env.unwrapped, "gravity", None)
        if gravity is None:
            return None
        try:
            return float(gravity)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and resample per-episode gravity.

        Args:
            seed: Optional per-episode env seed, forwarded to the wrapped env.
            options: Optional reset options, forwarded unchanged.

        Returns:
            The wrapped environment's ``(obs, info)`` reset tuple, with the
            observation already noised and ``info["gravity_scale"]`` recording
            the sampled factor when gravity was perturbed.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        gravity_scale = self._resample_gravity()
        info = dict(info)
        if gravity_scale is not None:
            info["gravity_scale"] = gravity_scale
        return self._noise_obs(obs), info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Perturb the action, step, and perturb the observation.

        Args:
            action: Action selected by the policy on the clean action scale.

        Returns:
            The wrapped ``(obs, reward, terminated, truncated, info)`` tuple
            with observation noise applied. Reward is left untouched so the
            fitness metric stays comparable to the clean condition.
        """
        perturbed_action = self._noise_action(action)
        obs, reward, terminated, truncated, info = self.env.step(perturbed_action)
        return self._noise_obs(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Perturbation primitives
    # ------------------------------------------------------------------ #
    def _resample_gravity(self) -> Optional[float]:
        """Sample and apply a fresh gravity scale for the new episode."""
        if self._base_gravity is None:
            return None
        low, high = 1.0 - self.gravity_frac, 1.0 + self.gravity_frac
        scale = float(self._rng.uniform(low, high))
        try:
            self.env.unwrapped.gravity = self._base_gravity * scale
        except AttributeError:
            return None
        return scale

    def _noise_obs(self, obs: Any) -> np.ndarray:
        """Add per-dimension Gaussian noise to an observation."""
        obs_arr = np.asarray(obs, dtype=np.float64)
        noise = self._rng.normal(0.0, self._obs_sigma, size=obs_arr.shape)
        noised = obs_arr + noise
        if isinstance(self.observation_space, gym.spaces.Box):
            noised = np.clip(
                noised, self.observation_space.low, self.observation_space.high
            )
        return noised.astype(np.float32)

    def _noise_action(self, action: Any) -> Any:
        """Add Gaussian noise to a continuous action, then clip to bounds.

        Discrete actions are returned unchanged because additive Gaussian
        noise on a categorical action index is not meaningful.
        """
        if self._action_sigma is None:
            return action
        action_arr = np.asarray(action, dtype=np.float64)
        noise = self._rng.normal(0.0, self._action_sigma, size=action_arr.shape)
        noised = action_arr + noise
        noised = np.clip(noised, self.action_space.low, self.action_space.high)
        return noised.astype(np.float32)
