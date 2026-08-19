"""A Gymnasium wrapper that counts ``env.step()`` calls exactly.

The compute-budget comparison hinges on measuring, not estimating, how many
environment interaction steps each method consumes. This wrapper increments a
counter on every ``step`` call and exposes it, so the runner can enforce and
report the realized budget precisely for both PPO and NEAT.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, SupportsFloat, Tuple

import gymnasium as gym

logger = logging.getLogger(__name__)


class StepCounter(gym.Wrapper):
    """Wrap an environment and count every ``step`` call.

    The count is process-local to this wrapper instance and is never reset by
    ``reset()`` -- only by the explicit :meth:`reset_counter`. This makes the
    counter a faithful record of total interaction steps across all episodes in
    a training run, which is exactly the quantity the budget is defined in.

    Attributes:
        step_count: Total number of ``step`` calls since the last
            :meth:`reset_counter` (or since construction).
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self.step_count: int = 0

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Take one environment step and increment the counter.

        Args:
            action: Action passed through to the wrapped environment.

        Returns:
            The wrapped environment's ``(obs, reward, terminated, truncated,
            info)`` tuple, with ``info["total_steps"]`` set to the running
            count for convenience.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        info = dict(info)
        info["total_steps"] = self.step_count
        return obs, reward, terminated, truncated, info

    def reset_counter(self) -> None:
        """Reset the step counter to zero.

        Use this only at the start of a training run, never between episodes,
        so the counter reflects the whole run's interaction budget.
        """
        logger.debug("StepCounter reset from %d to 0", self.step_count)
        self.step_count = 0
