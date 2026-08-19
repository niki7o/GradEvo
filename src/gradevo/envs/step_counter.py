from __future__ import annotations
import logging
from typing import Any, Dict, SupportsFloat, Tuple
import gymnasium as gym
logger = logging.getLogger(__name__)

class StepCounter(gym.Wrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.step_count: int = 0

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        (obs, reward, terminated, truncated, info) = self.env.step(action)
        self.step_count += 1
        info = dict(info)
        info['total_steps'] = self.step_count
        return (obs, reward, terminated, truncated, info)

    def reset_counter(self) -> None:
        logger.debug('StepCounter reset from %d to 0', self.step_count)
        self.step_count = 0
