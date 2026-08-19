from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np
from gradevo import config
logger = logging.getLogger(__name__)
Policy = Callable[[np.ndarray], np.ndarray]

@dataclass
class PPOTrainResult:
    model: object
    realized_steps: int
    curve_steps: List[int] = field(default_factory=list)
    curve_fitness: List[float] = field(default_factory=list)
    wall_clock_s: float = 0.0

def make_ppo_policy(model: object) -> Policy:

    def policy(observation: np.ndarray) -> np.ndarray:
        (action, _) = model.predict(observation, deterministic=True)
        return action
    return policy

def train_ppo(make_clean_env: Callable[[int], object], seed: int, run_cfg: config.RunConfig, eval_fitness_fn: Callable[[Policy, int], float], n_curve_points: int=20) -> PPOTrainResult:
    import time
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from gradevo.envs.step_counter import StepCounter
    train_env: StepCounter = make_clean_env(seed)
    train_env.reset_counter()
    curve_steps: List[int] = []
    curve_fitness: List[float] = []
    eval_freq = max(1, run_cfg.step_budget // n_curve_points)

    class _CurveCallback(BaseCallback):

        def _on_step(self) -> bool:
            if self.num_timesteps // eval_freq > len(curve_steps):
                policy = make_ppo_policy(self.model)
                fitness = eval_fitness_fn(policy, seed)
                curve_steps.append(int(train_env.step_count))
                curve_fitness.append(float(fitness))
            return True
    model = PPO(config.PPO_POLICY, train_env, seed=seed, **config.PPO_HYPERPARAMS)
    logger.info('PPO training start: seed=%d budget=%d', seed, run_cfg.step_budget)
    start = time.perf_counter()
    model.learn(total_timesteps=run_cfg.step_budget, callback=_CurveCallback())
    wall_clock = time.perf_counter() - start
    realized = int(train_env.step_count)
    logger.info('PPO training done: seed=%d realized_steps=%d wall=%.1fs', seed, realized, wall_clock)
    final_fitness = eval_fitness_fn(make_ppo_policy(model), seed)
    curve_steps.append(realized)
    curve_fitness.append(float(final_fitness))
    train_env.close()
    return PPOTrainResult(model=model, realized_steps=realized, curve_steps=curve_steps, curve_fitness=curve_fitness, wall_clock_s=wall_clock)
