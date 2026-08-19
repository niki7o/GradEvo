"""PPO training and evaluation via stable-baselines3 (Adam optimizer).

We use SB3's default PPO with the Adam optimizer and do not hand-tune it, to
avoid giving the gradient method an unfair advantage over NEAT (any deviation
from defaults is recorded in :data:`gradevo.config.PPO_HYPERPARAMS`). Training
runs on a step-counted **clean** environment; the realized step count is read
from the counter, not assumed from ``total_timesteps``. A callback records a
fitness-vs-steps learning curve for the sample-efficiency analysis (H2).

This module imports stable-baselines3 lazily inside functions so that the rest
of the package (and the statistics tests) can be imported without the heavy
torch/SB3 dependency present.
"""

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
    """Artifacts produced by a single PPO training run.

    Attributes:
        model: The trained SB3 PPO model (typed ``object`` to avoid a hard
            import of SB3 at module scope).
        realized_steps: Exact number of training ``env.step()`` calls, read
            from the step counter.
        curve_steps: Environment-step checkpoints at which fitness was recorded.
        curve_fitness: Mean evaluation fitness at each checkpoint.
        wall_clock_s: Wall-clock training time in seconds (secondary metric).
    """

    model: object
    realized_steps: int
    curve_steps: List[int] = field(default_factory=list)
    curve_fitness: List[float] = field(default_factory=list)
    wall_clock_s: float = 0.0


def make_ppo_policy(model: object) -> Policy:
    """Wrap a trained SB3 model as a deterministic obs->action policy.

    Args:
        model: A trained SB3 PPO model.

    Returns:
        A callable mapping an observation to a deterministic action.
    """

    def policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)  # type: ignore[attr-defined]
        return action

    return policy


def train_ppo(
    make_clean_env: Callable[[int], object],
    seed: int,
    run_cfg: config.RunConfig,
    eval_fitness_fn: Callable[[Policy, int], float],
    n_curve_points: int = 20,
) -> PPOTrainResult:
    """Train a PPO policy under the shared step budget on the clean env.

    Args:
        make_clean_env: Factory ``seed -> StepCounter(clean env)`` returning a
            step-counted clean training environment.
        seed: Deterministic seed for env, PPO init, and torch.
        run_cfg: Resolved run configuration (supplies the step budget).
        eval_fitness_fn: Callable ``(policy, seed) -> mean_fitness`` used to
            record the learning curve. It must build its own separate eval
            environment so evaluation steps do not count against the budget.
        n_curve_points: Number of fitness checkpoints across training.

    Returns:
        A :class:`PPOTrainResult` with the model, realized step count, and the
        recorded learning curve.
    """
    import time

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from gradevo.envs.step_counter import StepCounter

    train_env: StepCounter = make_clean_env(seed)  # type: ignore[assignment]
    train_env.reset_counter()

    curve_steps: List[int] = []
    curve_fitness: List[float] = []
    eval_freq = max(1, run_cfg.step_budget // n_curve_points)

    class _CurveCallback(BaseCallback):
        """Record mean eval fitness every ``eval_freq`` training steps."""

        def _on_step(self) -> bool:
            if self.num_timesteps // eval_freq > len(curve_steps):
                policy = make_ppo_policy(self.model)
                fitness = eval_fitness_fn(policy, seed)
                curve_steps.append(int(train_env.step_count))
                curve_fitness.append(float(fitness))
            return True

    model = PPO(
        config.PPO_POLICY,
        train_env,
        seed=seed,
        **config.PPO_HYPERPARAMS,
    )
    logger.info("PPO training start: seed=%d budget=%d", seed, run_cfg.step_budget)
    start = time.perf_counter()
    model.learn(total_timesteps=run_cfg.step_budget, callback=_CurveCallback())
    wall_clock = time.perf_counter() - start

    realized = int(train_env.step_count)
    logger.info(
        "PPO training done: seed=%d realized_steps=%d wall=%.1fs",
        seed,
        realized,
        wall_clock,
    )
    # Ensure the final point is recorded.
    final_fitness = eval_fitness_fn(make_ppo_policy(model), seed)
    curve_steps.append(realized)
    curve_fitness.append(float(final_fitness))
    train_env.close()

    return PPOTrainResult(
        model=model,
        realized_steps=realized,
        curve_steps=curve_steps,
        curve_fitness=curve_fitness,
        wall_clock_s=wall_clock,
    )
