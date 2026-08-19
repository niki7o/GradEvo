from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np
from gradevo import config
from gradevo.agents import es_agent
logger = logging.getLogger(__name__)
Policy = Callable[[np.ndarray], np.ndarray]

@dataclass
class CMAESTrainResult:
    theta: np.ndarray
    shapes: List[Tuple[int, ...]]
    realized_steps: int
    curve_steps: List[int] = field(default_factory=list)
    curve_fitness: List[float] = field(default_factory=list)
    wall_clock_s: float = 0.0
    generations: int = 0

def train_cmaes(make_clean_env: Callable[[int], object], seed: int, run_cfg: config.RunConfig, eval_fitness_fn: Callable[[Policy, int], float], obs_dim: int, action_dim: int, continuous: bool, action_low: np.ndarray, action_high: np.ndarray, population_size: int | None=None, sigma0: float | None=None, n_curve_points: int=20) -> CMAESTrainResult:
    import cma
    population_size = population_size if population_size is not None else config.CMAES_POPULATION_SIZE
    sigma0 = sigma0 if sigma0 is not None else config.CMAES_SIGMA0
    rng = np.random.default_rng(seed)
    params0 = es_agent._init_mlp_params(rng, obs_dim, action_dim)
    shapes = [p.shape for p in params0]
    theta0 = es_agent._flatten(params0)
    from gradevo.envs.step_counter import StepCounter
    train_env: StepCounter = make_clean_env(seed)
    train_env.reset_counter()
    curve_steps: List[int] = []
    curve_fitness: List[float] = []
    eval_freq = max(1, run_cfg.step_budget // n_curve_points)
    next_eval_at = eval_freq
    es = cma.CMAEvolutionStrategy(theta0.tolist(), sigma0, {'popsize': population_size, 'seed': seed + 1, 'verbose': -9, 'maxiter': 10 ** 9})
    logger.info('CMA-ES training start: seed=%d budget=%d pop=%d sigma0=%.3f', seed, run_cfg.step_budget, population_size, sigma0)
    start = time.perf_counter()
    generations = 0
    best_theta = theta0.copy()
    best_fitness = -np.inf
    while train_env.step_count < run_cfg.step_budget and (not es.stop()):
        candidates = es.ask()
        fitness = np.zeros(len(candidates), dtype=np.float64)
        for (i, cand) in enumerate(candidates):
            if train_env.step_count >= run_cfg.step_budget:
                break
            theta_i = np.asarray(cand, dtype=np.float64)
            policy = es_agent.make_es_policy(theta_i, shapes, continuous, action_low, action_high)
            (ep_return, _) = es_agent._episode_fitness(train_env, policy, seed=seed + generations * population_size + i)
            fitness[i] = ep_return
            if ep_return > best_fitness:
                best_fitness = ep_return
                best_theta = theta_i.copy()
        es.tell(candidates, (-fitness).tolist())
        generations += 1
        if train_env.step_count >= next_eval_at:
            mean_theta = np.asarray(es.mean, dtype=np.float64)
            policy = es_agent.make_es_policy(mean_theta, shapes, continuous, action_low, action_high)
            fit = eval_fitness_fn(policy, seed)
            curve_steps.append(int(train_env.step_count))
            curve_fitness.append(float(fit))
            next_eval_at += eval_freq
    wall_clock = time.perf_counter() - start
    realized = int(train_env.step_count)
    final_mean = np.asarray(es.mean, dtype=np.float64)
    final_policy = es_agent.make_es_policy(final_mean, shapes, continuous, action_low, action_high)
    final_fit = eval_fitness_fn(final_policy, seed)
    curve_steps.append(realized)
    curve_fitness.append(float(final_fit))
    train_env.close()
    logger.info('CMA-ES training done: seed=%d realized_steps=%d gens=%d best_train=%.1f wall=%.1fs', seed, realized, generations, best_fitness, wall_clock)
    return CMAESTrainResult(theta=final_mean, shapes=shapes, realized_steps=realized, curve_steps=curve_steps, curve_fitness=curve_fitness, wall_clock_s=wall_clock, generations=generations)
