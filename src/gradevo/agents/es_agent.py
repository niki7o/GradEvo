from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np
from gradevo import config
logger = logging.getLogger(__name__)
Policy = Callable[[np.ndarray], np.ndarray]

def _init_mlp_params(rng: np.random.Generator, obs_dim: int, action_dim: int, hidden: Tuple[int, ...]=(64, 64)) -> List[np.ndarray]:
    sizes = [obs_dim, *hidden, action_dim]
    params: List[np.ndarray] = []
    for (fan_in, fan_out) in zip(sizes[:-1], sizes[1:]):
        scale = np.sqrt(1.0 / fan_in)
        params.append(rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float64))
        params.append(np.zeros(fan_out, dtype=np.float64))
    return params

def _flatten(params: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([p.ravel() for p in params])

def _unflatten(theta: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    idx = 0
    for shape in shapes:
        n = int(np.prod(shape))
        out.append(theta[idx:idx + n].reshape(shape))
        idx += n
    return out

def _forward(obs: np.ndarray, params: List[np.ndarray], continuous: bool, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    x = obs.astype(np.float64)
    n_layers = len(params) // 2
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for i in range(n_layers):
            (w, b) = (params[2 * i], params[2 * i + 1])
            x = x @ w + b
            if i < n_layers - 1:
                x = np.tanh(x)
    x = np.nan_to_num(x, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    if continuous:
        squashed = np.tanh(x)
        return (action_low + (squashed + 1.0) * 0.5 * (action_high - action_low)).astype(np.float32)
    return np.int64(np.argmax(x))

def make_es_policy(theta: np.ndarray, shapes: List[Tuple[int, ...]], continuous: bool, action_low: np.ndarray, action_high: np.ndarray) -> Policy:
    params = _unflatten(theta, shapes)

    def policy(observation: np.ndarray) -> np.ndarray:
        return _forward(observation, params, continuous, action_low, action_high)
    return policy

def _rank_shape(fitness: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(fitness, dtype=np.float64)
    ranks[np.argsort(fitness)] = np.arange(len(fitness), dtype=np.float64)
    ranks = ranks / (len(fitness) - 1) - 0.5
    return ranks

def _episode_fitness(env: object, policy: Policy, seed: int) -> Tuple[float, int]:
    (obs, _) = env.reset(seed=seed)
    total = 0.0
    steps = 0
    done = False
    while not done and steps < config.MAX_EPISODE_STEPS:
        action = policy(np.asarray(obs))
        (obs, reward, terminated, truncated, _) = env.step(action)
        total += float(reward)
        steps += 1
        done = bool(terminated or truncated)
    return (total, steps)

@dataclass
class ESTrainResult:
    theta: np.ndarray
    shapes: List[Tuple[int, ...]]
    realized_steps: int
    curve_steps: List[int] = field(default_factory=list)
    curve_fitness: List[float] = field(default_factory=list)
    wall_clock_s: float = 0.0
    generations: int = 0

def train_es(make_clean_env: Callable[[int], object], seed: int, run_cfg: config.RunConfig, eval_fitness_fn: Callable[[Policy, int], float], obs_dim: int, action_dim: int, continuous: bool, action_low: np.ndarray, action_high: np.ndarray, population_size: int | None=None, sigma: float | None=None, learning_rate: float | None=None, n_curve_points: int=20) -> ESTrainResult:
    population_size = population_size if population_size is not None else config.ES_POPULATION_SIZE
    sigma = sigma if sigma is not None else config.ES_SIGMA
    learning_rate = learning_rate if learning_rate is not None else config.ES_LR
    if population_size % 2 != 0:
        raise ValueError('population_size must be even for antithetic sampling')
    rng = np.random.default_rng(seed)
    params0 = _init_mlp_params(rng, obs_dim, action_dim)
    shapes = [p.shape for p in params0]
    theta = _flatten(params0)
    from gradevo.envs.step_counter import StepCounter
    train_env: StepCounter = make_clean_env(seed)
    train_env.reset_counter()
    curve_steps: List[int] = []
    curve_fitness: List[float] = []
    eval_freq = max(1, run_cfg.step_budget // n_curve_points)
    next_eval_at = eval_freq
    logger.info('ES training start: seed=%d budget=%d pop=%d sigma=%.3f lr=%.3f', seed, run_cfg.step_budget, population_size, sigma, learning_rate)
    start = time.perf_counter()
    generations = 0
    half = population_size // 2
    while train_env.step_count < run_cfg.step_budget:
        eps_half = rng.standard_normal(size=(half, theta.size))
        eps = np.concatenate([eps_half, -eps_half], axis=0)
        fitness = np.zeros(population_size, dtype=np.float64)
        for i in range(population_size):
            if train_env.step_count >= run_cfg.step_budget:
                break
            candidate = theta + sigma * eps[i]
            policy = make_es_policy(candidate, shapes, continuous, action_low, action_high)
            (ep_return, _) = _episode_fitness(train_env, policy, seed=seed + generations * population_size + i)
            fitness[i] = ep_return
        shaped = _rank_shape(fitness)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            gradient_est = eps.T @ shaped / (population_size * sigma)
        gradient_est = np.nan_to_num(gradient_est, nan=0.0, posinf=0.0, neginf=0.0)
        theta = theta + learning_rate * gradient_est
        generations += 1
        if train_env.step_count >= next_eval_at:
            policy = make_es_policy(theta, shapes, continuous, action_low, action_high)
            fit = eval_fitness_fn(policy, seed)
            curve_steps.append(int(train_env.step_count))
            curve_fitness.append(float(fit))
            next_eval_at += eval_freq
    wall_clock = time.perf_counter() - start
    realized = int(train_env.step_count)
    logger.info('ES training done: seed=%d realized_steps=%d gens=%d wall=%.1fs', seed, realized, generations, wall_clock)
    final_policy = make_es_policy(theta, shapes, continuous, action_low, action_high)
    final_fit = eval_fitness_fn(final_policy, seed)
    curve_steps.append(realized)
    curve_fitness.append(float(final_fit))
    train_env.close()
    return ESTrainResult(theta=theta, shapes=shapes, realized_steps=realized, curve_steps=curve_steps, curve_fitness=curve_fitness, wall_clock_s=wall_clock, generations=generations)
