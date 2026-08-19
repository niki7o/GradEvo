from __future__ import annotations
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
import pandas as pd
from gradevo import config
from gradevo.agents.baseline import HeuristicAgent, RandomAgent
from gradevo.envs.perturbation import PerturbationWrapper
from gradevo.envs.step_counter import StepCounter
from gradevo.experiment import budget as budget_mod
logger = logging.getLogger(__name__)
Policy = Callable[[np.ndarray], np.ndarray]

def make_clean_env(env_id: str, seed: int, step_counted: bool=False) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    return StepCounter(env) if step_counted else env

def make_perturbed_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    return PerturbationWrapper(env, seed=seed)

def evaluate_policy(make_env: Callable[[int], gym.Env], policy: Policy, n_episodes: int, seed: int) -> List[float]:
    env = make_env(seed)
    returns: List[float] = []
    for ep in range(n_episodes):
        (obs, _) = env.reset(seed=seed + ep)
        (done, total, steps) = (False, 0.0, 0)
        while not done and steps < config.MAX_EPISODE_STEPS:
            (obs, reward, terminated, truncated, _) = env.step(policy(obs))
            total += float(reward)
            done = bool(terminated or truncated)
            steps += 1
        returns.append(total)
    env.close()
    return returns

def mean_fitness(make_env: Callable[[int], gym.Env], policy: Policy, n_episodes: int, seed: int) -> float:
    return float(np.mean(evaluate_policy(make_env, policy, n_episodes, seed)))

@dataclass
class SeedRun:
    method: str
    seed: int
    clean_fitness: float
    perturbed_fitness: float
    realized_steps: int
    wall_clock_s: float
    curve_steps: List[int] = field(default_factory=list)
    curve_fitness: List[float] = field(default_factory=list)

def _env_dims(env_id: str) -> Tuple[int, int, bool, np.ndarray, np.ndarray]:
    probe = gym.make(env_id)
    obs_dim = int(np.prod(probe.observation_space.shape))
    if isinstance(probe.action_space, gym.spaces.Box):
        continuous = True
        action_dim = int(np.prod(probe.action_space.shape))
        low = np.asarray(probe.action_space.low, dtype=np.float64)
        high = np.asarray(probe.action_space.high, dtype=np.float64)
    else:
        continuous = False
        action_dim = int(probe.action_space.n)
        low = np.zeros(action_dim)
        high = np.ones(action_dim)
    probe.close()
    return (obs_dim, action_dim, continuous, low, high)

def train_ppo_seed(env_id: str, seed: int, run_cfg: config.RunConfig) -> SeedRun:
    from gradevo.agents import ppo_agent
    clean_eval = lambda s: make_clean_env(env_id, s)
    result = ppo_agent.train_ppo(make_clean_env=lambda s: make_clean_env(env_id, s, step_counted=True), seed=seed, run_cfg=run_cfg, eval_fitness_fn=lambda pol, s: mean_fitness(clean_eval, pol, run_cfg.eval_episodes, s))
    policy = ppo_agent.make_ppo_policy(result.model)
    seed_run = _evaluate_both_conditions(env_id, 'ppo', seed, run_cfg, policy, result.realized_steps, result.wall_clock_s, result.curve_steps, result.curve_fitness)
    _save_ppo_model(result.model, env_id, seed)
    return seed_run

def train_neat_seed(env_id: str, seed: int, run_cfg: config.RunConfig) -> SeedRun:
    from gradevo.agents import neat_agent
    (obs_dim, action_dim, continuous, low, high) = _env_dims(env_id)
    spec = config.ENV_SPECS[env_id]
    max_gens = budget_mod.plan_neat_generations(run_cfg.step_budget, run_cfg.neat_population_size)
    threshold = (spec.solved_threshold or 1000000000.0) + 1000000.0
    result = neat_agent.train_neat(make_clean_env=lambda s: make_clean_env(env_id, s, step_counted=True), seed=seed, run_cfg=run_cfg, num_inputs=obs_dim, num_outputs=action_dim, continuous=continuous, action_low=low, action_high=high, max_generations=max_gens, fitness_threshold=threshold)
    policy = neat_agent.make_neat_policy(result.winner, result.neat_config, continuous, low, high)
    seed_run = _evaluate_both_conditions(env_id, 'neat', seed, run_cfg, policy, result.realized_steps, result.wall_clock_s, result.curve_steps, result.curve_fitness)
    _save_neat_genome(result.winner, env_id, seed)
    return seed_run

def _evaluate_both_conditions(env_id: str, method: str, seed: int, run_cfg: config.RunConfig, policy: Policy, realized_steps: int, wall_clock_s: float, curve_steps: List[int], curve_fitness: List[float]) -> SeedRun:
    clean = mean_fitness(lambda s: make_clean_env(env_id, s), policy, run_cfg.eval_episodes, seed)
    perturbed = mean_fitness(lambda s: make_perturbed_env(env_id, s), policy, run_cfg.eval_episodes, seed + 10000)
    logger.info('%s seed=%d clean=%.1f perturbed=%.1f', method, seed, clean, perturbed)
    return SeedRun(method=method, seed=seed, clean_fitness=clean, perturbed_fitness=perturbed, realized_steps=realized_steps, wall_clock_s=wall_clock_s, curve_steps=list(curve_steps), curve_fitness=list(curve_fitness))

def evaluate_baselines(env_id: str, run_cfg: config.RunConfig) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {'random': [], 'heuristic': []}
    probe = gym.make(env_id)
    action_space = probe.action_space
    probe.close()
    for seed in run_cfg.seeds:
        random_agent = RandomAgent(gym.make(env_id).action_space, seed)
        heuristic_agent = HeuristicAgent(action_space)
        results['random'].append(mean_fitness(lambda s: make_clean_env(env_id, s), random_agent.act, run_cfg.eval_episodes, seed))
        results['heuristic'].append(mean_fitness(lambda s: make_clean_env(env_id, s), heuristic_agent.act, run_cfg.eval_episodes, seed))
    return results

def _save_ppo_model(model: object, env_id: str, seed: int) -> None:
    config.PPO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PPO_MODELS_DIR / f'ppo_{env_id}_seed{seed}.zip'
    model.save(str(path))

def _save_neat_genome(genome: object, env_id: str, seed: int) -> None:
    config.NEAT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.NEAT_MODELS_DIR / f'neat_{env_id}_seed{seed}.pkl'
    with path.open('wb') as fh:
        pickle.dump(genome, fh)

def save_results(env_id: str, run_cfg: config.RunConfig, seed_runs: List[SeedRun], baselines: Dict[str, List[float]]) -> Dict[str, Path]:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    slug = env_id.replace('-', '_')
    paths: Dict[str, Path] = {}
    fitness_df = _fitness_dataframe(env_id, seed_runs, baselines, run_cfg)
    paths['fitness'] = config.DATA_DIR / f'fitness_{slug}.csv'
    fitness_df.to_csv(paths['fitness'], index=False)
    curves_df = _curves_dataframe(env_id, seed_runs)
    paths['curves'] = config.DATA_DIR / f'curves_{slug}.csv'
    curves_df.to_csv(paths['curves'], index=False)
    budget_df = _budget_dataframe(run_cfg, seed_runs)
    paths['budget'] = config.DATA_DIR / f'budget_{slug}.csv'
    budget_df.to_csv(paths['budget'], index=False)
    paths['metadata'] = config.DATA_DIR / f'metadata_{slug}.json'
    paths['metadata'].write_text(json.dumps(_metadata(env_id, run_cfg), indent=2))
    logger.info('Saved results for %s to %s', env_id, config.DATA_DIR)
    return paths

def _fitness_dataframe(env_id: str, seed_runs: List[SeedRun], baselines: Dict[str, List[float]], run_cfg: config.RunConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in seed_runs:
        rows.append({'env_id': env_id, 'method': run.method, 'condition': 'clean', 'seed': run.seed, 'fitness': run.clean_fitness})
        rows.append({'env_id': env_id, 'method': run.method, 'condition': 'perturbed', 'seed': run.seed, 'fitness': run.perturbed_fitness})
    for (name, values) in baselines.items():
        for (seed, val) in zip(run_cfg.seeds, values):
            rows.append({'env_id': env_id, 'method': name, 'condition': 'clean', 'seed': seed, 'fitness': val})
    return pd.DataFrame(rows)

def _curves_dataframe(env_id: str, seed_runs: List[SeedRun]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in seed_runs:
        for (step, fit) in zip(run.curve_steps, run.curve_fitness):
            rows.append({'env_id': env_id, 'method': run.method, 'seed': run.seed, 'step': step, 'fitness': fit})
    return pd.DataFrame(rows)

def _budget_dataframe(run_cfg: config.RunConfig, seed_runs: List[SeedRun]) -> pd.DataFrame:
    reports = []
    for method in config.METHODS:
        runs = [r for r in seed_runs if r.method == method]
        if not runs:
            continue
        reports.append(budget_mod.summarize_budget(method, run_cfg.step_budget, [r.realized_steps for r in runs], [r.wall_clock_s for r in runs]))
    return budget_mod.budget_table(reports)

def _metadata(env_id: str, run_cfg: config.RunConfig) -> Dict[str, object]:
    return {'env_id': env_id, 'quick': run_cfg.quick, 'n_seeds': run_cfg.n_seeds, 'seeds': run_cfg.seeds, 'step_budget': run_cfg.step_budget, 'neat_population_size': run_cfg.neat_population_size, 'eval_episodes': run_cfg.eval_episodes, 'alpha_corrected': config.ALPHA_CORRECTED, 'gradevo_version': __import__('gradevo').__version__}
