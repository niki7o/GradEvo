from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
from gradevo import config
logger = logging.getLogger(__name__)
Policy = Callable[[np.ndarray], np.ndarray]
_NEAT_CONFIG_TEMPLATE = '[NEAT]\nfitness_criterion       = max\nfitness_threshold       = {fitness_threshold}\nno_fitness_termination  = True\npop_size                = {pop_size}\nreset_on_extinction     = True\n\n[DefaultGenome]\nnum_inputs              = {num_inputs}\nnum_hidden              = 0\nnum_outputs             = {num_outputs}\ninitial_connection      = full_direct\nfeed_forward            = True\ncompatibility_disjoint_coefficient = 1.0\ncompatibility_weight_coefficient   = 0.5\nconn_add_prob           = 0.5\nconn_delete_prob        = 0.2\nnode_add_prob           = 0.2\nnode_delete_prob        = 0.1\nactivation_default      = tanh\nactivation_options      = tanh relu\nactivation_mutate_rate  = 0.05\naggregation_default     = sum\naggregation_options     = sum\naggregation_mutate_rate = 0.0\nbias_init_mean          = 0.0\nbias_init_stdev         = 1.0\nbias_max_value          = 30.0\nbias_min_value          = -30.0\nbias_mutate_power       = 0.5\nbias_mutate_rate        = 0.7\nbias_replace_rate       = 0.1\nenabled_default         = True\nenabled_mutate_rate     = 0.01\nresponse_init_mean      = 1.0\nresponse_init_stdev     = 0.0\nresponse_max_value      = 30.0\nresponse_min_value      = -30.0\nresponse_mutate_power   = 0.0\nresponse_mutate_rate    = 0.0\nresponse_replace_rate   = 0.0\nweight_init_mean        = 0.0\nweight_init_stdev       = 1.0\nweight_max_value        = 30.0\nweight_min_value        = -30.0\nweight_mutate_power     = 0.5\nweight_mutate_rate      = 0.8\nweight_replace_rate     = 0.1\n\n[DefaultSpeciesSet]\ncompatibility_threshold = 3.0\n\n[DefaultStagnation]\nspecies_fitness_func = max\nmax_stagnation       = 20\nspecies_elitism      = 2\n\n[DefaultReproduction]\nelitism            = 2\nsurvival_threshold = 0.2\n'

@dataclass
class NEATTrainResult:
    winner: object
    neat_config: object
    realized_steps: int
    generations: int
    curve_steps: List[int] = field(default_factory=list)
    curve_fitness: List[float] = field(default_factory=list)
    wall_clock_s: float = 0.0

def write_neat_config(path: Path, num_inputs: int, num_outputs: int, pop_size: int, fitness_threshold: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_NEAT_CONFIG_TEMPLATE.format(num_inputs=num_inputs, num_outputs=num_outputs, pop_size=pop_size, fitness_threshold=fitness_threshold))
    return path

def _load_neat_config(config_path: Path) -> object:
    import neat
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))

def make_neat_policy(genome: object, neat_config: object, continuous: bool, action_low: np.ndarray, action_high: np.ndarray) -> Policy:
    import neat
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    def policy(observation: np.ndarray) -> np.ndarray:
        outputs = np.asarray(net.activate(np.asarray(observation, dtype=np.float64)))
        if continuous:
            scaled = action_low + (outputs + 1.0) * 0.5 * (action_high - action_low)
            return np.clip(scaled, action_low, action_high).astype(np.float32)
        return int(np.argmax(outputs))
    return policy

def train_neat(make_clean_env: Callable[[int], object], seed: int, run_cfg: config.RunConfig, num_inputs: int, num_outputs: int, continuous: bool, action_low: np.ndarray, action_high: np.ndarray, max_generations: int, fitness_threshold: float) -> NEATTrainResult:
    import random
    import time
    import neat
    from gradevo.envs.step_counter import StepCounter
    random.seed(seed)
    np.random.seed(seed)
    cfg_path = write_neat_config(config.NEAT_CONFIG_DIR / f'neat_{run_cfg.env_id}.cfg', num_inputs=num_inputs, num_outputs=num_outputs, pop_size=run_cfg.neat_population_size, fitness_threshold=fitness_threshold)
    neat_config = _load_neat_config(cfg_path)
    env: StepCounter = make_clean_env(seed)
    env.reset_counter()
    curve_steps: List[int] = []
    curve_fitness: List[float] = []

    def eval_genomes(genomes: list, cfg: object) -> None:
        for (_, genome) in genomes:
            policy = make_neat_policy(genome, cfg, continuous, action_low, action_high)
            genome.fitness = _rollout_fitness(env, policy, config.NEAT_FITNESS_EPISODES, seed)
    population = neat.Population(neat_config)
    population.add_reporter(neat.StatisticsReporter())
    logger.info('NEAT training start: seed=%d budget=%d', seed, run_cfg.step_budget)
    start = time.perf_counter()
    best = None
    generations = 0
    for gen in range(max_generations):
        best = population.run(eval_genomes, 1)
        generations += 1
        curve_steps.append(int(env.step_count))
        curve_fitness.append(float(best.fitness))
        if env.step_count >= run_cfg.step_budget:
            logger.info('NEAT budget reached at generation %d', generations)
            break
    wall_clock = time.perf_counter() - start
    realized = int(env.step_count)
    env.close()
    logger.info('NEAT training done: seed=%d gens=%d realized_steps=%d wall=%.1fs', seed, generations, realized, wall_clock)
    return NEATTrainResult(winner=best, neat_config=neat_config, realized_steps=realized, generations=generations, curve_steps=curve_steps, curve_fitness=curve_fitness, wall_clock_s=wall_clock)

def _rollout_fitness(env: object, policy: Policy, n_episodes: int, seed: int) -> float:
    returns: List[float] = []
    for ep in range(n_episodes):
        (obs, _) = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < config.MAX_EPISODE_STEPS:
            action = policy(obs)
            (obs, reward, terminated, truncated, _) = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
            steps += 1
        returns.append(total)
    return float(np.mean(returns))
