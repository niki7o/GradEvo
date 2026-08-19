"""NEAT training and evaluation via neat-python.

NEAT evolves both weights and topology. Fitness is the mean episodic return of
a genome's network on the **clean** environment. Every environment step across
every genome and generation is counted by a shared step-counting wrapper, and
evolution stops as soon as the realized step count reaches the shared budget --
this is what makes the PPO/NEAT comparison budget-matched on interaction steps
rather than on generations (which are structurally incomparable to PPO epochs).

neat-python is imported lazily inside functions so the package imports without
it present. The NEAT configuration file is generated from a template so the
input/output counts and population size come from :mod:`gradevo.config` rather
than a hand-edited file with hidden magic numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from gradevo import config

logger = logging.getLogger(__name__)

Policy = Callable[[np.ndarray], np.ndarray]

# NEAT config template. Only structural/capacity fields are parameterized; the
# rest are conservative, widely-used defaults for control tasks. Kept here (not
# a committed static file) so counts stay consistent with config.py.
_NEAT_CONFIG_TEMPLATE = """\
[NEAT]
fitness_criterion       = max
fitness_threshold       = {fitness_threshold}
no_fitness_termination  = True
pop_size                = {pop_size}
reset_on_extinction     = True

[DefaultGenome]
num_inputs              = {num_inputs}
num_hidden              = 0
num_outputs             = {num_outputs}
initial_connection      = full_direct
feed_forward            = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.2
node_add_prob           = 0.2
node_delete_prob        = 0.1
activation_default      = tanh
activation_options      = tanh relu
activation_mutate_rate  = 0.05
aggregation_default     = sum
aggregation_options     = sum
aggregation_mutate_rate = 0.0
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


@dataclass
class NEATTrainResult:
    """Artifacts produced by a single NEAT training run.

    Attributes:
        winner: The best genome found (neat-python ``DefaultGenome``).
        neat_config: The neat-python ``Config`` object used (needed to rebuild
            the network from ``winner`` at evaluation time).
        realized_steps: Exact number of training ``env.step()`` calls consumed.
        generations: Number of generations actually run.
        curve_steps: Cumulative env-step checkpoints (one per generation).
        curve_fitness: Best-genome fitness at each generation checkpoint.
        wall_clock_s: Wall-clock training time in seconds (secondary metric).
    """

    winner: object
    neat_config: object
    realized_steps: int
    generations: int
    curve_steps: List[int] = field(default_factory=list)
    curve_fitness: List[float] = field(default_factory=list)
    wall_clock_s: float = 0.0


def write_neat_config(
    path: Path,
    num_inputs: int,
    num_outputs: int,
    pop_size: int,
    fitness_threshold: float,
) -> Path:
    """Render the NEAT config template to ``path`` and return it.

    Args:
        path: Destination file path.
        num_inputs: Observation dimensionality (network inputs).
        num_outputs: Action dimensionality (network outputs).
        pop_size: Population size for this run.
        fitness_threshold: Early-stop fitness; set high so the step budget, not
            fitness, terminates training.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _NEAT_CONFIG_TEMPLATE.format(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            pop_size=pop_size,
            fitness_threshold=fitness_threshold,
        )
    )
    return path


def _load_neat_config(config_path: Path) -> object:
    """Load a neat-python Config object from a file path."""
    import neat

    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )


def make_neat_policy(
    genome: object, neat_config: object, continuous: bool, action_low: np.ndarray, action_high: np.ndarray
) -> Policy:
    """Build an obs->action policy from a genome.

    Args:
        genome: A neat-python genome.
        neat_config: The neat-python Config the genome was evolved under.
        continuous: Whether the action space is continuous.
        action_low: Lower action bounds (continuous only).
        action_high: Upper action bounds (continuous only).

    Returns:
        A callable mapping an observation to an action. For continuous spaces
        the tanh network outputs are affinely mapped from ``[-1, 1]`` to the
        action bounds; for discrete spaces the argmax output index is used.
    """
    import neat

    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    def policy(observation: np.ndarray) -> np.ndarray:
        outputs = np.asarray(net.activate(np.asarray(observation, dtype=np.float64)))
        if continuous:
            # tanh outputs in [-1, 1] -> [low, high]
            scaled = action_low + (outputs + 1.0) * 0.5 * (action_high - action_low)
            return np.clip(scaled, action_low, action_high).astype(np.float32)
        return int(np.argmax(outputs))

    return policy


def train_neat(
    make_clean_env: Callable[[int], object],
    seed: int,
    run_cfg: config.RunConfig,
    num_inputs: int,
    num_outputs: int,
    continuous: bool,
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_generations: int,
    fitness_threshold: float,
) -> NEATTrainResult:
    """Evolve a NEAT population under the shared step budget on the clean env.

    Evolution runs one generation at a time; after each generation the realized
    step count is checked against the budget and evolution stops once the budget
    is met. All environment steps are consumed on the clean environment only.

    Args:
        make_clean_env: Factory ``seed -> StepCounter(clean env)``.
        seed: Deterministic seed for NEAT population init and env resets.
        run_cfg: Resolved run configuration (supplies the step budget).
        num_inputs: Network input count (observation dim).
        num_outputs: Network output count (action dim, or #actions for
            discrete).
        continuous: Whether actions are continuous.
        action_low: Lower action bounds (continuous only).
        action_high: Upper action bounds (continuous only).
        max_generations: Hard cap on generations (a safety bound; the step
            budget is the primary terminator).
        fitness_threshold: Early-stop fitness threshold written to the config.

    Returns:
        A :class:`NEATTrainResult` with the winner genome, config, realized step
        count, and per-generation learning curve.
    """
    import random
    import time

    import neat

    from gradevo.envs.step_counter import StepCounter

    random.seed(seed)
    np.random.seed(seed)

    cfg_path = write_neat_config(
        config.NEAT_CONFIG_DIR / f"neat_{run_cfg.env_id}.cfg",
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        pop_size=run_cfg.neat_population_size,
        fitness_threshold=fitness_threshold,
    )
    neat_config = _load_neat_config(cfg_path)

    env: StepCounter = make_clean_env(seed)  # type: ignore[assignment]
    env.reset_counter()

    curve_steps: List[int] = []
    curve_fitness: List[float] = []

    def eval_genomes(genomes: list, cfg: object) -> None:
        """Assign each genome its mean episodic return on the clean env."""
        for _, genome in genomes:
            policy = make_neat_policy(genome, cfg, continuous, action_low, action_high)
            genome.fitness = _rollout_fitness(env, policy, config.NEAT_FITNESS_EPISODES, seed)

    population = neat.Population(neat_config)
    population.add_reporter(neat.StatisticsReporter())

    logger.info("NEAT training start: seed=%d budget=%d", seed, run_cfg.step_budget)
    start = time.perf_counter()
    best = None
    generations = 0
    for gen in range(max_generations):
        best = population.run(eval_genomes, 1)
        generations += 1
        curve_steps.append(int(env.step_count))
        curve_fitness.append(float(best.fitness))
        if env.step_count >= run_cfg.step_budget:
            logger.info("NEAT budget reached at generation %d", generations)
            break
    wall_clock = time.perf_counter() - start
    realized = int(env.step_count)
    env.close()
    logger.info(
        "NEAT training done: seed=%d gens=%d realized_steps=%d wall=%.1fs",
        seed,
        generations,
        realized,
        wall_clock,
    )

    return NEATTrainResult(
        winner=best,
        neat_config=neat_config,
        realized_steps=realized,
        generations=generations,
        curve_steps=curve_steps,
        curve_fitness=curve_fitness,
        wall_clock_s=wall_clock,
    )


def _rollout_fitness(
    env: object, policy: Policy, n_episodes: int, seed: int
) -> float:
    """Run ``n_episodes`` and return the mean episodic return.

    Args:
        env: A (step-counted) environment.
        policy: obs->action callable.
        n_episodes: Number of episodes to average.
        seed: Base seed; episode seeds are derived deterministically.

    Returns:
        Mean total reward across episodes.
    """
    returns: List[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)  # type: ignore[attr-defined]
        done = False
        total = 0.0
        steps = 0
        while not done and steps < config.MAX_EPISODE_STEPS:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)  # type: ignore[attr-defined]
            total += float(reward)
            done = bool(terminated or truncated)
            steps += 1
        returns.append(total)
    return float(np.mean(returns))
