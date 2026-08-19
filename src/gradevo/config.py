"""Central configuration for GradEvo.

Every constant that influences an experiment lives here so that logic modules
contain no magic numbers and every stochastic process can be seeded explicitly.
Importing this module has no side effects other than defining constants and a
handful of small, frozen dataclasses.

The values below are the *defaults*. The experiment runner and the ``--quick``
CLI flag override the seed count and budget for fast verification; whenever an
override is used the realized values are logged and reported so that the
notebook never presents a reduced run as if it were the full one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_ROOT.parent.parent
DATA_DIR: Path = REPO_ROOT / "data" / "processed"
MODELS_DIR: Path = REPO_ROOT / "models"
PPO_MODELS_DIR: Path = MODELS_DIR / "ppo"
NEAT_MODELS_DIR: Path = MODELS_DIR / "neat"
NEAT_CONFIG_DIR: Path = PACKAGE_ROOT / "agents"
MLFLOW_DB_PATH: Path = REPO_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_EXPERIMENT_NAME: str = "gradevo"

# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
PRIMARY_ENV_ID: str = "LunarLanderContinuous-v3"
SECONDARY_ENV_ID: str = "CartPole-v1"

# Per-episode step cap used when we need a bounded rollout (NEAT fitness eval,
# step accounting). Gymnasium's own TimeLimit still applies; this is an
# additional guard so the step budget is predictable.
MAX_EPISODE_STEPS: int = 1000

# --------------------------------------------------------------------------- #
# Compute-budget matching (the methodological crux -- see notebook section 4)
# --------------------------------------------------------------------------- #
# The shared budget is defined as total env.step() calls across a whole
# training run. Both methods are held to the same target; the *realized* count
# is measured exactly by a step-counting wrapper and reported per run.
TOTAL_STEP_BUDGET: int = 500_000
QUICK_STEP_BUDGET: int = 20_000

# NEAT budget bookkeeping. generations is derived from the step budget and the
# expected per-generation step cost (population_size * fitness_episodes *
# expected_episode_length); see experiment.budget for the exact arithmetic.
NEAT_POPULATION_SIZE: int = 60
NEAT_FITNESS_EPISODES: int = 1  # episodes averaged per genome per generation
QUICK_NEAT_POPULATION_SIZE: int = 20

# Episode-length estimates used ONLY to plan the NEAT generation *cap*. The true
# step count is always measured, never assumed. LunarLander episodes vary a lot
# (early crashes are short; competent hovering is long). The generation cap is a
# safety bound only: it is deliberately derived from a *small* episode-length
# estimate so the cap is generous and the measured step budget -- not the cap --
# terminates evolution. If episodes turn out long, the budget is simply reached
# in fewer generations and the loop breaks early.
EXPECTED_EPISODE_LENGTH: int = 250   # rough central estimate (documentation)
NEAT_PLANNING_MIN_EPISODE_LENGTH: int = 40  # conservative floor for the gen cap

# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
# N seeds per method per condition. Target 20; 15 is the documented fallback.
N_SEEDS: int = 20
QUICK_N_SEEDS: int = 3
MIN_ACCEPTABLE_SEEDS: int = 15

# Base seed; per-run seeds are BASE_SEED + index so runs are fully reproducible
# and independent across methods/conditions.
BASE_SEED: int = 1234

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
# Episodes used to estimate a trained policy's fitness at evaluation time.
EVAL_EPISODES: int = 20
QUICK_EVAL_EPISODES: int = 5

# --------------------------------------------------------------------------- #
# Perturbation protocol for H3 (domain randomization, eval-time ONLY)
# --------------------------------------------------------------------------- #
GRAVITY_PERTURB_FRAC: float = 0.10  # +/-10% per episode
OBS_NOISE_FRAC: float = 0.02        # sigma = 2% of each dim's typical range
ACTION_NOISE_FRAC: float = 0.05     # sigma = 5% of action range

# Typical per-dimension observation ranges for LunarLander, used to scale the
# observation-noise sigma. These are the documented physical ranges of the
# 8-dim observation (x, y, vx, vy, angle, angular_vel, leg1_contact,
# leg2_contact). The contact flags are boolean {0,1}; we do not inject noise
# into them (range set to 0.0) because additive Gaussian noise on a boolean
# sensor is not a meaningful perturbation.
LUNARLANDER_OBS_RANGES: Tuple[float, ...] = (
    2.0,   # x position
    2.0,   # y position
    5.0,   # x velocity
    5.0,   # y velocity
    6.28,  # angle (radians)
    5.0,   # angular velocity
    0.0,   # left leg contact (boolean -- no noise)
    0.0,   # right leg contact (boolean -- no noise)
)

# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
ALPHA: float = 0.05
# Family-wise Bonferroni correction shared by H1, H2, H3 (three tests).
N_FAMILY_TESTS: int = 3
ALPHA_CORRECTED: float = ALPHA / N_FAMILY_TESTS  # 0.0167
# H0 is a precondition gate, reported separately, also corrected across its
# own comparisons (each trained method vs. two baselines -> not part of the
# H1-H3 family).
SHAPIRO_ALPHA: float = 0.05

# --------------------------------------------------------------------------- #
# PPO hyperparameters (stable-baselines3 defaults unless noted)
# --------------------------------------------------------------------------- #
# We keep SB3 PPO defaults (Adam optimizer) to avoid an unfair, hand-tuned
# advantage over NEAT. Any deviation from defaults must be recorded here with a
# justification comment so the notebook can cite it.
PPO_POLICY: str = "MlpPolicy"
PPO_HYPERPARAMS: Dict[str, object] = {
    # Two hidden layers of 64 units -- SB3 default MLP. Chosen to keep the
    # gradient policy's capacity comparable to the topologies NEAT typically
    # discovers on this task, rather than to maximize PPO in isolation.
    "policy_kwargs": {"net_arch": [64, 64]},
    "verbose": 0,
}

# --------------------------------------------------------------------------- #
# Small typed containers used across modules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnvSpec:
    """Description of a task environment.

    Attributes:
        env_id: Gymnasium registration id.
        continuous: Whether the action space is continuous (Box) or discrete.
        solved_threshold: Published "solved" return threshold, for reference
            lines in plots and prose. ``None`` if no canonical threshold.
    """

    env_id: str
    continuous: bool
    solved_threshold: float | None


ENV_SPECS: Dict[str, EnvSpec] = {
    PRIMARY_ENV_ID: EnvSpec(PRIMARY_ENV_ID, continuous=True, solved_threshold=200.0),
    SECONDARY_ENV_ID: EnvSpec(SECONDARY_ENV_ID, continuous=False, solved_threshold=475.0),
}

METHODS: List[str] = ["ppo", "neat"]
BASELINES: List[str] = ["random", "heuristic"]
CONDITIONS: List[str] = ["clean", "perturbed"]


@dataclass(frozen=True)
class RunConfig:
    """Resolved configuration for one experiment invocation.

    Bundles the (possibly overridden) budget and seed settings so every module
    reads a single source of truth instead of re-deriving quick-mode values.

    Attributes:
        env_id: Environment to run on.
        n_seeds: Number of seeds per method per condition actually used.
        step_budget: Target total env.step() budget per training run.
        neat_population_size: NEAT population size for this run.
        eval_episodes: Episodes per fitness evaluation.
        quick: Whether this is a reduced quick-verification run.
        seed_offset: Added to BASE_SEED to derive per-run seeds.
    """

    env_id: str = PRIMARY_ENV_ID
    n_seeds: int = N_SEEDS
    step_budget: int = TOTAL_STEP_BUDGET
    neat_population_size: int = NEAT_POPULATION_SIZE
    eval_episodes: int = EVAL_EPISODES
    quick: bool = False
    seed_offset: int = 0

    @property
    def seeds(self) -> List[int]:
        """Return the explicit list of per-run seeds for this configuration."""
        start = BASE_SEED + self.seed_offset
        return [start + i for i in range(self.n_seeds)]


def quick_config(env_id: str = PRIMARY_ENV_ID) -> RunConfig:
    """Return a reduced configuration for fast end-to-end verification.

    Args:
        env_id: Environment to run the quick pipeline on.

    Returns:
        A :class:`RunConfig` with small seed count, budget, and population so
        the whole grid finishes in minutes rather than hours. Results produced
        with this config must be labelled as quick runs and never presented as
        the full experiment.
    """
    return RunConfig(
        env_id=env_id,
        n_seeds=QUICK_N_SEEDS,
        step_budget=QUICK_STEP_BUDGET,
        neat_population_size=QUICK_NEAT_POPULATION_SIZE,
        eval_episodes=QUICK_EVAL_EPISODES,
        quick=True,
    )


def full_config(env_id: str = PRIMARY_ENV_ID) -> RunConfig:
    """Return the full pre-registered configuration (N=20, 500k steps).

    Args:
        env_id: Environment to run the full pipeline on.

    Returns:
        A :class:`RunConfig` matching the pre-registered protocol.
    """
    return RunConfig(env_id=env_id)
