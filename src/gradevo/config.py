from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_ROOT.parent.parent
DATA_DIR: Path = REPO_ROOT / 'data' / 'processed'
MODELS_DIR: Path = REPO_ROOT / 'models'
PPO_MODELS_DIR: Path = MODELS_DIR / 'ppo'
NEAT_MODELS_DIR: Path = MODELS_DIR / 'neat'
NEAT_CONFIG_DIR: Path = PACKAGE_ROOT / 'agents'
MLFLOW_DB_PATH: Path = REPO_ROOT / 'mlflow.db'
MLFLOW_TRACKING_URI: str = f'sqlite:///{MLFLOW_DB_PATH}'
MLFLOW_EXPERIMENT_NAME: str = 'gradevo'
PRIMARY_ENV_ID: str = 'LunarLanderContinuous-v3'
SECONDARY_ENV_ID: str = 'CartPole-v1'
MAX_EPISODE_STEPS: int = 1000
TOTAL_STEP_BUDGET: int = 500000
QUICK_STEP_BUDGET: int = 20000
NEAT_POPULATION_SIZE: int = 60
NEAT_FITNESS_EPISODES: int = 1
QUICK_NEAT_POPULATION_SIZE: int = 20
EXPECTED_EPISODE_LENGTH: int = 250
NEAT_PLANNING_MIN_EPISODE_LENGTH: int = 40
N_SEEDS: int = 20
QUICK_N_SEEDS: int = 3
MIN_ACCEPTABLE_SEEDS: int = 15
BASE_SEED: int = 1234
EVAL_EPISODES: int = 20
QUICK_EVAL_EPISODES: int = 5
GRAVITY_PERTURB_FRAC: float = 0.1
OBS_NOISE_FRAC: float = 0.02
ACTION_NOISE_FRAC: float = 0.05
LUNARLANDER_OBS_RANGES: Tuple[float, ...] = (2.0, 2.0, 5.0, 5.0, 6.28, 5.0, 0.0, 0.0)
ALPHA: float = 0.05
N_FAMILY_TESTS: int = 3
ALPHA_CORRECTED: float = ALPHA / N_FAMILY_TESTS
SHAPIRO_ALPHA: float = 0.05
PPO_POLICY: str = 'MlpPolicy'
PPO_HYPERPARAMS: Dict[str, object] = {'policy_kwargs': {'net_arch': [64, 64]}, 'verbose': 0}
ES_POPULATION_SIZE: int = 40
ES_SIGMA: float = 0.1
ES_LR: float = 0.03
QUICK_ES_POPULATION_SIZE: int = 12
CMAES_POPULATION_SIZE: int = 40
CMAES_SIGMA0: float = 0.1
QUICK_CMAES_POPULATION_SIZE: int = 12

@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    continuous: bool
    solved_threshold: float | None
ENV_SPECS: Dict[str, EnvSpec] = {PRIMARY_ENV_ID: EnvSpec(PRIMARY_ENV_ID, continuous=True, solved_threshold=200.0), SECONDARY_ENV_ID: EnvSpec(SECONDARY_ENV_ID, continuous=False, solved_threshold=475.0)}
METHODS: List[str] = ['ppo', 'neat']
BASELINES: List[str] = ['random', 'heuristic']
CONDITIONS: List[str] = ['clean', 'perturbed']

@dataclass(frozen=True)
class RunConfig:
    env_id: str = PRIMARY_ENV_ID
    n_seeds: int = N_SEEDS
    step_budget: int = TOTAL_STEP_BUDGET
    neat_population_size: int = NEAT_POPULATION_SIZE
    eval_episodes: int = EVAL_EPISODES
    quick: bool = False
    seed_offset: int = 0

    @property
    def seeds(self) -> List[int]:
        start = BASE_SEED + self.seed_offset
        return [start + i for i in range(self.n_seeds)]

def quick_config(env_id: str=PRIMARY_ENV_ID) -> RunConfig:
    return RunConfig(env_id=env_id, n_seeds=QUICK_N_SEEDS, step_budget=QUICK_STEP_BUDGET, neat_population_size=QUICK_NEAT_POPULATION_SIZE, eval_episodes=QUICK_EVAL_EPISODES, quick=True)

def full_config(env_id: str=PRIMARY_ENV_ID) -> RunConfig:
    return RunConfig(env_id=env_id)
