from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from gradevo import config

def mlp_forward_flops(layer_sizes: Sequence[int]) -> int:
    total = 0
    for (m, n) in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += 2 * m * n
        total += n
        total += n
    return int(total)

def neat_genome_forward_flops(n_nodes: int, n_connections: int) -> int:
    return int(2 * n_connections + n_nodes)

@dataclass(frozen=True)
class MethodFlops:
    method: str
    seed: int
    per_step_forward: int
    inference_flops: int
    update_flops: int
    total_flops: int
    realized_steps: int

def _ppo_update_flops(realized_steps: int, obs_dim: int, action_dim: int) -> int:
    fwd = mlp_forward_flops([obs_dim, 64, 64, action_dim])
    return int(3 * fwd * 10 * realized_steps)

def _es_update_flops(realized_steps: int, obs_dim: int, action_dim: int, pop: int) -> int:
    param_count = obs_dim * 64 + 64 + 64 * 64 + 64 + 64 * action_dim + action_dim
    est_gens = max(1, realized_steps // (pop * config.EXPECTED_EPISODE_LENGTH))
    return int(3 * param_count * est_gens)

def _cmaes_update_flops(realized_steps: int, obs_dim: int, action_dim: int, pop: int) -> int:
    d = obs_dim * 64 + 64 + 64 * 64 + 64 + 64 * action_dim + action_dim
    est_gens = max(1, realized_steps // (pop * config.EXPECTED_EPISODE_LENGTH))
    return int(d * d * est_gens)

def _neat_update_flops(realized_steps: int) -> int:
    return int(10000.0 * max(1, realized_steps // 5000))

def estimate_run_flops(method: str, realized_steps: int, obs_dim: int, action_dim: int, seed: int=0, *, neat_nodes: int=0, neat_connections: int=0, es_population: int | None=None) -> MethodFlops:
    if method == 'ppo':
        fwd = mlp_forward_flops([obs_dim, 64, 64, action_dim])
        infer = fwd * realized_steps
        update = _ppo_update_flops(realized_steps, obs_dim, action_dim)
    elif method == 'es':
        fwd = mlp_forward_flops([obs_dim, 64, 64, action_dim])
        infer = fwd * realized_steps
        pop = es_population if es_population is not None else config.ES_POPULATION_SIZE
        update = _es_update_flops(realized_steps, obs_dim, action_dim, pop)
    elif method == 'cmaes':
        fwd = mlp_forward_flops([obs_dim, 64, 64, action_dim])
        infer = fwd * realized_steps
        pop = es_population if es_population is not None else config.CMAES_POPULATION_SIZE
        update = _cmaes_update_flops(realized_steps, obs_dim, action_dim, pop)
    elif method == 'neat':
        fwd = neat_genome_forward_flops(neat_nodes, neat_connections)
        infer = fwd * realized_steps
        update = _neat_update_flops(realized_steps)
    else:
        raise ValueError(f'unknown method: {method}')
    return MethodFlops(method=method, seed=seed, per_step_forward=fwd, inference_flops=int(infer), update_flops=int(update), total_flops=int(infer + update), realized_steps=int(realized_steps))

def flops_table(records: Sequence[MethodFlops]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method in config.METHODS:
        method_records = [r for r in records if r.method == method]
        if not method_records:
            continue
        infer = np.array([r.inference_flops for r in method_records], dtype=np.float64)
        update = np.array([r.update_flops for r in method_records], dtype=np.float64)
        total = infer + update
        rows.append({'method': method, 'n_seeds': len(method_records), 'per_step_forward_flops': int(np.median([r.per_step_forward for r in method_records])), 'inference_flops_mean': float(infer.mean()), 'update_flops_mean': float(update.mean()), 'total_flops_mean': float(total.mean()), 'total_flops_std': float(total.std(ddof=1)) if len(total) > 1 else 0.0})
    return pd.DataFrame(rows)
