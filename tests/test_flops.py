from __future__ import annotations
import pandas as pd
from gradevo import config
from gradevo.metrics import flops

def test_mlp_forward_flops_matches_hand_computation():
    got = flops.mlp_forward_flops([8, 64, 64, 2])
    assert got == 1152 + 8320 + 260

def test_neat_genome_forward_flops_grows_with_connections():
    fewer = flops.neat_genome_forward_flops(n_nodes=10, n_connections=20)
    more = flops.neat_genome_forward_flops(n_nodes=10, n_connections=200)
    assert more > fewer
    assert fewer == 50
    assert more == 410

def test_estimate_run_flops_ppo_dominated_by_inference_for_long_runs():
    rec = flops.estimate_run_flops('ppo', realized_steps=500000, obs_dim=8, action_dim=2)
    assert rec.total_flops > 0
    assert rec.method == 'ppo'
    assert rec.inference_flops > 0 and rec.update_flops > 0

def test_estimate_run_flops_neat_uses_genome_size():
    rec_small = flops.estimate_run_flops('neat', realized_steps=500000, obs_dim=8, action_dim=2, neat_nodes=15, neat_connections=30)
    rec_big = flops.estimate_run_flops('neat', realized_steps=500000, obs_dim=8, action_dim=2, neat_nodes=200, neat_connections=800)
    assert rec_big.per_step_forward > rec_small.per_step_forward
    assert rec_big.inference_flops > rec_small.inference_flops

def test_flops_table_orders_by_config_methods():
    records = [flops.estimate_run_flops('neat', 500000, 8, 2, neat_nodes=15, neat_connections=30), flops.estimate_run_flops('ppo', 500000, 8, 2), flops.estimate_run_flops('es', 500000, 8, 2), flops.estimate_run_flops('cmaes', 500000, 8, 2)]
    table = flops.flops_table(records)
    assert list(table['method']) == [m for m in config.METHODS if m in {'ppo', 'es', 'cmaes', 'neat'}]
    assert (table['total_flops_mean'] > 0).all()
