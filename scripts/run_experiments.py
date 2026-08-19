from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from gradevo import config
from gradevo.experiment import runner
from gradevo.metrics import hypothesis_tests as ht
from gradevo import tracking

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the GradEvo experiment grid.')
    parser.add_argument('--env', default=config.PRIMARY_ENV_ID, choices=[config.PRIMARY_ENV_ID, config.SECONDARY_ENV_ID], help='Environment id to run on.')
    parser.add_argument('--quick', action='store_true', help='Reduced N/budget/population for fast verification.')
    parser.add_argument('--seeds', type=int, default=None, help='Override number of seeds per method/condition.')
    parser.add_argument('--steps', type=int, default=None, help='Override the per-run step budget (e.g. 200000 for a medium diagnostic run).')
    parser.add_argument('--no-tracking', action='store_true', help='Disable MLflow tracking.')
    parser.add_argument('--no-analyze', action='store_true', help='Skip printing H0-H3 after the run.')
    parser.add_argument('--log-level', default='INFO', help='Python logging level (e.g. DEBUG, INFO).')
    return parser

def resolve_config(args: argparse.Namespace) -> config.RunConfig:
    from dataclasses import replace
    run_cfg = config.quick_config(args.env) if args.quick else config.full_config(args.env)
    if args.seeds is not None:
        run_cfg = replace(run_cfg, n_seeds=args.seeds)
    if args.steps is not None:
        run_cfg = replace(run_cfg, step_budget=args.steps)
    return run_cfg

def run(run_cfg: config.RunConfig, tracking_enabled: bool) -> None:
    tracking.setup_tracking(enabled=tracking_enabled)
    logging.info('Run config: %s', run_cfg)
    seed_runs = []
    trainers = (runner.train_ppo_seed, runner.train_es_seed, runner.train_cmaes_seed, runner.train_neat_seed)
    for seed in run_cfg.seeds:
        for trainer in trainers:
            seed_run = trainer(run_cfg.env_id, seed, run_cfg)
            with tracking.track_run(f'{seed_run.method}_seed{seed}', {'method': seed_run.method, 'seed': seed, 'step_budget': run_cfg.step_budget, 'env': run_cfg.env_id}, enabled=tracking_enabled):
                tracking.log_metrics({'clean_fitness': seed_run.clean_fitness, 'perturbed_fitness': seed_run.perturbed_fitness, 'realized_steps': seed_run.realized_steps, 'wall_clock_s': seed_run.wall_clock_s}, enabled=tracking_enabled)
            seed_runs.append(seed_run)
    baselines = runner.evaluate_baselines(run_cfg.env_id, run_cfg)
    paths = runner.save_results(run_cfg.env_id, run_cfg, seed_runs, baselines)
    logging.info('Wrote: %s', {k: str(v) for (k, v) in paths.items()})

def analyze(run_cfg: config.RunConfig) -> None:
    import pandas as pd
    slug = run_cfg.env_id.replace('-', '_')
    fitness_df = pd.read_csv(config.DATA_DIR / f'fitness_{slug}.csv')
    curves_df = pd.read_csv(config.DATA_DIR / f'curves_{slug}.csv')
    results = ht.run_pre_registered_suite(fitness_df, curves_df, run_cfg.step_budget)
    print('\n=== Pre-registered hypothesis tests (Bonferroni alpha = {:.4f}) ==='.format(config.ALPHA_CORRECTED))
    print(ht.results_to_dataframe(results).to_string(index=False))

def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    run_cfg = resolve_config(args)
    if run_cfg.n_seeds < config.MIN_ACCEPTABLE_SEEDS and (not run_cfg.quick):
        logging.warning('N=%d is below the documented minimum of %d seeds.', run_cfg.n_seeds, config.MIN_ACCEPTABLE_SEEDS)
    run(run_cfg, tracking_enabled=not args.no_tracking)
    if not args.no_analyze:
        analyze(run_cfg)
if __name__ == '__main__':
    main()
