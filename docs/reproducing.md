# Reproducing GradEvo

This document gives exact commands and expected runtimes to reproduce every
number and figure in the notebook, on a clean machine.

## 1. Environment setup (macOS / Linux)

```bash
# From the repository root:
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                     # makes `import gradevo` work everywhere
```

### Box2D note (LunarLander only)

`LunarLanderContinuous-v3` depends on Box2D, which builds from source and needs
SWIG. On macOS the reliable path is:

```bash
pip install swig
PATH="$(python -c 'import sys,os;print(os.path.dirname(sys.executable))'):$PATH" \
  pip install --no-build-isolation box2d-py
pip install pygame
```

The secondary task `CartPole-v1` needs **no** Box2D, so you can run and test the
entire pipeline without it.

## 2. Fast verification (minutes)

This runs the whole pipeline end-to-end at reduced fidelity (quick config:
N=3 seeds, 20,000-step budget, population 20) and prints the H0–H3 table. Use it
to confirm the code works before committing to the full run.

```bash
# Primary task, reduced fidelity (~3–8 min on a laptop CPU):
python scripts/run_experiments.py --env LunarLanderContinuous-v3 --quick

# Secondary sanity-check task (~1–2 min, no Box2D needed):
python scripts/run_experiments.py --env CartPole-v1 --quick
```

> **Quick runs are clearly reduced.** Their tables are labelled `quick=True` in
> `data/processed/metadata_*.json`. Never present quick-run numbers as the full
> pre-registered result.

## 3. Full pre-registered run (hours)

The full protocol is **N = 20** seeds per method per condition and a
**500,000-step** budget per training run. With two methods and per-seed clean +
perturbed evaluation, expect the primary task to take **several hours** on a CPU
(dominated by PPO's 20 × 500k = 10M training steps plus NEAT's equivalent).

```bash
python scripts/run_experiments.py --env LunarLanderContinuous-v3
```

If compute is constrained, **N = 15** is the documented fallback; state it
explicitly:

```bash
python scripts/run_experiments.py --env LunarLanderContinuous-v3 --seeds 15
```

The actual N is recorded in every results table and in the notebook's printed
"Actual N" line, so a reduced N is never silently substituted.

### Approximate runtime budget (CPU, order-of-magnitude)

| Run | Config | Env steps (total) | Approx. wall-clock |
|-----|--------|-------------------|--------------------|
| `--quick` LunarLander | N=3, 20k | ~120k train + eval | 3–8 min |
| `--quick` CartPole | N=3, 20k | ~120k train + eval | 1–2 min |
| Full LunarLander | N=20, 500k | ~20M train + eval | several hours |
| Full CartPole | N=20, 500k | ~20M train + eval | ~1–2 hours |

Exact wall-clock is recorded per method in `data/processed/budget_*.csv` and is
reported as a *secondary* metric — the controlled variable is environment steps.

## 4. Regenerate the notebook and view results

```bash
python scripts/build_notebook.py                 # deterministic rebuild
jupyter lab notebooks/gradevo_analysis.ipynb     # or: jupyter notebook
```

Inside the notebook, set `ENV_ID` and `STEP_BUDGET` at the top of the setup cell
to match the run you want to analyze (defaults: primary env, full budget). For a
quick-run analysis, set `STEP_BUDGET = config.QUICK_STEP_BUDGET`.

## 5. Experiment tracking (optional)

Runs are logged to a local MLflow SQLite store unless `--no-tracking` is passed.
Browse them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then open http://127.0.0.1:5000
```

## 6. Tests

```bash
pytest                    # full suite
pytest -k hypothesis      # just the statistics known-answer tests
```

The environment/agent tests use CartPole and an in-process dummy env, so they
run **without Box2D**. If Box2D/SB3/neat-python are absent, the training-specific
paths are simply not exercised; the statistics and wrapper tests still run.

## 7. Outputs produced

| Path | Contents |
|------|----------|
| `data/processed/fitness_<env>.csv` | Per-seed clean/perturbed fitness (methods + baselines) |
| `data/processed/curves_<env>.csv` | Per-seed learning curves (step, fitness) |
| `data/processed/budget_<env>.csv` | Requested vs. realized steps + wall-clock |
| `data/processed/metadata_<env>.json` | Seeds, budget, N, α, version |
| `models/ppo/ppo_<env>_seed<N>.zip` | Trained SB3 PPO checkpoints |
| `models/neat/neat_<env>_seed<N>.pkl` | Pickled NEAT winner genomes |
