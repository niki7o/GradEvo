# GradEvo

**A matched-compute comparison of gradient reinforcement learning (PPO) and
neuroevolution (NEAT) on continuous control.**

GradEvo asks a deliberately narrow, rigorously tested question: *given a fixed
budget of environment interaction steps, does gradient descent through a
fixed-topology policy (PPO, Adam) or population-based search over weights **and**
topology (NEAT) reach higher task performance - and whose solutions generalize
better to a held-out perturbed environment?* The contribution is **methodological
rigor** - pre-registered hypotheses, exact step-budget matching, non-parametric
tests with a family-wise correction, and calibrated reporting - not a new
algorithm.

> **Reading guide.** The narrative lives in
> [`notebooks/gradevo_analysis.ipynb`](notebooks/gradevo_analysis.ipynb). All
> logic lives in [`src/gradevo`](src/gradevo). The notebook only imports, loads
> result tables, and displays figures - by design.

---

## Research question

> Under a matched budget of total environment interaction steps
> (`env.step()` calls), does PPO or NEAT reach higher fitness on a
> continuous-control task, and which method's solutions degrade less under a
> held-out domain-randomized perturbation?

## Pre-registered hypotheses

**H1, H2, H3** share a family-wise **Bonferroni** correction,
α_corrected = 0.05 / 3 ≈ **0.0167**. **H0** is a precondition gate, reported
separately.

| ID | Hypothesis | Test | Correction |
|----|------------|------|------------|
| **H0** | Both trained methods beat a random-action baseline **and** a hand-coded heuristic | Mann-Whitney U (one-sided) vs. each baseline | Bonferroni over the 2 comparisons |
| **H1** | NEAT's final fitness is **not significantly different** from PPO's | Mann-Whitney U (two-sided), rank-biserial effect size | Bonferroni, α = 0.0167 |
| **H2** | PPO is **more sample-efficient** (higher normalized learning-curve AUC) | Wilcoxon signed-rank (paired, one-sided) | Bonferroni, α = 0.0167 |
| **H3** | NEAT **degrades less** under held-out perturbation (smaller Δfitness) | Mann-Whitney U (one-sided) on the fitness drop | Bonferroni, α = 0.0167 |

We default to **non-parametric** tests given small seed counts, report a
**Shapiro-Wilk** normality check, and include a **parametric t-test cross-check**
alongside each result for transparency. A **null result is a legitimate outcome** -
H1 is framed so that *failing to reject* is itself informative.

## Method at a glance

- **Primary task:** `LunarLanderContinuous-v3` (continuous, 8-D obs, 2-D action,
  "solved" ≈ 200). **Secondary sanity check:** `CartPole-v1` (discrete).
- **Compute budget = total `env.step()` calls**, measured exactly by a
  step-counting wrapper (default 500,000 steps). This is the controlled
  variable; wall-clock time is reported only as secondary context.
- **Held-out perturbation (H3), eval-time only:** ±10% per-episode gravity, 2%
  observation noise, 5% action noise, via a `gymnasium.Wrapper`. **Never active
  during training.**
- **Seeds:** target **N = 20** per method per condition; **N = 15** is the
  documented fallback. The actual N is recorded in every results table.
- **Libraries:** `stable-baselines3` (PPO/Adam), `neat-python` (NEAT),
  `gymnasium`, `scipy.stats`, `mlflow` (SQLite tracking). Established
  implementations are used deliberately - the contribution is the experimental
  design, not reimplementing well-known algorithms.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
pip install -e .                                        # installs the gradevo package

# Fast end-to-end verification (minutes; reduced N/budget/population):
python scripts/run_experiments.py --quick

# Full pre-registered run on the primary task (hours; N=20, 500k steps):
python scripts/run_experiments.py --env LunarLanderContinuous-v3

# Regenerate the notebook deterministically, then open it:
python scripts/build_notebook.py
jupyter lab notebooks/gradevo_analysis.ipynb
```

`LunarLanderContinuous-v3` needs Box2D. On macOS: `brew install swig` first if
the `box2d-py` build fails, then reinstall requirements. See
[`docs/reproducing.md`](docs/reproducing.md) for exact commands and expected
runtimes.

## Repository layout

```
gradevo/
├── notebooks/gradevo_analysis.ipynb   # narrative + figures ONLY (no logic)
├── src/gradevo/
│   ├── config.py                      # seeds, budgets, α, perturbation magnitudes
│   ├── envs/                          # step-counter + perturbation wrappers
│   ├── agents/                        # baselines, PPO (SB3), NEAT (neat-python)
│   ├── experiment/                    # N-seed runner + budget-matching logic
│   ├── metrics/                       # H0-H3 tests, effect sizes, AUC
│   ├── plots/figures.py              # all figure generation
│   └── tracking.py                    # MLflow (SQLite) helper
├── scripts/                           # run_experiments.py, build_notebook.py
├── models/                            # exported PPO .zip + NEAT pickled genomes
├── data/processed/                    # tidy result CSVs + run metadata
├── tests/                             # pytest suite (envs, agents, budget, stats)
└── docs/reproducing.md
```

## Testing

```bash
pytest                 # full suite
pytest tests/test_hypothesis_tests.py   # statistics known-answer tests
```

The statistical and environment modules are the ones where a silent bug would
invalidate the science, so they are tested hardest: identical distributions must
yield large p-values, separated distributions small ones; the step counter is
checked exactly against a fixed-length episode; the perturbation wrapper is
verified to change observations beyond the noise floor and to leave zero-range
(boolean) dimensions untouched; and a leakage guard asserts the training
environment is never the perturbed one. A fixed-seed regression test guards
against silent library/environment drift.

## Previous research

- **Stanley & Miikkulainen (2002).** *Evolving Neural Networks through Augmenting
  Topologies.* Evolutionary Computation 10(2). - the NEAT algorithm.
- **Schulman, Wolski, Dhariwal, Radford & Klimov (2017).** *Proximal Policy
  Optimization Algorithms.* arXiv:1707.06347. - the PPO algorithm.
- **Salimans, Ho, Chen, Sidor & Sutskever (2017).** *Evolution Strategies as a
  Scalable Alternative to Reinforcement Learning.* arXiv:1703.03864. - the
  closest prior "evolution vs. gradient RL" comparison.
- **Such, Madhavan, Conti, Lehman, Stanley & Clune (2017).** *Deep
  Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training
  Deep Neural Networks for Reinforcement Learning.* arXiv:1712.06567. - directly
  relevant prior comparison; §7 of the notebook compares numbers with explicit
  caveats.

The notebook's §7 contains a "prior reported results vs. this project" table with
an honest note on why direct comparison is imperfect (different environments,
network sizes, and compute scales).

## Threats to validity (summary)

Step-count is an imperfect proxy for equal *compute* (per-step FLOP costs differ
between PPO and NEAT - wall-clock is reported so the gap is visible); the chosen
perturbation magnitudes are small by design and results are specific to that
family; the environment scope is two Gym tasks; and seed counts give modest
statistical power, so nulls are reported as "no detected difference," never as
proof of equality. Full discussion in the notebook's §8.

## License

MIT - see [`LICENSE`](LICENSE).
