# GradEvo

**A matched-compute, pre-registered comparison of four policy-search families
(PPO, ES, CMA-ES, NEAT) on continuous control.**

GradEvo asks a narrow, rigorously tested question: given a fixed budget of
environment interaction steps, how do four mechanically different families of
policy search compare on final fitness, sample efficiency, robustness under a
held-out perturbation, and the gradient-vs-topology decomposition of the
PPO-NEAT gap? The contribution is **methodological rigor**: pre-registered
hypotheses, exact step-budget matching, non-parametric tests with a
family-wise Bonferroni correction, bias-corrected bootstrap confidence
intervals, a FLOPs sensitivity table, and calibrated reporting including
null results. Not a new algorithm.

> **Reading guide.** The narrative lives in
> [`notebooks/gradevo_analysis.ipynb`](notebooks/gradevo_analysis.ipynb). All
> logic lives in [`src/gradevo`](src/gradevo). The notebook only imports,
> loads result tables, and displays figures.

---

## The 2x2 design

Four search paradigms optimize the same class of control problem in
mechanically different ways:

| Method   | Gradient? | Topology              | Search adaptivity            |
|----------|-----------|-----------------------|------------------------------|
| **PPO**  | yes (Adam + clipped surrogate) | fixed [64, 64] MLP | N/A                          |
| **ES**   | no        | fixed [64, 64] MLP    | isotropic Gaussian, fixed σ  |
| **CMA-ES** | no      | fixed [64, 64] MLP    | full covariance, adaptive    |
| **NEAT** | no        | grown via mutation    | speciated population         |

The pairwise contrasts each isolate a single factor:

- **PPO vs ES.** Same architecture, same env-step budget. The only
  difference is the gradient. Contribution attributable to *gradient
  information*.
- **ES vs CMA-ES.** Both gradient-free, same architecture. Contribution
  attributable to *learning the search geometry*.
- **CMA-ES vs NEAT.** Both gradient-free and adaptive; NEAT also grows
  topology. Contribution attributable to *topology search*.

Prior work (Salimans 2017, Such 2017) each compare one gradient-free method
against gradient RL and stop there. Placing three gradient-free methods
along a progression of increasing search sophistication is the specific
novel angle here.

## Pre-registered hypotheses

**H1, H2, H3, H4** share a family-wise **Bonferroni** correction,
α_corrected = 0.05 / 4 = **0.0125**. **H0** is a precondition gate,
reported separately per method with its own Bonferroni over the two
baseline comparisons.

| ID   | Hypothesis                                                                                              | Test                                      |
|------|---------------------------------------------------------------------------------------------------------|-------------------------------------------|
| **H0** | Each trained method beats a random-action baseline **and** a hand-coded heuristic                    | Mann-Whitney U (one-sided), Bonferroni over 2 |
| **H1** | NEAT's final fitness is **not significantly different** from PPO's                                    | Mann-Whitney U (two-sided)                |
| **H2** | PPO is **more sample-efficient** than NEAT (higher normalized learning-curve AUC)                     | Wilcoxon signed-rank (paired, one-sided)  |
| **H3** | NEAT **degrades less** under held-out perturbation (smaller Δfitness)                                 | Mann-Whitney U (one-sided)                |
| **H4** | **PPO > ES** on final fitness, holding architecture constant (the "gradient contribution")            | Mann-Whitney U (one-sided)                |

Non-parametric tests are the default given small seed counts. Rank-biserial
correlation is reported as the effect size for every rank-based test. A
Shapiro-Wilk normality check and a parametric t-test cross-check are
computed for transparency but the non-parametric p-value is authoritative.
Every reported mean also carries a **BCa bootstrap 95% CI** (bias-corrected
and accelerated, 10,000 resamples), preferred over the naive percentile
bootstrap for small skewed samples. A null result is a legitimate outcome:
H1, H3, H4 are framed so that failing to reject is itself informative.

## Headline results (LunarLanderContinuous-v3, N=20, 500k steps)

The pre-registered suite ran on the primary task. The results are reported
plainly, including where the gates did not pass:

- **H0 vs random.** All four learned methods clearly beat the random-action
  baseline (p < 1e-4, effect size ≈ +1.0 for PPO, ES, CMA-ES; +0.79 for NEAT).
- **H0 vs heuristic.** None of the learned methods decisively beat the
  hand-coded PD-style heuristic. Effect sizes: PPO −0.95, NEAT −0.90 (both
  clearly worse than the heuristic), ES −0.33 (mildly worse), CMA-ES −0.05
  (essentially tied). A competent domain-knowledge baseline remains
  competitive with untuned learned policies at this budget.
- **H1 (NEAT ≠ PPO).** Failed to reject (p = 0.014, just above the
  Bonferroni-corrected α = 0.0125). Consistent with no detected difference
  in final fitness.
- **H2 (PPO more sample-efficient).** Failed to reject.
- **H3 (NEAT more robust).** Failed to reject.
- **H4 (PPO > ES, the gradient contribution).** Failed to reject strongly
  (p = 0.9997, effect −0.63 in the *opposite* direction). At 500k env-steps
  on this task with default hyperparameters, the exact policy gradient did
  **not** produce a detectable advantage over gradient-free weight
  evolution on the same fixed [64, 64] MLP.

These results are bounded to this task, budget, and untuned hyperparameter
setting. The notebook conclusion (§10) discusses their scope and
interpretation in more detail.

## Method at a glance

- **Primary task:** `LunarLanderContinuous-v3` (continuous, 8-D observation,
  2-D action, published "solved" threshold 200). **Secondary sanity check:**
  `CartPole-v1` (discrete).
- **Compute budget = total `env.step()` calls**, counted exactly by a
  step-counting wrapper (default 500,000 steps per training run). This is
  the controlled variable. Wall-clock time is reported only as secondary
  context.
- **FLOPs sensitivity.** Since a PPO step and a NEAT step do not cost the
  same computation, `gradevo.metrics.flops` reports per-method analytical
  FLOPs so the "env-step matching is a fair unit" claim is empirically
  defensible, not just asserted.
- **Held-out perturbation (H3), eval-time only:** ±10% per-episode gravity,
  2% observation noise (boolean leg-contact flags excluded), 5% action
  noise, via a `gymnasium.Wrapper`. Never active during training.
- **Seeds:** N = 20 per method per condition; **N = 15** is the documented
  fallback. The actual N is recorded in every results table.
- **Libraries:** `stable-baselines3` (PPO/Adam), `neat-python` (NEAT),
  `cma` (Hansen's CMA-ES), `gymnasium`, `scipy.stats`. Established
  implementations are used deliberately. The contribution is the
  experimental design, not reimplementing well-known algorithms.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
pip install -e .                                        # installs the gradevo package

# Fast end-to-end verification (minutes; reduced N/budget/population):
python scripts/run_experiments.py --quick

# Full pre-registered run on the primary task (hours; N=20, 500k steps, 4 methods):
python scripts/run_experiments.py --env LunarLanderContinuous-v3

# Regenerate the notebook deterministically, then open it:
python scripts/build_notebook.py
jupyter lab notebooks/gradevo_analysis.ipynb
```

`LunarLanderContinuous-v3` needs Box2D. On macOS: `brew install swig` first
if the `box2d-py` build fails, then reinstall requirements.

## Repository layout

```
gradevo/
├── notebooks/gradevo_analysis.ipynb    # narrative + figures (no logic)
├── src/gradevo/
│   ├── config.py                       # seeds, budgets, alpha, perturbation magnitudes
│   ├── envs/                           # step-counter + perturbation wrappers
│   ├── agents/                         # baselines, PPO, ES, CMA-ES, NEAT
│   ├── experiment/                     # N-seed runner + budget-matching logic
│   ├── metrics/                        # H0-H4 tests, BCa CIs, FLOPs, effect sizes
│   └── plots/figures.py                # all figure generation
├── scripts/                            # run_experiments.py, build_notebook.py
├── models/                             # exported checkpoints per method per seed
├── data/processed/                     # tidy result CSVs + run metadata
└── tests/                              # 65 pytest cases
```

## Testing

```bash
pytest                                    # full suite (65 tests)
pytest tests/test_hypothesis_tests.py     # statistics known-answer tests
pytest tests/test_bootstrap.py            # BCa CI coverage tests
pytest tests/test_flops.py                # FLOPs arithmetic tests
```

The statistical, environment, and compute-accounting modules are where a
silent bug would invalidate the science, so they are tested hardest:
identical distributions must yield large p-values, separated distributions
small ones; H4 rejects when PPO clearly beats ES and fails to reject when
they are equal; the step counter is checked exactly against a fixed-length
episode; the perturbation wrapper is verified to change observations beyond
the noise floor and to leave zero-range (boolean) dimensions untouched; a
leakage guard asserts the training environment is never the perturbed one;
BCa CIs are verified to cover the true mean on normal samples and degenerate
correctly on constant samples; FLOPs counts match hand-computed values and
scale monotonically with genome size. A fixed-seed regression test guards
against silent library or environment drift.

## Previous research

- **Stanley & Miikkulainen (2002).** *Evolving Neural Networks through
  Augmenting Topologies.* Evolutionary Computation 10(2). The NEAT algorithm.
- **Schulman, Wolski, Dhariwal, Radford & Klimov (2017).** *Proximal Policy
  Optimization Algorithms.* arXiv:1707.06347. The PPO algorithm.
- **Salimans, Ho, Chen, Sidor & Sutskever (2017).** *Evolution Strategies as
  a Scalable Alternative to Reinforcement Learning.* arXiv:1703.03864. The
  ES formulation used here (antithetic sampling, rank shaping) and the
  closest prior "evolution vs. gradient RL" comparison.
- **Hansen (2016).** *The CMA Evolution Strategy: A Tutorial.*
  arXiv:1604.00772. CMA-ES.
- **Such, Madhavan, Conti, Lehman, Stanley & Clune (2017).** *Deep
  Neuroevolution: Genetic Algorithms Are a Competitive Alternative for
  Training Deep Neural Networks for Reinforcement Learning.*
  arXiv:1712.06567. Directly relevant prior comparison.

The notebook's §8 contains a "prior reported results vs. this project" table
with an honest note on why direct comparison is imperfect (different
environments, network sizes, and compute scales).

## Threats to validity (summary)

Step-count is an imperfect proxy for equal *compute* since per-step FLOP
costs differ between the four methods. The FLOPs sensitivity table (§6 of
the notebook) makes this gap empirically visible; wall-clock is also
reported. The chosen perturbation magnitudes are small by design and
results are specific to that family. The environment scope is two Gym tasks
(one primary, one secondary). Seed counts give modest statistical power, so
nulls are reported as "no detected difference," never as proof of equality.
No per-task hyperparameter tuning is applied by design (to avoid an unfair
tuned advantage), which means no method is shown at its ceiling. Full
discussion in the notebook's §9.

## License

MIT. See [`LICENSE`](LICENSE).
