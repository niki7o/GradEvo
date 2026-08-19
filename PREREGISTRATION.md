# Pre-registration: GradEvo Retake Experiments

This document locks the hypotheses, protocol, and analysis plan for the
GradEvo retake experiments **before** any retake data has been collected or
inspected. The prior submission's data (commit `abf787a`, published
August 2026) is treated as pilot data and is not reused as retake results.

Timestamp proof:
- This file's git commit hash is timestamped by GitHub servers.
- The commit is tagged `prereg-retake-v1` for stable citation.
- An Internet Archive Wayback Machine snapshot of the tag URL provides
  a second independent timestamp.
- A Zenodo DOI will be added below once minted (does not affect the
  primary GitHub + Wayback proof).

**Zenodo DOI:** [10.5281/zenodo.22010887](https://doi.org/10.5281/zenodo.22010887)

---

## Research question

Under a matched budget of 500,000 environment interaction steps on
continuous-control tasks, do four policy-search methods (PPO, ES, CMA-ES,
NEAT) differ in final fitness, sample efficiency, robustness under a
held-out perturbation, and pairwise comparison of the PPO-vs-NEAT gap,
under identically fixed decoding settings and no per-task hyperparameter
tuning?

## Environments (locked)

- **Primary:** `LunarLanderContinuous-v3` (Gymnasium; 8-D observation, 2-D
  continuous action; published solved threshold 200).
- **Secondary:** `Pendulum-v1` (Gymnasium; 3-D observation, 1-D continuous
  action; no official solved threshold, mean return over -200 conventionally
  treated as a solid policy).

## Methods (locked)

Four learned methods, all sharing the same fixed [64, 64] MLP policy where
applicable:

- **PPO** (stable-baselines3 defaults, no per-task tuning).
- **ES** (OpenAI-style Evolution Strategies with antithetic sampling and
  rank shaping; population 40, sigma 0.1, learning rate 0.03).
- **CMA-ES** (Hansen's `cma` package; population 40, sigma_0 0.1, library
  defaults for adaptive covariance updates).
- **NEAT** (neat-python defaults; population 60, speciated, feed-forward
  genomes only).

Two baselines:
- **Random-action baseline.**
- **Hand-coded PD-style heuristic controller.**

One additional arm added for the retake:
- **Tuned-PPO** using hyperparameters from Schulman et al. (2017) or the
  stable-baselines3-zoo tuned configuration for LunarLanderContinuous.
  Reported separately from the four-method core comparison and used only
  to answer H5.

## Compute-budget matching (locked)

The controlled variable is total `env.step()` calls per training run,
fixed at **500,000 per training run** for the primary task and at the
same or documented reduced value for `Pendulum-v1`. Wall-clock time is
reported only as secondary context. Per-method FLOPs are reported for
compute-fairness sensitivity.

## Perturbation protocol (locked, eval-only)

Applied only at evaluation, never during training, via a
`gymnasium.Wrapper`:

- Per-episode gravity scaled by a factor drawn in ±10%.
- Additive Gaussian observation noise, sigma = 2% of each dimension's
  typical range, boolean flags excluded.
- Additive Gaussian action noise, sigma = 5% of the action range, applied
  before physics and clipped.

## Seeds and N (locked)

**N = 20** independent seeds per method per condition per environment.
Seeds are `1234, 1235, ..., 1253`. Every random source in every method
consumes the seed explicitly; no hidden global RNG. If a training run
crashes for a specific seed, it is documented and the seed is retried
once; if it crashes again, the seed is dropped and reported.

## Pre-registered hypotheses

Family-wise Bonferroni correction across H1-H5:
alpha_corrected = 0.05 / 5 = **0.010**.

H0 is a precondition gate, reported separately with its own Bonferroni
correction over per-method baseline comparisons.

| ID | Hypothesis | Test | Direction |
|----|------------|------|-----------|
| **H0** | Each learned method beats the random-action baseline **and** the hand-coded heuristic on clean fitness | Mann-Whitney U | one-sided, greater |
| **H1** | NEAT's final clean fitness differs from PPO's | Mann-Whitney U | two-sided |
| **H2** | PPO is more sample-efficient than NEAT (higher normalized learning-curve AUC) | Wilcoxon signed-rank (paired) | one-sided, PPO > NEAT |
| **H3** | NEAT degrades less than PPO under held-out perturbation (smaller clean-minus-perturbed drop) | Mann-Whitney U | one-sided, NEAT drop < PPO drop |
| **H4** | PPO-at-SB3-defaults and ES-at-reference-defaults differ in final clean fitness. Framed as a bounded defaults-vs-defaults comparison, not a general claim about "gradient vs. gradient-free" learning | Mann-Whitney U | two-sided |
| **H5** | Tuned-PPO beats PPO-at-SB3-defaults on final clean fitness | Mann-Whitney U | one-sided, tuned > default |

### Explicit framing note on H4

Prior submission framed H4 as "gradient contribution (PPO > ES)." That
framing was too clean, because PPO and ES differ not only in whether they
use a gradient but also in library implementation, RNG streams, tuned vs.
default hyperparameters, and update-rule structure. The retake H4 is
therefore reframed as a **two-sided, bounded** comparison between two
specific default configurations, and no claim about the general effect of
"having a gradient" is made from this contrast.

## Statistical protocol (locked)

- Non-parametric primary tests as tabled above.
- Rank-biserial correlation as effect size for every rank-based test.
- Shapiro-Wilk normality check and parametric t-test cross-check reported
  for transparency; non-parametric p-value is authoritative.
- BCa bootstrap 95% confidence intervals on every reported mean,
  10,000 resamples.
- **Robustness check:** a permutation test with 10,000 permutations is
  reported alongside every Mann-Whitney U result. If the two tests
  disagree qualitatively on reject/fail-to-reject, this is flagged.
- Null results are reported as "consistent with no detected difference,"
  never as proof of equality.

## Analyses that are NOT pre-registered

The following, if reported, will be explicitly labeled as **exploratory**:

- Any hypothesis not in the H0-H5 table above.
- Any subgroup analysis by seed range, environment, or method beyond
  those in the tabled tests.
- Any post-hoc reframing of a fail-to-reject result as a positive claim.

## Data handling

Raw per-seed training logs, model checkpoints, and evaluation fitness
values are committed to the repository under `data/processed/` and
`models/`. Aggregation and analysis logic is in `src/gradevo`; the
analysis notebook (`notebooks/gradevo_analysis.ipynb`) only imports,
loads, and displays.

## Contamination and threats to validity

Threats acknowledged in advance:

- Env-step matching does not equalize per-step FLOPs; a FLOPs sensitivity
  table is reported.
- Perturbation magnitudes are small by design; results are specific to
  this perturbation family.
- Only two environments are used; conclusions do not generalize beyond
  low-dimensional continuous control.
- N = 20 gives modest statistical power for detecting small effects; this
  is why non-parametric tests, effect sizes, and BCa CIs are reported.
- No per-task hyperparameter tuning by design (except the tuned-PPO arm
  used only for H5); no method is claimed to be at its ceiling.

---

**Signed by commit hash of this file** on the retake branch of
`github.com/niki7o/GradEvo`.
