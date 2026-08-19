from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from gradevo import config
from gradevo.metrics import learning_curves as lc
ArrayLike = Sequence[float]

@dataclass(frozen=True)
class HypothesisResult:
    hypothesis_id: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    alpha: float
    reject_null: bool
    n_a: int
    n_b: Optional[int] = None
    parametric_p: Optional[float] = None
    normal_a: Optional[bool] = None
    normal_b: Optional[bool] = None
    extra: Dict[str, float] = field(default_factory=dict)

def normality_check(sample: ArrayLike, alpha: float=config.SHAPIRO_ALPHA) -> bool:
    arr = np.asarray(sample, dtype=np.float64)
    if arr.size < 3:
        return False
    if np.allclose(arr, arr[0]):
        return False
    (_, p_value) = stats.shapiro(arr)
    return bool(p_value > alpha)

def rank_biserial(a: ArrayLike, b: ArrayLike) -> float:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    (n_a, n_b) = (arr_a.size, arr_b.size)
    if n_a == 0 or n_b == 0:
        return 0.0
    (u_a, _) = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
    return float(2.0 * u_a / (n_a * n_b) - 1.0)

def matched_pairs_rank_biserial(deltas: ArrayLike) -> float:
    arr = np.asarray(deltas, dtype=np.float64)
    nonzero = arr[arr != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    total = ranks.sum()
    pos = ranks[nonzero > 0].sum()
    neg = ranks[nonzero < 0].sum()
    return float((pos - neg) / total)

def mann_whitney(a: ArrayLike, b: ArrayLike, alternative: str='two-sided') -> Tuple[float, float]:
    (u_stat, p_value) = stats.mannwhitneyu(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), alternative=alternative)
    return (float(u_stat), float(p_value))

def wilcoxon_paired(a: ArrayLike, b: ArrayLike, alternative: str='two-sided') -> Tuple[float, float]:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError('Wilcoxon requires paired samples of equal length')
    if np.allclose(arr_a, arr_b):
        raise ValueError('All paired differences are zero; Wilcoxon undefined')
    (w_stat, p_value) = stats.wilcoxon(arr_a, arr_b, alternative=alternative)
    return (float(w_stat), float(p_value))

def bonferroni_reject(p_value: float, alpha_corrected: float) -> bool:
    return bool(p_value < alpha_corrected)

def test_h0_baselines(trained: ArrayLike, random_baseline: ArrayLike, heuristic_baseline: ArrayLike, method_name: str='method') -> List[HypothesisResult]:
    alpha_h0 = config.ALPHA / 2
    results: List[HypothesisResult] = []
    for (label, baseline) in (('random', random_baseline), ('heuristic', heuristic_baseline)):
        (u_stat, p_value) = mann_whitney(trained, baseline, alternative='greater')
        results.append(HypothesisResult(hypothesis_id=f'H0[{method_name}>{label}]', test_name='Mann-Whitney U (one-sided, greater)', statistic=u_stat, p_value=p_value, effect_size=rank_biserial(trained, baseline), effect_size_name='rank-biserial', alpha=alpha_h0, reject_null=bonferroni_reject(p_value, alpha_h0), n_a=len(list(trained)), n_b=len(list(baseline)), parametric_p=float(stats.ttest_ind(trained, baseline, alternative='greater').pvalue), normal_a=normality_check(trained), normal_b=normality_check(baseline), extra={'median_trained': float(np.median(trained)), 'median_baseline': float(np.median(baseline))}))
    return results

def test_h1_final_fitness(ppo_final: ArrayLike, neat_final: ArrayLike, alpha_corrected: float=config.ALPHA_CORRECTED) -> HypothesisResult:
    (u_stat, p_value) = mann_whitney(neat_final, ppo_final, alternative='two-sided')
    return HypothesisResult(hypothesis_id='H1', test_name='Mann-Whitney U (two-sided)', statistic=u_stat, p_value=p_value, effect_size=rank_biserial(neat_final, ppo_final), effect_size_name='rank-biserial (NEAT vs PPO)', alpha=alpha_corrected, reject_null=bonferroni_reject(p_value, alpha_corrected), n_a=len(list(neat_final)), n_b=len(list(ppo_final)), parametric_p=float(stats.ttest_ind(neat_final, ppo_final).pvalue), normal_a=normality_check(neat_final), normal_b=normality_check(ppo_final), extra={'median_ppo': float(np.median(ppo_final)), 'median_neat': float(np.median(neat_final))})

def test_h2_sample_efficiency(ppo_auc: ArrayLike, neat_auc: ArrayLike, alpha_corrected: float=config.ALPHA_CORRECTED) -> HypothesisResult:
    ppo_arr = np.asarray(ppo_auc, dtype=np.float64)
    neat_arr = np.asarray(neat_auc, dtype=np.float64)
    (w_stat, p_value) = wilcoxon_paired(ppo_arr, neat_arr, alternative='greater')
    deltas = ppo_arr - neat_arr
    parametric_p = float(stats.ttest_rel(ppo_arr, neat_arr, alternative='greater').pvalue)
    return HypothesisResult(hypothesis_id='H2', test_name='Wilcoxon signed-rank (paired, one-sided PPO>NEAT)', statistic=w_stat, p_value=p_value, effect_size=matched_pairs_rank_biserial(deltas), effect_size_name='matched-pairs rank-biserial', alpha=alpha_corrected, reject_null=bonferroni_reject(p_value, alpha_corrected), n_a=ppo_arr.size, n_b=None, parametric_p=parametric_p, normal_a=normality_check(deltas), normal_b=None, extra={'median_delta_auc': float(np.median(deltas)), 'mean_ppo_auc': float(ppo_arr.mean()), 'mean_neat_auc': float(neat_arr.mean())})

def test_h3_robustness(ppo_drop: ArrayLike, neat_drop: ArrayLike, alpha_corrected: float=config.ALPHA_CORRECTED) -> HypothesisResult:
    (u_stat, p_value) = mann_whitney(neat_drop, ppo_drop, alternative='less')
    return HypothesisResult(hypothesis_id='H3', test_name='Mann-Whitney U (one-sided, NEAT drop < PPO drop)', statistic=u_stat, p_value=p_value, effect_size=rank_biserial(neat_drop, ppo_drop), effect_size_name='rank-biserial (NEAT vs PPO drop)', alpha=alpha_corrected, reject_null=bonferroni_reject(p_value, alpha_corrected), n_a=len(list(neat_drop)), n_b=len(list(ppo_drop)), parametric_p=float(stats.ttest_ind(neat_drop, ppo_drop, alternative='less').pvalue), normal_a=normality_check(neat_drop), normal_b=normality_check(ppo_drop), extra={'median_ppo_drop': float(np.median(ppo_drop)), 'median_neat_drop': float(np.median(neat_drop))})

def _paired_seed_arrays(fitness_df: pd.DataFrame, method: str, condition: str) -> 'pd.Series':
    sub = fitness_df[(fitness_df['method'] == method) & (fitness_df['condition'] == condition)]
    return sub.set_index('seed')['fitness'].sort_index()

def per_seed_auc(curves_df: pd.DataFrame, method: str, step_budget: int, fitness_lo: float, fitness_hi: float) -> 'pd.Series':
    method_df = curves_df[curves_df['method'] == method]
    aucs: Dict[int, float] = {}
    for (seed, seed_df) in method_df.groupby('seed'):
        seed_df = seed_df.sort_values('step')
        aucs[int(seed)] = lc.normalized_auc(seed_df['step'].to_numpy(), seed_df['fitness'].to_numpy(), step_budget, fitness_lo, fitness_hi)
    return pd.Series(aucs).sort_index()

def run_pre_registered_suite(fitness_df: pd.DataFrame, curves_df: pd.DataFrame, step_budget: int) -> Dict[str, object]:
    ppo_clean = _paired_seed_arrays(fitness_df, 'ppo', 'clean')
    neat_clean = _paired_seed_arrays(fitness_df, 'neat', 'clean')
    ppo_pert = _paired_seed_arrays(fitness_df, 'ppo', 'perturbed')
    neat_pert = _paired_seed_arrays(fitness_df, 'neat', 'perturbed')
    rnd = _paired_seed_arrays(fitness_df, 'random', 'clean')
    heur = _paired_seed_arrays(fitness_df, 'heuristic', 'clean')
    fitness_lo = float(rnd.mean()) if not rnd.empty else 0.0
    fitness_hi = float(max(ppo_clean.max(), neat_clean.max()))
    ppo_auc = per_seed_auc(curves_df, 'ppo', step_budget, fitness_lo, fitness_hi)
    neat_auc = per_seed_auc(curves_df, 'neat', step_budget, fitness_lo, fitness_hi)
    common_seeds = ppo_auc.index.intersection(neat_auc.index)
    h0 = test_h0_baselines(ppo_clean.values, rnd.values, heur.values, 'ppo')
    h0 += test_h0_baselines(neat_clean.values, rnd.values, heur.values, 'neat')
    return {'H0': h0, 'H1': test_h1_final_fitness(ppo_clean.values, neat_clean.values), 'H2': test_h2_sample_efficiency(ppo_auc.loc[common_seeds].values, neat_auc.loc[common_seeds].values), 'H3': test_h3_robustness((ppo_clean - ppo_pert).dropna().values, (neat_clean - neat_pert).dropna().values)}

def results_to_dataframe(results: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ordered: List[HypothesisResult] = list(results['H0'])
    for key in ('H1', 'H2', 'H3'):
        ordered.append(results[key])
    for r in ordered:
        rows.append({'id': r.hypothesis_id, 'test': r.test_name, 'statistic': round(r.statistic, 3), 'p_value': round(r.p_value, 5), 'alpha': round(r.alpha, 4), 'reject_null': r.reject_null, 'effect_size': round(r.effect_size, 3), 'effect_name': r.effect_size_name, 'parametric_p': None if r.parametric_p is None else round(r.parametric_p, 5), 'n_a': r.n_a, 'n_b': r.n_b})
    return pd.DataFrame(rows)
