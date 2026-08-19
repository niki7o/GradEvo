"""All figure generation for the notebook.

Each function takes tidy DataFrames (as written by
:func:`gradevo.experiment.runner.save_results`) and returns a Matplotlib
``Figure``. Every figure sets axis labels with units, a title, and, where a
reference threshold exists, a horizontal "solved" line. The notebook adds the
one-sentence captions and any "do not conclude X" warnings in markdown.

These functions never mutate global Matplotlib state beyond creating figures,
and they do not call ``plt.show`` -- the notebook controls display.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from gradevo import config
from gradevo.metrics import learning_curves as lc

_METHOD_COLORS: Dict[str, str] = {
    "ppo": "#1f77b4",
    "neat": "#d62728",
    "random": "#7f7f7f",
    "heuristic": "#2ca02c",
}


def _method_color(method: str) -> str:
    """Return a stable color for a method label."""
    return _METHOD_COLORS.get(method, "#333333")


def plot_learning_curves(
    curves_df: pd.DataFrame,
    step_budget: int,
    solved_threshold: Optional[float] = None,
    n_points: int = 100,
) -> Figure:
    """Overlay PPO and NEAT learning curves with seed-variance bands.

    Args:
        curves_df: Long-form curve table with columns
            ``[method, seed, step, fitness]``.
        step_budget: Step budget defining the shared x-grid extent.
        solved_threshold: Optional "solved" return to draw as a reference line.
        n_points: Grid resolution for curve alignment.

    Returns:
        The Matplotlib figure. Mean line with +/-1 SD shaded band per method.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    grid = lc.common_grid(step_budget, n_points)
    for method in config.METHODS:
        method_df = curves_df[curves_df["method"] == method]
        if method_df.empty:
            continue
        aligned = _align_seed_curves(method_df, grid)
        mean, lower, upper = lc.seed_variance_band(aligned)
        color = _method_color(method)
        ax.plot(grid, mean, label=method.upper(), color=color, linewidth=2)
        ax.fill_between(grid, lower, upper, color=color, alpha=0.2)
    if solved_threshold is not None:
        ax.axhline(solved_threshold, color="black", linestyle="--", linewidth=1, label="solved")
    ax.set_xlabel("Environment interaction steps (env.step calls)")
    ax.set_ylabel("Mean episodic return (fitness)")
    ax.set_title("Learning curves: fitness vs. environment steps (mean +/- 1 SD across seeds)")
    ax.legend()
    fig.tight_layout()
    return fig


def _align_seed_curves(method_df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Interpolate every seed's curve onto the shared grid -> (n_seeds, n_pts)."""
    rows: List[np.ndarray] = []
    for _, seed_df in method_df.groupby("seed"):
        seed_df = seed_df.sort_values("step")
        rows.append(lc.align_curves(seed_df["step"].to_numpy(), seed_df["fitness"].to_numpy(), grid))
    return np.vstack(rows)


def plot_final_fitness_distributions(fitness_df: pd.DataFrame) -> Figure:
    """Violin+box plot of final fitness per method, clean vs. perturbed.

    Args:
        fitness_df: Long-form fitness table with columns
            ``[method, condition, seed, fitness]``.

    Returns:
        The Matplotlib figure comparing distributions across methods and
        conditions for the two trained methods.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = [m for m in config.METHODS]
    positions, data, labels, colors = [], [], [], []
    pos = 0
    for method in methods:
        for cond in config.CONDITIONS:
            vals = fitness_df[(fitness_df["method"] == method) & (fitness_df["condition"] == cond)]["fitness"].to_numpy()
            if vals.size == 0:
                continue
            positions.append(pos)
            data.append(vals)
            labels.append(f"{method.upper()}\n{cond}")
            colors.append(_method_color(method))
            pos += 1
    parts = ax.violinplot(data, positions=positions, showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.3)
    ax.boxplot(data, positions=positions, widths=0.25, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean episodic return (fitness)")
    ax.set_title("Final fitness distributions by method and condition")
    fig.tight_layout()
    return fig


def plot_robustness_drop(fitness_df: pd.DataFrame) -> Figure:
    """Bar chart of performance drop (clean - perturbed) per method with scatter.

    Args:
        fitness_df: Long-form fitness table.

    Returns:
        The Matplotlib figure. Bars show mean drop with a 95% CI error bar and
        per-seed scatter overlaid, so the reader sees the raw spread.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, method in enumerate(config.METHODS):
        drops = _per_seed_drop(fitness_df, method)
        if drops.size == 0:
            continue
        mean = float(drops.mean())
        ci = 1.96 * drops.std(ddof=1) / np.sqrt(drops.size) if drops.size > 1 else 0.0
        ax.bar(i, mean, yerr=ci, capsize=6, color=_method_color(method), alpha=0.6, label=method.upper())
        ax.scatter(np.full(drops.size, i) + np.random.default_rng(0).uniform(-0.08, 0.08, drops.size), drops, color="black", s=18, zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(config.METHODS)))
    ax.set_xticklabels([m.upper() for m in config.METHODS])
    ax.set_ylabel("Fitness drop under perturbation (clean - perturbed)")
    ax.set_title("Robustness: performance drop under held-out perturbation (mean, 95% CI, per-seed points)")
    fig.tight_layout()
    return fig


def _per_seed_drop(fitness_df: pd.DataFrame, method: str) -> np.ndarray:
    """Compute per-seed (clean - perturbed) fitness drop for a method."""
    clean = fitness_df[(fitness_df["method"] == method) & (fitness_df["condition"] == "clean")].set_index("seed")["fitness"]
    pert = fitness_df[(fitness_df["method"] == method) & (fitness_df["condition"] == "perturbed")].set_index("seed")["fitness"]
    joined = clean.subtract(pert, fill_value=np.nan).dropna()
    return joined.to_numpy()


def plot_budget_table(budget_df: pd.DataFrame) -> Figure:
    """Render the realized-vs-requested budget table as a figure.

    Args:
        budget_df: Budget table from
            :func:`gradevo.experiment.budget.budget_table`.

    Returns:
        A Matplotlib figure containing the table (for embedding in the
        notebook alongside the prose discussion).
    """
    fig, ax = plt.subplots(figsize=(10, 1.6 + 0.4 * len(budget_df)))
    ax.axis("off")
    table = ax.table(cellText=budget_df.values, colLabels=budget_df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Compute budget: requested vs. realized environment steps", pad=12)
    fig.tight_layout()
    return fig


def plot_neat_topology(genome: object, neat_config: object) -> Figure:
    """Draw a NEAT winner genome's network topology with networkx.

    Args:
        genome: A neat-python genome with ``nodes`` and ``connections``.
        neat_config: The neat-python Config (for input/output key metadata).

    Returns:
        A Matplotlib figure showing enabled connections between nodes, colored
        by node role (input / output / hidden).
    """
    import networkx as nx

    graph = nx.DiGraph()
    input_keys = list(neat_config.genome_config.input_keys)
    output_keys = list(neat_config.genome_config.output_keys)
    for node_key in list(genome.nodes) + input_keys:
        graph.add_node(node_key)
    for conn_key, conn in genome.connections.items():
        if conn.enabled:
            src, dst = conn_key
            graph.add_edge(src, dst, weight=conn.weight)

    color_map = ["#1f77b4" if n in input_keys else "#d62728" if n in output_keys else "#999999" for n in graph.nodes]
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(graph, seed=0)
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=500, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    ax.set_title("Representative NEAT winner genome topology (blue=input, red=output, grey=hidden)")
    ax.axis("off")
    fig.tight_layout()
    return fig
