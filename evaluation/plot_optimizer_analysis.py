# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Create publication-ready figures from the final optimizer analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METHODS = ("random", "tpe", "cmaes")
METHOD_LABELS = {"random": "Random", "tpe": "TPE", "cmaes": "CMA-ES"}
METHOD_COLORS = {"random": "#0072B2", "tpe": "#E69F00", "cmaes": "#009E73"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True, help="Optimizer analysis JSON")
    parser.add_argument("--robustness", type=Path, help="Optional repeated-run summary JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Figure output directory")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: Any, output_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


def plot_improvement_heatmap(analysis: dict[str, Any], output_dir: Path) -> None:
    """Plot representative-video objective improvement relative to MCP."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    comparisons = analysis.get("comparisons", [])
    values = np.full((len(comparisons), len(METHODS)), np.nan)
    for row_index, comparison in enumerate(comparisons):
        for column_index, method in enumerate(METHODS):
            improvement = comparison.get("methods", {}).get(method, {}).get("representative_improvement_fraction")
            if improvement is not None:
                values[row_index, column_index] = 100.0 * float(improvement)

    fig_height = max(3.0, 0.42 * len(comparisons) + 1.2)
    fig, ax = plt.subplots(figsize=(5.6, fig_height), constrained_layout=True)
    finite = values[np.isfinite(values)]
    extent = max(10.0, float(np.max(np.abs(finite))) if finite.size else 10.0)
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent),
    )
    ax.set_xticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    ax.set_yticks(range(len(comparisons)), [row["case"] for row in comparisons])
    ax.set_xlabel("Optimization method")
    ax.set_ylabel("Benchmark case")
    ax.set_title("Representative-video objective improvement over MCP")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = "n/a" if np.isnan(value) else f"{value:+.1f}%"
            color = "white" if np.isfinite(value) and abs(value) > 0.55 * extent else "black"
            ax.text(column_index, row_index, label, ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Improvement (%)")
    _save(fig, output_dir, "optimizer_improvement_heatmap")


def _robustness_cases(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    cases = payload.get("cases")
    if isinstance(cases, list):
        return cases
    if isinstance(payload.get("comparisons"), list):
        return payload["comparisons"]
    return []


def plot_robustness(robustness: dict[str, Any], output_dir: Path) -> None:
    """Plot repeated-run objectives normalized to each case's MCP mean."""
    import matplotlib.pyplot as plt

    cases = _robustness_cases(robustness)
    if not cases:
        return
    labels = [str(case.get("case")) for case in cases]
    x = np.arange(len(cases), dtype=float)
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(6.2, len(cases) * 1.55), 3.8), constrained_layout=True)
    displayed = ("mcp", *METHODS)
    colors = {"mcp": "#6B7280", **METHOD_COLORS}
    for method_index, method in enumerate(displayed):
        means: list[float] = []
        errors: list[float] = []
        for case in cases:
            summaries = case.get("methods", case.get("summaries", {}))
            baseline = summaries.get("mcp", {})
            current = summaries.get(method, {})
            baseline_mean = baseline.get("raw_objective_mean")
            current_mean = current.get("raw_objective_mean")
            current_std = current.get("raw_objective_std", 0.0)
            if baseline_mean in {None, 0.0} or current_mean is None:
                means.append(np.nan)
                errors.append(0.0)
            else:
                means.append(float(current_mean) / float(baseline_mean))
                errors.append(float(current_std or 0.0) / float(baseline_mean))
        offset = (method_index - 1.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=2,
            label="MCP" if method == "mcp" else METHOD_LABELS[method],
            color=colors[method],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.axhline(1.0, color="#4B5563", linestyle="--", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Normalized raw objective (MCP = 1)")
    ax.set_title("Repeated full-fidelity replay (mean ± SD)")
    ax.legend(ncols=4, frameon=False, loc="upper center")
    ax.set_ylim(bottom=0.0)
    _save(fig, output_dir, "robustness_normalized_objective")


def _total_tokens(usage: dict[str, Any]) -> float:
    return float(usage.get("input_tokens", 0) or 0) + float(usage.get("output_tokens", 0) or 0)


def plot_agent_efficiency(analysis: dict[str, Any], output_dir: Path) -> None:
    """Plot paired generation time and token usage for Direct-Newton and MCP."""
    import matplotlib.pyplot as plt

    comparisons = analysis.get("comparisons", [])
    cases: list[str] = []
    times: list[tuple[float, float]] = []
    tokens: list[tuple[float, float]] = []
    for comparison in comparisons:
        methods = comparison.get("generation_methods", {})
        direct = methods.get("direct-newton", {})
        mcp = methods.get("mcp", {})
        direct_time, mcp_time = direct.get("elapsed_seconds"), mcp.get("elapsed_seconds")
        direct_tokens, mcp_tokens = _total_tokens(direct.get("token_usage", {})), _total_tokens(
            mcp.get("token_usage", {})
        )
        if None in {direct_time, mcp_time} or min(direct_tokens, mcp_tokens) <= 0:
            continue
        cases.append(str(comparison["case"]))
        times.append((float(direct_time), float(mcp_time)))
        tokens.append((direct_tokens, mcp_tokens))

    if not cases:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(cases)))
    for index, (case, time_pair, token_pair) in enumerate(zip(cases, times, tokens, strict=True)):
        axes[0].plot((0, 1), time_pair, marker="o", color=colors[index], linewidth=1.0, label=case)
        axes[1].plot((0, 1), token_pair, marker="o", color=colors[index], linewidth=1.0)
    for ax, ylabel, title in (
        (axes[0], "Elapsed time (s)", "Agent generation time"),
        (axes[1], "Input + output tokens", "Agent token usage"),
    ):
        ax.set_xticks((0, 1), ("Direct-Newton", "MCP"))
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=6,
        ncols=min(5, len(legend_labels)),
        loc="outside lower center",
    )
    _save(fig, output_dir, "agent_generation_efficiency")


def main() -> None:
    args = _parse_args()
    _configure_style()
    analysis = _read_json(args.analysis)
    plot_improvement_heatmap(analysis, args.output_dir)
    plot_agent_efficiency(analysis, args.output_dir)
    if args.robustness is not None and args.robustness.is_file():
        plot_robustness(_read_json(args.robustness), args.output_dir)


if __name__ == "__main__":
    main()
