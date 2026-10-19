"""Figures used in the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import TARGET
from .strategy import StrategySimulator

COLORS = {"SOFT": "#d62728", "MEDIUM": "#e6b800", "HARD": "#4d4d4d"}
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)


def plot_degradation(
    frame: pd.DataFrame,
    curves: pd.DataFrame,
    path: str | Path,
    min_laps: int = 40,
) -> Path:
    """Observed binned deltas and the fitted curve, per compound."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for axis, compound in zip(axes, ["SOFT", "MEDIUM", "HARD"], strict=False):
        subset = frame[frame["compound"] == compound]
        if len(subset) >= min_laps:
            bins = pd.cut(subset["tyre_age"], bins=[-1, 2, 5, 8, 12, 16, 20, 25, 30, 40])
            grouped = subset.groupby(bins, observed=True)[TARGET].agg(["mean", "count", "sem"])
            centres = [interval.mid for interval in grouped.index]
            axis.errorbar(
                centres,
                grouped["mean"],
                yerr=grouped["sem"].fillna(0),
                fmt="o",
                ms=3.5,
                color=COLORS[compound],
                alpha=0.85,
                label="observed",
            )
        curve = curves[curves["compound"] == compound]
        axis.plot(curve["tyre_age"], curve["predicted_delta"], color=COLORS[compound], lw=1.6, label="model")
        axis.set_title(compound.title())
        axis.set_xlabel("Tyre age (laps)")
        axis.axhline(0, color="black", lw=0.7, alpha=0.4)
    axes[0].set_ylabel("Fuel corrected stint delta (s)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Tyre degradation: observed running vs fitted model", y=1.02)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_strategy_map(
    simulator: StrategySimulator,
    current_lap: int,
    compound: str,
    age: int,
    path: str | Path,
    horizon_laps: int = 45,
) -> Path:
    """Cost of every one stop plan, as a gap to the best plan."""
    laps = np.arange(current_lap + 5, min(simulator.context.total_laps - 3, current_lap + horizon_laps))
    compounds = ["SOFT", "MEDIUM", "HARD"]
    grid = np.full((len(compounds), len(laps)), np.nan)
    for i, new_compound in enumerate(compounds):
        for j, lap in enumerate(laps):
            if new_compound == compound:
                continue
            plan = ((int(lap), new_compound),)
            grid[i, j] = simulator.plan_cost(current_lap, compound, age, plan)

    best = np.nanmin(grid)
    fig, axis = plt.subplots(figsize=(8.5, 3.2))
    mesh = axis.pcolormesh(laps, np.arange(len(compounds) + 1), grid - best, cmap="viridis_r", shading="flat")
    axis.set_yticks(np.arange(len(compounds)) + 0.5)
    axis.set_yticklabels([name.title() for name in compounds])
    axis.set_xlabel("Pit lap")
    axis.set_ylabel("New compound")
    axis.set_title(f"One stop plan cost, lap {current_lap}, {compound.title()} age {age}")
    axis.grid(False)
    bar = fig.colorbar(mesh, ax=axis)
    bar.set_label("Seconds behind best one stop")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_plan(
    simulator: StrategySimulator,
    plans: list,
    current_lap: int,
    compound: str,
    age: int,
    path: str | Path,
    limit: int = 3,
) -> Path:
    """Cumulative cost of the leading plans over the remaining race."""
    fig, axis = plt.subplots(figsize=(8.5, 3.6))
    for plan in plans[:limit]:
        replay = simulator.replay(current_lap, compound, age, plan.stops)
        replay["cumulative"] = replay["delta"].cumsum() + replay.get("pit_loss", 0).fillna(0).cumsum()
        label = plan.describe()
        axis.plot(replay["lap"], replay["cumulative"], lw=1.7, label=f"{label} ({plan.cost:.1f}s)")
        for stop_lap, _ in plan.stops:
            axis.axvline(stop_lap, color="grey", lw=0.6, alpha=0.5)
    axis.set_xlabel("Lap")
    axis.set_ylabel("Cumulative delta plus pit loss (s)")
    axis.set_title("Leading plans over the remaining race")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_regret(frame: pd.DataFrame, path: str | Path) -> Path:
    """Distribution of time left on the table, split by the number of stops made."""
    valid = frame[frame["valid_actual"]].copy()
    valid["regret"] = valid["actual_cost"] - valid["best_cost"]
    groups = [
        valid.loc[valid["actual_stops"] == 1, "regret"],
        valid.loc[valid["actual_stops"] == 2, "regret"],
    ]
    labels = ["One stop", "Two stops"]
    fig, axis = plt.subplots(figsize=(6.2, 3.4))
    data = [values.dropna().to_numpy() for values in groups if values.notna().any()]
    used_labels = [label for label, values in zip(labels, groups, strict=False) if values.notna().any()]
    parts = axis.boxplot(data, patch_artist=True, widths=0.45)
    for patch in parts["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.8)
    axis.set_xticks(range(1, len(used_labels) + 1))
    axis.set_xticklabels(used_labels)
    axis.set_ylabel("Time behind best plan (s)")
    axis.set_title("Strategy regret at the decision lap")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_predictions(truth: np.ndarray, predicted: np.ndarray, path: str | Path, title: str) -> Path:
    """Predicted against actual lap delta on held out races."""
    fig, axis = plt.subplots(figsize=(4.6, 4.2))
    axis.hexbin(truth, predicted, gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
    low = float(min(truth.min(), predicted.min()))
    high = float(max(truth.max(), predicted.max()))
    axis.plot([low, high], [low, high], color="#d62728", lw=1, label="perfect prediction")
    axis.set_xlabel("Observed lap delta (s)")
    axis.set_ylabel("Predicted lap delta (s)")
    axis.set_title(title)
    axis.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
