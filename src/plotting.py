"""Matplotlib figures. Static PNGs for the paper-style report.

Color roles: categorical slots in fixed order for identity, a single-hue blue ramp for ordered
series (weak-model size), blue<->red diverging with a gray midpoint for signed PGR.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
CEILING = "#0b0b0b"
# negative (student worse than its supervisor) = red, positive = blue
DIVERGING = LinearSegmentedColormap.from_list("rgb", ["#c0302f", "#ef9a99", "#f0efec", "#86b6ef", "#1c5cab"])


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
    })


def ordinal_colors(n: int) -> list[str]:
    if n <= len(BLUE_ORDINAL):
        idx = np.linspace(0, len(BLUE_ORDINAL) - 1, n).round().astype(int) if n > 1 else [2]
        return [BLUE_ORDINAL[i] for i in idx]
    return [plt.cm.Blues(x) for x in np.linspace(0.35, 0.95, n)]


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _xaxis(ax, xs: dict[str, float], log: bool):
    if log:
        ax.set_xscale("log")
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(list(xs.keys()))
    ax.minorticks_off()


def model_x(models: list[dict]) -> tuple[dict[str, float], bool]:
    if all("params" in m for m in models):
        return {m["name"]: float(m["params"]) for m in models}, True
    return {m["name"]: float(i) for i, m in enumerate(models)}, False


# ---------------------------------------------------------------- base W2S

def plot_w2s_accuracy(summary: pd.DataFrame, models: list[dict], path: Path):
    """Test accuracy vs strong-model size: ceiling (GT) line + one W2S line per weak supervisor."""
    _style()
    xs, log = model_x(models)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ceil = summary[summary.condition == "ceiling_gt"].set_index("model")
    order = [m["name"] for m in models if m["name"] in ceil.index]
    ax.errorbar([xs[m] for m in order], ceil.loc[order, "test_acc_mean"], yerr=ceil.loc[order, "test_acc_std"].fillna(0),
                color=CEILING, marker="o", label="strong ceiling (GT labels)", capsize=3, zorder=3)
    teach = summary[summary.condition == "teacher_gt"].set_index("model")
    w2s = summary[summary.condition == "w2s"]
    weak_names = [m["name"] for m in models if m["name"] in set(w2s.teacher)]
    for c, w in zip(ordinal_colors(len(weak_names)), weak_names):
        sub = w2s[w2s.teacher == w].set_index("model")
        strong = [m["name"] for m in models if m["name"] in sub.index]
        x = [xs[w]] + [xs[s] for s in strong]
        y = [teach.loc[w, "test_acc_mean"]] + list(sub.loc[strong, "test_acc_mean"])
        e = [teach.loc[w, "test_acc_std"]] + list(sub.loc[strong, "test_acc_std"])
        ax.errorbar(x, y, yerr=np.nan_to_num(e), color=c, marker="o", capsize=3, label=f"weak = {w}")
        ax.plot([xs[w]], [y[0]], marker="s", color=c, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2,
                linestyle="none", zorder=4)
    _xaxis(ax, xs, log)
    ax.set_xlabel("strong student model")
    ax.set_ylabel("held-out test accuracy")
    ax.set_title("Student accuracy by weak supervisor", loc="left", pad=18)
    ax.text(0, 1.01, "square = the weak supervisor itself; error bars = std over seeds", transform=ax.transAxes,
            fontsize=8, color=INK_2, va="bottom")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    _save(fig, path)


def plot_pgr_grid(summary: pd.DataFrame, models: list[dict], path: Path):
    _style()
    w2s = summary[summary.condition == "w2s"]
    names = [m["name"] for m in models]
    weak = [n for n in names if n in set(w2s.teacher)]
    strong = [n for n in names if n in set(w2s.model)]
    M = np.full((len(weak), len(strong)), np.nan)
    S = np.full_like(M, np.nan)
    for _, r in w2s.iterrows():
        M[weak.index(r.teacher), strong.index(r.model)] = r.pgr_mean
        S[weak.index(r.teacher), strong.index(r.model)] = r.pgr_std
    fig, ax = plt.subplots(figsize=(1.6 * len(strong) + 2.5, 1.1 * len(weak) + 1.8))
    ax.grid(False)
    finite = M[np.isfinite(M)]
    lim = max(1.0, float(np.abs(finite).max())) if finite.size else 1.0
    im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    for i in range(len(weak)):
        for j in range(len(strong)):
            txt = "n/a" if not np.isfinite(M[i, j]) else f"{M[i, j]:.2f}" + (f"\n±{S[i, j]:.2f}" if np.isfinite(S[i, j]) else "")
            dark = np.isfinite(M[i, j]) and abs(M[i, j]) > 0.6 * lim
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=SURFACE if dark else INK)
    ax.set_xticks(range(len(strong)), strong)
    ax.set_yticks(range(len(weak)), weak)
    ax.set_xlabel("strong student")
    ax.set_ylabel("weak supervisor")
    ax.set_title("PGR (mean ± std over seeds)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="PGR")
    _save(fig, path)


def plot_confidence_bins(bins: pd.DataFrame, path: Path, group_col: str, title: str):
    """Student error-correction rate on teacher errors, by teacher confidence bin."""
    _style()
    groups = list(dict.fromkeys(bins[group_col]))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = CATEGORICAL if len(groups) <= len(CATEGORICAL) else ordinal_colors(len(groups))
    for c, g in zip(colors, groups):
        sub = bins[bins[group_col] == g].sort_values("bin")
        mid = (sub.conf_low + sub.conf_high) / 2
        ax.plot(mid, sub.error_correction, marker="o", color=c, label=g)
    ax.set_xlabel("teacher confidence (max probability)")
    ax.set_ylabel("fraction of teacher errors corrected")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _save(fig, path)
