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


# ---------------------------------------------------------------- daisy chain

def _mean_std(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    return df.groupby(by, sort=False)[col].agg(["mean", "std"]).reset_index()


def plot_accuracy_by_generation(metrics: pd.DataFrame, models: list[dict], path: Path):
    """Held-out accuracy vs generation for the chain, the GT upper bounds, direct students and zero-shot."""
    _style()
    names = [m["name"] for m in models]
    gen = {n: i + 1 for i, n in enumerate(names)}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    m1 = metrics[metrics.condition == "seed_gt"]
    up = pd.concat([m1, metrics[metrics.condition == "gt_upper"]])
    if len(up):
        u = _mean_std(up, ["model"], "test_acc")
        u["g"] = u.model.map(gen)
        u = u.sort_values("g")
        ax.errorbar(u.g, u["mean"], yerr=u["std"].fillna(0), color=CEILING, marker="o", capsize=3,
                    label="GT-trained (upper bound)", zorder=3)
    ch = pd.concat([m1, metrics[metrics.condition == "chain"]])
    if len(ch):
        c = _mean_std(ch, ["model"], "test_acc")
        c["g"] = c.model.map(gen)
        c = c.sort_values("g")
        ax.errorbar(c.g, c["mean"], yerr=c["std"].fillna(0), color=CATEGORICAL[0], marker="o", capsize=3,
                    label="daisy chain (GT → M1 → M2 → …)", zorder=4)
    d = metrics[metrics.condition == "direct"]
    for idx, (t, sub) in enumerate(d.groupby("teacher", sort=False)):
        s = _mean_std(sub, ["model"], "test_acc")
        s["g"] = s.model.map(gen)
        ax.errorbar(s.g + 0.06 * (idx + 1), s["mean"], yerr=s["std"].fillna(0), color=CATEGORICAL[1 + idx],
                    marker="D", linestyle="none", capsize=3, markersize=8, label=f"direct from {t}", zorder=5)
    z = metrics[metrics.condition == "zero_shot"]
    if len(z):
        zz = z.assign(g=z.model.map(gen)).sort_values("g")
        ax.plot(zz.g, zz.test_acc, color=MUTED, marker="x", linestyle=":", label="zero-shot (no training)")
    if len(m1):
        ax.axhline(m1.test_acc.mean(), color=CATEGORICAL[0], linewidth=1, linestyle="--", alpha=0.6)
        ax.text(len(names) + 0.35, m1.test_acc.mean(), "M1", color=INK_2, va="center", fontsize=8)
    ax.set_xticks(range(1, len(names) + 1), [f"M{i}\n{n}" for i, n in enumerate(names, 1)])
    ax.set_xlim(0.7, len(names) + 0.5)
    ax.set_xlabel("generation / model")
    ax.set_ylabel("held-out test accuracy")
    ax.set_title("Accuracy by supervision generation", loc="left", pad=18)
    ax.text(0, 1.01, "error bars = std over seeds; dashed = M1 (the original weak supervisor)",
            transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    _save(fig, path)


def plot_teacher_vs_student(metrics: pd.DataFrame, path: Path):
    """Dumbbell: each supervision step's teacher accuracy -> student accuracy (and the student's GT upper bound)."""
    _style()
    st = metrics[metrics.condition.isin(["chain", "direct"])].copy()
    st["step"] = st.condition + ": " + st.teacher + " → " + st.model
    cols = ["teacher_test_acc", "test_acc"] + (["upper_acc"] if "upper_acc" in st else [])
    g = st.groupby("step", sort=False)[cols].mean().reset_index()
    order = st.drop_duplicates("step").sort_values(["condition", "generation", "teacher"]).step.tolist()
    g = g.set_index("step").loc[order[::-1]].reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(g) + 1.6))
    y = np.arange(len(g))
    for i, r in g.iterrows():
        better = r.test_acc >= r.teacher_test_acc
        ax.plot([r.teacher_test_acc, r.test_acc], [i, i], color=CATEGORICAL[0] if better else CATEGORICAL[7],
                linewidth=2, zorder=2)
    ax.scatter(g.teacher_test_acc, y, color=MUTED, s=64, zorder=3, label="teacher (supervisor)",
               edgecolors=SURFACE, linewidths=2)
    ax.scatter(g.test_acc, y, color=CATEGORICAL[0], s=64, zorder=4, label="student", edgecolors=SURFACE, linewidths=2)
    if "upper_acc" in g:
        ax.scatter(g.upper_acc, y, color=CEILING, marker="|", s=220, zorder=4, label="student GT upper bound")
    ax.set_yticks(y, g.step)
    ax.set_xlabel("held-out test accuracy (mean over seeds)")
    ax.set_title("Does each student beat its supervisor?", loc="left", pad=18)
    ax.text(0, 1.01, "blue segment = student beats teacher; red = student worse", transform=ax.transAxes,
            fontsize=8, color=INK_2, va="bottom")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_chain_vs_direct(metrics: pd.DataFrame, names: list[str], path: Path):
    """Final model: chain-MN vs every direct-to-MN student vs GT-MN, with M1 as the reference."""
    _style()
    last = names[-1]
    rows = []

    def add(label, df, color):
        if len(df):
            rows.append((label, df.test_acc.mean(), df.test_acc.std(), df.test_acc.to_numpy(), color))

    add(f"M1 {names[0]} (weak supervisor)", metrics[metrics.condition == "seed_gt"], MUTED)
    for idx, (t, sub) in enumerate(metrics[(metrics.condition == "direct") & (metrics.model == last)]
                                   .groupby("teacher", sort=False)):
        add(f"direct {t} → {last}", sub, CATEGORICAL[1 + idx])
    add(f"daisy chain → {last}", metrics[(metrics.condition == "chain") & (metrics.model == last)], CATEGORICAL[0])
    add(f"GT-trained {last} (upper bound)", metrics[(metrics.condition == "gt_upper") & (metrics.model == last)], CEILING)
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(rows) + 1.6))
    for i, (label, mu, sd, vals, c) in enumerate(rows):
        ax.errorbar(mu, i, xerr=0 if not np.isfinite(sd) else sd, color=c, marker="o", markersize=9, capsize=4)
        ax.scatter(vals, np.full(len(vals), i) + 0.18, color=c, s=14, alpha=0.6)
        ax.text(mu, i - 0.3, f"{mu:.3f}", ha="center", va="top", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_xlabel("held-out test accuracy")
    ax.set_title(f"Chain vs direct supervision of {last}", loc="left", pad=18)
    ax.text(0, 1.01, "big dot = mean ± std over seeds; small dots = individual seeds", transform=ax.transAxes,
            fontsize=8, color=INK_2, va="bottom")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_error_transitions(trans: pd.DataFrame, path: Path):
    """Per consecutive chain step: corrected (W->C), newly introduced (C->W), and inherited (W->W) errors."""
    _style()
    t = trans.groupby("transition", sort=False)[["W->C", "C->W", "W->W"]].agg(["mean", "std"])
    labels = list(t.index)
    x = np.arange(len(labels))
    kinds = [("W->C", "corrected (wrong → correct)", CATEGORICAL[0]),
             ("C->W", "new error (correct → wrong)", CATEGORICAL[7]),
             ("W->W", "inherited (wrong → wrong)", MUTED)]
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (k, lab, c) in enumerate(kinds):
        ax.bar(x + (i - 1) * w, t[(k, "mean")], w - 0.02, yerr=t[(k, "std")].fillna(0), color=c, label=lab,
               capsize=3, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [f"{lab}" for lab in labels])
    ax.set_ylabel("fraction of test examples")
    ax.set_title("Error inheritance and correction along the chain", loc="left", pad=18)
    ax.text(0, 1.01, "remaining mass is correct → correct; error bars = std over seeds", transform=ax.transAxes,
            fontsize=8, color=INK_2, va="bottom")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def plot_error_correlation(corr: pd.DataFrame, names: list[str], path: Path):
    """Phi between error indicators: chain generations vs independently GT-trained models of the same sizes."""
    _style()
    fams = [f for f in ("chain", "gt_upper") if f in set(corr.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams) + 1.2, 3.9), squeeze=False)
    n = len(names)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    for ax, fam in zip(axes[0], fams):
        Mx = np.full((n, n), np.nan)
        g = corr[corr.family == fam].groupby(["gen_a", "gen_b"]).phi.mean()
        for (a, b), v in g.items():
            Mx[a - 1, b - 1] = Mx[b - 1, a - 1] = v
        ax.grid(False)
        im = ax.imshow(Mx, cmap=cmap, vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                if np.isfinite(Mx[i, j]):
                    ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=9,
                            color=SURFACE if Mx[i, j] > 0.6 else INK)
        ax.set_xticks(range(n), names, rotation=0)
        ax.set_yticks(range(n), names)
        ax.set_title("chain generations" if fam == "chain" else "GT-trained (independent)", loc="left")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04, label="φ (error correlation)")
    fig.suptitle("Are errors increasingly correlated along the chain?", x=0.02, y=1.06, ha="left",
                 fontweight="bold", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
