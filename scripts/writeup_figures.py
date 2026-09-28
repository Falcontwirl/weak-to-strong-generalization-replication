"""Write-up figures that combine several runs. Run from the repo root:

    python scripts/writeup_figures.py

Reads outputs/<run>/metrics.csv (and error_correlation.csv, test predictions) and writes PNGs to
docs/figures/. Colors reuse src/plotting.py: student (W2S) = categorical slot 1 (blue), teacher =
slot 2 (orange), ground-truth-trained models = ink. Values are means over 3 seeds; error bars are
the standard deviation across seeds.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plotting import CATEGORICAL, DIVERGING, GRID, INK, INK_2, MUTED, SURFACE, _style  # noqa: E402

OUT = ROOT / "outputs"
FIG = ROOT / "docs" / "figures"
RUNS = {
    "E1": "base_full_48gb-1b976761",     # SST-2 naive, 160M-1.4B
    "E2": "chain_full_48gb-6faef36c",    # SST-2 daisy chain
    "E3": "base_full_48gb_conf-3f207cb5",  # SST-2 conf loss, alpha 0.75 for all students, linear 20% warmup
    "E4": "base_full_2p8b-3c122e41",     # SST-2 naive, 160M-2.8B
    "E6": "boolq_base-cb7d7926",         # BoolQ naive, 160M-1.4B
    "E9": "base_full_2p8b_conf_ref-c0acc09f",  # SST-2 conf loss, released-code settings
    "E10": "base_full_48gb_conf_a0-0563a52b",  # SST-2 conf loss with alpha 0 (sanity check)
}
STUDENT, TEACHER, GT = CATEGORICAL[0], CATEGORICAL[1], INK
SIZE = {"p70m": 7.0e7, "p160m": 1.6e8, "p410m": 4.1e8, "p1b": 1.0e9, "p1.4b": 1.4e9, "p2.8b": 2.8e9}
LABEL = {"p160m": "160M", "p410m": "410M", "p1b": "1B", "p1.4b": "1.4B", "p2.8b": "2.8B"}
ORDER = list(LABEL)


def metrics(run: str) -> pd.DataFrame:
    return pd.read_csv(OUT / RUNS[run] / "metrics.csv").fillna({"teacher": ""})


def ms(df: pd.DataFrame, col: str = "test_acc") -> tuple[float, float]:
    return float(df[col].mean()), float(df[col].std(ddof=1)) if len(df) > 1 else 0.0


def majority_rate(run: str) -> float:
    """Accuracy of always predicting the majority class on that run's test set."""
    f = sorted(glob.glob(str(OUT / RUNS[run] / "jobs" / "*" / "test_preds.parquet")))[0]
    y = pd.read_parquet(f)["label"].to_numpy()
    p = y.mean()
    return float(max(p, 1 - p))


def save(fig, name: str):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def pair_rows(m: pd.DataFrame) -> pd.DataFrame:
    """One row per (teacher, student) pair: teacher / W2S / ceiling accuracy and PGR, mean and sd over seeds."""
    rows = []
    w = m[m.condition == "w2s"]
    for (t, s), g in w.groupby(["teacher", "model"]):
        tm, ts = ms(m[(m.condition == "teacher_gt") & (m.model == t)])
        cm, cs = ms(m[(m.condition == "ceiling_gt") & (m.model == s)])
        wm, ws = ms(g)
        pm, psd = ms(g[g.pgr_valid.astype(bool)], "pgr")
        rows.append(dict(teacher=t, student=s, t=tm, t_sd=ts, w=wm, w_sd=ws, c=cm, c_sd=cs, pgr=pm, pgr_sd=psd))
    df = pd.DataFrame(rows)
    df["ti"], df["si"] = df.teacher.map(ORDER.index), df.student.map(ORDER.index)
    return df.sort_values(["ti", "si"]).reset_index(drop=True)


# 1 ---------------------------------------------------------------- what PGR measures
def fig_pgr_setup(pairs: pd.DataFrame):
    r = pairs[(pairs.teacher == "p160m") & (pairs.student == "p2.8b")].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    y = 0
    ax.plot([r.t, r.w], [y, y], color=STUDENT, lw=10, solid_capstyle="butt", alpha=0.85, zorder=1)
    ax.plot([r.w, r.c], [y, y], color=GRID, lw=10, solid_capstyle="butt", zorder=1)
    for x, col, name in [(r.t, TEACHER, "Weak teacher\n160M, true labels"),
                         (r.w, STUDENT, "Weak-to-strong student\n2.8B, teacher's labels"),
                         (r.c, GT, "Ceiling\n2.8B, true labels")]:
        ax.scatter([x], [y], s=110, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{name}\n{x:.3f}", (x, y), xytext=(0, 16), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9, color=INK)
    ax.annotate("gap recovered", ((r.t + r.w) / 2, y), xytext=(0, -20), textcoords="offset points",
                ha="center", va="top", fontsize=9, color=INK_2)
    ax.annotate("gap remaining", ((r.w + r.c) / 2, y), xytext=(0, -20), textcoords="offset points",
                ha="center", va="top", fontsize=9, color=INK_2)
    ax.set_title(f"PGR = recovered / (ceiling − teacher) = ({r.w:.3f} − {r.t:.3f}) / ({r.c:.3f} − {r.t:.3f}) "
                 f"= {(r.w - r.t) / (r.c - r.t):.2f}", loc="left")
    ax.text(0, -0.42, f"PGR from the mean accuracies shown; the average of per-seed PGRs is {r.pgr:.2f} ± {r.pgr_sd:.2f}.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="top")
    ax.set_xlim(r.t - 0.02, r.c + 0.02)
    ax.set_ylim(-1, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("SST-2 test accuracy (mean of 3 seeds)")
    save(fig, "fig1_pgr_setup.png")


# 2 ---------------------------------------------------------------- every pair
def fig_pairs(pairs: pd.DataFrame):
    n = len(pairs)
    fig, ax = plt.subplots(figsize=(8.5, 0.48 * n + 1.4))
    ys = np.arange(n)[::-1]
    for y, r in zip(ys, pairs.itertuples()):
        ax.plot([r.t, r.c], [y, y], color=GRID, lw=2, zorder=1)
        for x, sd, col, mk in [(r.t, r.t_sd, TEACHER, "o"), (r.w, r.w_sd, STUDENT, "o"), (r.c, r.c_sd, GT, "D")]:
            ax.errorbar(x, y, xerr=sd, fmt=mk, ms=8 if mk == "o" else 6, color=col, mec=SURFACE, mew=1.5,
                        elinewidth=1.2, capsize=0, zorder=3)
        ax.text(0.945, y, f"{r.pgr:+.2f} ± {r.pgr_sd:.2f}", va="center", ha="left", fontsize=9, color=INK_2)
    ax.text(0.945, n - 0.35, "PGR", ha="left", va="bottom", fontsize=9, color=INK, weight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{LABEL[r.teacher]} → {LABEL[r.student]}" for r in pairs.itertuples()])
    for y, (a, b) in zip(ys[:-1], zip(pairs.teacher[:-1], pairs.teacher[1:])):
        if a != b:
            ax.axhline(y - 0.5, color=MUTED, lw=0.6)
    ax.set_xlim(0.82, 0.94)
    ax.set_ylim(-0.7, n - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("SST-2 test accuracy (mean ± sd over 3 seeds)")
    ax.set_ylabel("teacher → student")
    h = [plt.Line2D([], [], ls="", marker="o", ms=8, color=TEACHER, label="weak teacher (true labels)"),
         plt.Line2D([], [], ls="", marker="o", ms=8, color=STUDENT, label="weak-to-strong student"),
         plt.Line2D([], [], ls="", marker="D", ms=6, color=GT, label="ceiling (student size, true labels)")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=9)
    save(fig, "fig2_all_pairs.png")


# 3 ---------------------------------------------------------------- the accuracy band
def fig_band():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for run, task, col in [("E4", "SST-2", STUDENT), ("E6", "BoolQ", TEACHER)]:
        m = metrics(run)
        gt = m[m.condition.isin(["teacher_gt", "ceiling_gt"])]
        models = [x for x in ORDER if x in set(gt.model)]
        stats = [ms(gt[gt.model == x]) for x in models]
        xs = [SIZE[x] for x in models]
        mean = [s[0] for s in stats]
        ax.errorbar(xs, mean, yerr=[s[1] for s in stats], color=col, marker="o", ms=7, mec=SURFACE, mew=1.5,
                    capsize=0, elinewidth=1.2, lw=2)
        base = majority_rate(run)
        ax.axhline(base, color=col, lw=1.2, ls="--", alpha=0.8)
        ax.text(xs[-1] * 1.12, mean[-1], f"{task} (true labels)", color=INK, va="center", fontsize=9)
        ax.text(6.8e9, base - 0.008, f"{task}: always predict majority class = {base:.3f}", color=INK_2,
                fontsize=8.5, ha="right", va="top")
        ax.fill_between([1.4e8, 3.3e9], min(mean), max(mean), color=col, alpha=0.08, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([SIZE[x] for x in ORDER])
    ax.set_xticklabels([LABEL[x] for x in ORDER])
    ax.minorticks_off()
    ax.set_xlim(1.3e8, 7e9)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Pythia model size (parameters)")
    ax.set_ylabel("test accuracy after fine-tuning on true labels")
    ax.set_title("Where the models land: SST-2 crowds near the top, BoolQ near the floor", loc="left")
    save(fig, "fig3_accuracy_band.png")


# 4 ---------------------------------------------------------------- daisy chain vs direct
def fig_chain():
    m = metrics("E2")
    gens = ["p160m", "p410m", "p1b", "p1.4b"]
    x = np.arange(1, 5)
    seed = ms(m[m.condition == "seed_gt"])
    chain = [seed] + [ms(m[(m.condition == "chain") & (m.model == g)]) for g in gens[1:]]
    upper = [seed] + [ms(m[(m.condition == "gt_upper") & (m.model == g)]) for g in gens[1:]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.errorbar(x, [u[0] for u in upper], yerr=[u[1] for u in upper], color=GT, marker="D", ms=6, lw=2,
                mec=SURFACE, mew=1.5, capsize=0, elinewidth=1.2)
    ax.errorbar(x, [c[0] for c in chain], yerr=[c[1] for c in chain], color=STUDENT, marker="o", ms=8, lw=2,
                mec=SURFACE, mew=1.5, capsize=0, elinewidth=1.2)
    d = [(3, ms(m[(m.condition == "direct") & (m.model == "p1b") & (m.teacher == "p160m")])),
         (4, ms(m[(m.condition == "direct") & (m.model == "p1.4b") & (m.teacher == "p160m")]))]
    for xi, (mu, sd) in d:
        ax.errorbar(xi + 0.08, mu, yerr=sd, color=TEACHER, marker="s", ms=8, ls="", mec=SURFACE, mew=1.5,
                    capsize=0, elinewidth=1.2)
    h = [plt.Line2D([], [], color=GT, marker="D", ms=6, label="trained on true labels"),
         plt.Line2D([], [], color=STUDENT, marker="o", ms=8, label="chain: each model taught by the previous one"),
         plt.Line2D([], [], ls="", color=TEACHER, marker="s", ms=8, label="direct: taught by 160M")]
    ax.legend(handles=h, loc="upper left", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"gen {i}\n{LABEL[g]}" for i, g in zip(x, gens)])
    ax.set_xlim(0.7, 4.4)
    ax.set_ylabel("SST-2 test accuracy (mean ± sd, 3 seeds)")
    diff = chain[-1][0] - d[-1][1][0]
    ax.set_title(f"Daisy chain ends where direct supervision does (1.4B: {chain[-1][0]:.3f} vs {d[-1][1][0]:.3f})",
                 loc="left")
    save(fig, "fig4_chain_vs_direct.png")
    return diff


# 5 ---------------------------------------------------------------- error correlation
def fig_error_corr():
    e = pd.read_csv(OUT / RUNS["E2"] / "error_correlation.csv")
    g = e.groupby(["family", "model_a", "model_b"]).phi.agg(["mean", "std"]).reset_index()
    pairs = [(a, b) for a, b in g[g.family == "chain"][["model_a", "model_b"]].itertuples(index=False)]
    pairs.sort(key=lambda p: (ORDER.index(p[0]), ORDER.index(p[1])))
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    ys = np.arange(len(pairs))[::-1]
    for y, (a, b) in zip(ys, pairs):
        vals = {}
        for fam in ["chain", "gt_upper"]:
            r = g[(g.family == fam) & (g.model_a == a) & (g.model_b == b)].iloc[0]
            vals[fam] = r
        ax.plot([vals["gt_upper"]["mean"], vals["chain"]["mean"]], [y, y], color=GRID, lw=2, zorder=1)
        for fam, col, mk in [("gt_upper", GT, "D"), ("chain", STUDENT, "o")]:
            r = vals[fam]
            ax.errorbar(r["mean"], y, xerr=r["std"], fmt=mk, color=col, ms=8 if mk == "o" else 6, mec=SURFACE,
                        mew=1.5, elinewidth=1.2, capsize=0, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{LABEL[a]} vs {LABEL[b]}" for a, b in pairs])
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("error correlation φ on the test set (0 = independent mistakes, 1 = identical)")
    h = [plt.Line2D([], [], ls="", marker="o", ms=8, color=STUDENT, label="models in the daisy chain"),
         plt.Line2D([], [], ls="", marker="D", ms=6, color=GT, label="same sizes, trained independently on true labels")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    save(fig, "fig5_error_correlation.png")


# 6 ---------------------------------------------------------------- confidence loss, paired
def fig_conf():
    key = ["seed", "model", "teacher"]
    panels = [("E3", "E1", "α = 0.75 for all students,\nlinear 20% warmup"),
              ("E9", "E4", "released-code settings\n(α = 0.5, 10% step warmup)"),
              ("E10", "E1", "α = 0 (sanity check:\nidentical to naive loss)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)
    lo, hi = 0.70, 0.94
    for ax, (conf, naive, title) in zip(axes, panels):
        a, b = metrics(naive), metrics(conf)
        p = a[a.condition == "w2s"][key + ["test_acc"]].merge(b[b.condition == "w2s"][key + ["test_acc"]],
                                                              on=key, suffixes=("_n", "_c"))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(p.test_acc_n, p.test_acc_c, s=40, color=STUDENT, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        d = p.test_acc_c - p.test_acc_n
        ax.set_title(f"{conf}: {title}", loc="left", fontsize=10)
        ax.text(0.04, 0.96, f"mean change {100 * d.mean():+.1f} pts\nworse in {(d < 0).sum()}/{len(d)}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("naive loss (test accuracy)")
    axes[0].set_ylabel("confidence loss (test accuracy)")
    axes[0].text(0.83, 0.705, "below the line =\nconfidence loss hurt", fontsize=8.5, color=INK_2, ha="right", va="bottom")
    fig.suptitle("Confidence loss vs naive loss, same teachers (one dot per student and seed, SST-2)",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    save(fig, "fig6_confidence_loss.png")


# 7 ---------------------------------------------------------------- PGR heatmap
def fig_pgr_heatmap(pairs: pd.DataFrame):
    teachers = [x for x in ORDER if x in set(pairs.teacher)]
    students = [x for x in ORDER if x in set(pairs.student)]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for r in pairs.itertuples():
        i, j = teachers.index(r.teacher), students.index(r.student)
        unreliable = r.pgr_sd > abs(r.pgr)
        face = "#f0efec" if unreliable else DIVERGING(norm(np.clip(r.pgr, -1, 1)))
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=face, edgecolor=SURFACE, lw=2,
                                   hatch="///" if unreliable else None))
        dark = not unreliable and abs(r.pgr) > 0.6
        ax.text(j + 0.5, i + 0.5, f"{r.pgr:+.2f}\n± {r.pgr_sd:.2f}", ha="center", va="center", fontsize=9,
                color=SURFACE if dark else INK)
    ax.set_xlim(0, len(students))
    ax.set_ylim(len(teachers), 0)
    ax.set_xticks(np.arange(len(students)) + 0.5)
    ax.set_xticklabels([LABEL[s] for s in students])
    ax.set_yticks(np.arange(len(teachers)) + 0.5)
    ax.set_yticklabels([LABEL[t] for t in teachers])
    ax.set_xlabel("student (strong)")
    ax.set_ylabel("teacher (weak)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("PGR (mean over 3 seeds)")
    cb.outline.set_visible(False)
    ax.set_title("PGR by pair, SST-2 (hatched gray: seed spread larger than the mean)", loc="left", fontsize=10)
    save(fig, "fig7_pgr_heatmap.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    _style()
    pairs = pair_rows(metrics("E4"))
    fig_pgr_setup(pairs)
    fig_pairs(pairs)
    fig_band()
    fig_chain()
    fig_error_corr()
    fig_conf()
    fig_pgr_heatmap(pairs)


if __name__ == "__main__":
    main()
