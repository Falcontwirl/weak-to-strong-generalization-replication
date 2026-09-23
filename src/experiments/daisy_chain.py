"""Stage 2: daisy-chained weak-to-strong supervision.

Ladder M1..MN (config `models`, weak -> strong). Disjoint splits: seed_true_train, g2..gN, val.

  seed      M1 trained on GT labels of seed_true_train
  chain     Mk trained on g_k labeled by chain-M(k-1)           (k = 2..N; chain-M1 = seed M1)
  direct    Mj trained on g_j labeled by chain-Mi, j - i >= 2   (e.g. M1->M4, M1->M3, M2->M4)
  gt_upper  Mk trained on GT labels of g_k                      (matched data; the achievable ceiling)
  zero_shot base LM prompted, no task training

Chain-MN, direct M1->MN, direct M2->MN and GT-MN all train on the same g_N examples.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import metrics as M
from .. import plotting as P
from ..analysis import aggregate, failures_md, md_table, ok, pm, pred_metrics, probs, student_vs_teacher
from ..jobs import Job, JobResult, RunContext

NAME = "daisy_chain"
SEED_SPLIT = "seed_true_train"


def names(cfg):
    return [m["name"] for m in cfg["models"]]


def gsplit(k: int) -> str:
    return f"g{k}"


def required_splits(cfg: dict) -> tuple[str, ...]:
    n = len(cfg["models"])
    if n < 2:
        raise ValueError("daisy_chain needs at least 2 models")
    return ("val", SEED_SPLIT, *[gsplit(k) for k in range(2, n + 1)])


def direct_pairs(cfg: dict) -> list[tuple[int, int]]:
    """1-based (teacher generation i, student generation j) pairs that skip >= 1 rung."""
    n = len(cfg["models"])
    spec = cfg["daisy_chain"].get("direct", "auto")
    if spec == "auto":
        return [(i, j) for i, j in itertools.combinations(range(1, n + 1), 2) if j - i >= 2]
    nm = names(cfg)
    out = []
    for w, s in spec:
        i, j = nm.index(w) + 1, nm.index(s) + 1
        if j - i < 2:
            raise ValueError(f"direct pair ({w}, {s}) must skip at least one rung (adjacent pairs are the chain)")
        out.append((i, j))
    return out


def chain_name(seed, k, cfg):
    m = names(cfg)[k - 1]
    return f"s{seed}/seed_gt/{m}" if k == 1 else f"s{seed}/chain/g{k}/{m}"


def upper_name(seed, k, cfg):
    return chain_name(seed, 1, cfg) if k == 1 else f"s{seed}/gt_upper/{names(cfg)[k - 1]}"


def direct_name(seed, i, j, cfg):
    nm = names(cfg)
    return f"s{seed}/direct/{nm[i - 1]}->{nm[j - 1]}"


def zs_name(m):
    return f"zero_shot/{m}"


def build_jobs(cfg: dict, seed: int, first: bool) -> list[Job]:
    nm = names(cfg)
    n = len(nm)
    jobs = [Job(chain_name(seed, 1, cfg), nm[0], seed, train_split=SEED_SPLIT, condition="seed_gt", generation=1)]
    for k in range(2, n + 1):
        jobs.append(Job(chain_name(seed, k, cfg), nm[k - 1], seed, train_split=gsplit(k),
                        label_source=chain_name(seed, k - 1, cfg), condition="chain", generation=k,
                        meta={"teacher": nm[k - 2], "teacher_generation": k - 1}))
        jobs.append(Job(upper_name(seed, k, cfg), nm[k - 1], seed, train_split=gsplit(k), condition="gt_upper",
                        generation=k))
    for i, j in direct_pairs(cfg):
        jobs.append(Job(direct_name(seed, i, j, cfg), nm[j - 1], seed, train_split=gsplit(j),
                        label_source=chain_name(seed, i, cfg), condition="direct", generation=j,
                        meta={"teacher": nm[i - 1], "teacher_generation": i}))
    if first and cfg["zero_shot"]["enabled"]:
        jobs += [Job(zs_name(m), m, 0, kind="zero_shot", condition="zero_shot") for m in nm]
    return jobs


# ------------------------------------------------------------------------------------------ analysis

def analyze(ctx: RunContext, results: dict[str, JobResult]) -> None:
    cfg, out, task = ctx.cfg, ctx.out_dir, ctx.task
    acfg = cfg["analysis"]
    eb, nb = acfg["ece_bins"], acfg["n_bootstrap"]
    nm = names(cfg)
    n = len(nm)
    cache: dict[str, pd.DataFrame] = {}

    def preds(name):
        if name not in cache:
            cache[name] = results[name].test_preds()
        return cache[name]

    rows, boots, bins, trans, corr, traj = [], [], [], [], [], []
    for m in nm:
        if ok(results, zs_name(m)):
            rows.append({"seed": 0, "experiment": NAME, "condition": "zero_shot", "model": m, "teacher": "",
                         "generation": nm.index(m) + 1, **pred_metrics(preds(zs_name(m)), eb)})

    for seed in cfg["seeds"]:
        m1 = chain_name(seed, 1, cfg)
        m1_acc = M.accuracy(preds(m1)["pred"], preds(m1)["label"]) if ok(results, m1) else float("nan")
        if ok(results, m1):
            rows.append({"seed": seed, "experiment": NAME, "condition": "seed_gt", "model": nm[0], "teacher": "",
                         "generation": 1, **pred_metrics(preds(m1), eb)})
        for k in range(2, n + 1):
            if ok(results, upper_name(seed, k, cfg)):
                rows.append({"seed": seed, "experiment": NAME, "condition": "gt_upper", "model": nm[k - 1],
                             "teacher": "", "generation": k, **pred_metrics(preds(upper_name(seed, k, cfg)), eb)})

        def student_row(cond, sname, tname, i, j, split):
            if not ok(results, sname, tname):
                return
            sp, tp = preds(sname), preds(tname)
            lab = results[tname].labels(split, task.label_names)
            r = {"seed": seed, "experiment": NAME, "condition": cond, "model": nm[j - 1], "teacher": nm[i - 1],
                 "generation": j, "teacher_generation": i, **pred_metrics(sp, eb), **student_vs_teacher(sp, tp),
                 "teacher_test_acc": M.accuracy(tp["pred"], tp["label"]),
                 "teacher_label_acc": float(lab["teacher_correct"].mean())}
            r["w2s_gain"] = r["test_acc"] - r["teacher_test_acc"]
            un = upper_name(seed, j, cfg)
            if ok(results, un):
                up = preds(un)
                r["upper_acc"] = M.accuracy(up["pred"], up["label"])
                r["pgr"], v = M.pgr(r["teacher_test_acc"], r["test_acc"], r["upper_acc"])
                r["pgr_valid"] = int(v)
                # PGR measured from the original weak supervisor M1: comparable between chain and direct.
                r["pgr_vs_m1"], v1 = M.pgr(m1_acc, r["test_acc"], r["upper_acc"])
                r["pgr_vs_m1_valid"] = int(v1)
                boots.append({"seed": seed, "comparison": f"{cond} {nm[i - 1]}->{nm[j - 1]}: student - gt_upper",
                              **M.paired_bootstrap(sp["correct"], up["correct"], nb, seed)})
            rows.append(r)
            boots.append({"seed": seed, "comparison": f"{cond} {nm[i - 1]}->{nm[j - 1]}: student - teacher",
                          **M.paired_bootstrap(sp["correct"], tp["correct"], nb, seed)})
            for b in M.confidence_bins(probs(tp), tp["pred"], tp["label"], sp["pred"], acfg["n_conf_bins"]):
                bins.append({"seed": seed, "step": f"{cond}: {nm[i - 1]}->{nm[j - 1]}", **b})

        for k in range(2, n + 1):
            student_row("chain", chain_name(seed, k, cfg), chain_name(seed, k - 1, cfg), k - 1, k, gsplit(k))
            if ok(results, chain_name(seed, k, cfg), chain_name(seed, k - 1, cfg)):
                a, b = preds(chain_name(seed, k - 1, cfg)), preds(chain_name(seed, k, cfg))
                trans.append({"seed": seed, "transition": f"{nm[k - 2]}->{nm[k - 1]}", "step": k - 1,
                              **M.transitions(a["correct"], b["correct"])})
        for i, j in direct_pairs(cfg):
            student_row("direct", direct_name(seed, i, j, cfg), chain_name(seed, i, cfg), i, j, gsplit(j))

        # Final model: chain vs direct vs GT upper bound.
        cN, uN = chain_name(seed, n, cfg), upper_name(seed, n, cfg)
        for i, j in direct_pairs(cfg):
            dn = direct_name(seed, i, j, cfg)
            if j == n and ok(results, cN, dn):
                boots.append({"seed": seed, "comparison": f"chain {nm[-1]} - direct {nm[i - 1]}->{nm[-1]}",
                              **M.paired_bootstrap(preds(cN)["correct"], preds(dn)["correct"], nb, seed)})

        # Error correlation among chain generations vs among independently GT-trained models.
        for fam, fn in (("chain", chain_name), ("gt_upper", upper_name)):
            for a, b in itertools.combinations(range(1, n + 1), 2):
                na, nb_ = fn(seed, a, cfg), fn(seed, b, cfg)
                if ok(results, na, nb_):
                    corr.append({"seed": seed, "family": fam, "gen_a": a, "gen_b": b, "model_a": nm[a - 1],
                                 "model_b": nm[b - 1],
                                 **M.error_correlation(~preds(na)["correct"].to_numpy(),
                                                       ~preds(nb_)["correct"].to_numpy())})

        # Per-example trajectories on the test set.
        cols = {}
        for k in range(1, n + 1):
            if ok(results, chain_name(seed, k, cfg)):
                p = preds(chain_name(seed, k, cfg))
                cols[f"chain_g{k}_{nm[k - 1]}_correct"] = p["correct"].to_numpy()
                cols[f"chain_g{k}_{nm[k - 1]}_conf"] = probs(p).max(1)
        for i, j in direct_pairs(cfg):
            if ok(results, direct_name(seed, i, j, cfg)):
                cols[f"direct_{nm[i - 1]}->{nm[j - 1]}_correct"] = preds(direct_name(seed, i, j, cfg))["correct"].to_numpy()
        for k in range(2, n + 1):
            if ok(results, upper_name(seed, k, cfg)):
                cols[f"gt_upper_{nm[k - 1]}_correct"] = preds(upper_name(seed, k, cfg))["correct"].to_numpy()
        if cols and ok(results, m1):
            t = pd.DataFrame({"seed": seed, "example_id": preds(m1)["example_id"], "label": preds(m1)["label"],
                              **cols})
            chain_cols = [c for c in cols if c.startswith("chain_") and c.endswith("_correct")]
            t["chain_pattern"] = ["".join("C" if v else "W" for v in r) for r in t[chain_cols].to_numpy()]
            traj.append(t)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary = aggregate(metrics) if len(metrics) else pd.DataFrame()
    summary.to_csv(out / "metrics_summary.csv", index=False)
    boots = pd.DataFrame(boots)
    boots.to_csv(out / "bootstrap.csv", index=False)
    trans = pd.DataFrame(trans)
    trans.to_csv(out / "error_transitions.csv", index=False)
    corr = pd.DataFrame(corr)
    corr.to_csv(out / "error_correlation.csv", index=False)
    traj = pd.concat(traj) if traj else pd.DataFrame()
    traj.to_csv(out / "example_trajectories.csv", index=False)
    binsdf = pd.DataFrame(bins)
    pooled = pd.DataFrame()
    if len(binsdf):
        pooled = (binsdf.assign(corr=binsdf.error_correction.fillna(0) * binsdf.n_teacher_errors)
                  .groupby(["step", "bin", "conf_low", "conf_high"], as_index=False, sort=False)
                  .agg(n=("n", "sum"), n_teacher_errors=("n_teacher_errors", "sum"), corr=("corr", "sum")))
        pooled["error_correction"] = pooled["corr"] / pooled["n_teacher_errors"].where(pooled.n_teacher_errors > 0)
        pooled = pooled.drop(columns="corr")
        pooled.to_csv(out / "confidence_bins.csv", index=False)
        P.plot_confidence_bins(pooled, out / "confidence_bins.png", "step",
                               "Teacher errors corrected, by teacher confidence")
    if len(metrics):
        P.plot_accuracy_by_generation(metrics, cfg["models"], out / "accuracy_by_generation.png")
        P.plot_teacher_vs_student(metrics, out / "teacher_vs_student.png")
        P.plot_chain_vs_direct(metrics, nm, out / "chain_vs_direct.png")
    if len(trans):
        P.plot_error_transitions(trans, out / "error_transition.png")
    if len(corr):
        P.plot_error_correlation(corr, nm, out / "error_correlation.png")
    _write_summary(ctx, summary, boots, trans, corr, traj, results)


def _write_summary(ctx, summary, boots, trans, corr, traj, results) -> None:
    cfg = ctx.cfg
    nm = names(cfg)
    n = len(nm)
    L = [f"# Daisy-chained weak-to-strong generalization: {cfg['name']}", "",
         f"Dataset `{cfg['dataset']['path']}/{cfg['dataset'].get('name')}`; test n = {len(ctx.task.splits['test'])}; "
         f"seeds = {cfg['seeds']}; loss = `{cfg['train']['loss']}`; selection = `{cfg['train']['selection']}`.", "",
         "Chain: GT → " + " → ".join(f"`{m}`" for m in nm) + ". Each student sees only its immediate "
         "predecessor's labels on a fresh disjoint split; all conditions are evaluated on the same held-out test set.",
         ""]
    if len(cfg["seeds"]) == 1:
        L += ["> Single seed: treat differences as descriptive, not significant.", ""]

    def get(cond, model, teacher=""):
        if not len(summary):
            return None
        s = summary[(summary.condition == cond) & (summary.model == model) & (summary.teacher == teacher)]
        return s.iloc[0] if len(s) else None

    def col(r, c, d=3):
        return pm(r.get(f"{c}_mean"), r.get(f"{c}_std"), d) if r is not None else "–"

    if len(summary):
        L += ["## Reference points (test accuracy)", ""]
        ref = []
        for k, m in enumerate(nm, 1):
            z = get("zero_shot", m)
            u = get("seed_gt", m) if k == 1 else get("gt_upper", m)
            ref.append({"gen": k, "model": m, "zero-shot": col(z, "test_acc"),
                        "GT-trained (upper bound)": col(u, "test_acc")})
        L += [md_table(ref, list(ref[0])), ""]
        L += ["## Chain: does each student beat its immediate supervisor?", "",
              "`PGR` uses the immediate teacher; `PGR vs M1` measures recovery of the gap between the original "
              "weak supervisor M1 and this model's GT upper bound, so chain and direct are directly comparable.", ""]
        tab, flagged = [], []
        for cond in ("chain", "direct"):
            s = summary[summary.condition == cond]
            for _, r in s.sort_values(["generation", "teacher"]).iterrows():
                if not (r.get("upper_acc_mean", np.nan) > r.get("teacher_test_acc_mean", np.nan)):
                    flagged.append(f"{cond} {r.teacher}->{r.model}")
                tab.append({"condition": cond, "teacher→student": f"{r.teacher}→{r.model}",
                            "teacher": col(r, "teacher_test_acc"), "student": col(r, "test_acc"),
                            "gain": col(r, "w2s_gain"), "GT upper": col(r, "upper_acc"),
                            "PGR": col(r, "pgr", 2), "PGR vs M1": col(r, "pgr_vs_m1", 2),
                            "agree": col(r, "agreement"), "err. inherited": col(r, "error_inheritance"),
                            "err. corrected": col(r, "error_correction"), "new err.": col(r, "new_error_rate")})
        if tab:
            L += [md_table(tab, list(tab[0])), ""]
        if flagged:
            L += ["**No capability gap** (GT upper bound ≤ teacher; PGR not meaningful): " + ", ".join(flagged), ""]
    if len(boots):
        L += ["## Paired bootstrap, 95% CI on test-accuracy differences", ""]
        key = boots[boots.comparison.str.contains("student - teacher|chain .* - direct|- gt_upper", regex=True)]
        agg = key.groupby("comparison", sort=False).agg(diff=("diff", "mean"), diff_std=("diff", "std"),
                                                        ci_low=("ci_low", "min"), ci_high=("ci_high", "max"),
                                                        seeds=("seed", "count")).reset_index()
        br = [{"comparison": r.comparison, "mean diff": f"{r['diff']:+.4f}" + (f" ± {r.diff_std:.4f}" if r.seeds > 1 else ""),
               "per-seed CI envelope": f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]", "seeds": r.seeds}
              for _, r in agg.iterrows()]
        L += [md_table(br, ["comparison", "mean diff", "per-seed CI envelope", "seeds"]),
              "", "Per-seed CIs are in `bootstrap.csv`. The envelope is the widest interval across seeds.", ""]
    if len(trans):
        t = trans.groupby("transition", sort=False)[["C->C", "W->C", "C->W", "W->W"]].mean().reset_index()
        L += ["## Error transitions between consecutive chain generations (fraction of test, mean over seeds)", "",
              md_table([{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}
                        for r in t.to_dict("records")], list(t.columns)), ""]
    if len(corr):
        c = corr.groupby(["family", "model_a", "model_b"], sort=False)[["phi", "p_err_b_given_err_a"]].mean().reset_index()
        L += ["## Error correlation (φ between error indicators on test)", "",
              "If chain errors are more correlated than errors of the independently GT-trained models of the "
              "same sizes, supervision is propagating the supervisor's specific mistakes.", "",
              md_table([{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}
                        for r in c.to_dict("records")], list(c.columns)), ""]
    if len(traj):
        pat = traj["chain_pattern"].value_counts(normalize=True).head(8)
        L += ["## Most common per-example chain trajectories (C = correct, W = wrong; gen 1 → N)", "",
              md_table([{"pattern": k, "fraction": f"{v:.3f}"} for k, v in pat.items()], ["pattern", "fraction"]), ""]
    L += ["## Figures", ""] + [f"![{f}]({f}.png)" for f in
                               ("accuracy_by_generation", "teacher_vs_student", "chain_vs_direct", "error_transition",
                                "error_correlation", "confidence_bins")]
    L += ["", "## Caveats", "",
          "- Pseudo-labeling chains mix capability scaling with ordinary self-training / noisy-label dynamics; "
          "interpret the chain only relative to the direct and GT-upper controls.",
          "- A pair whose GT upper bound does not exceed its teacher has no meaningful weak-to-strong gap (flagged above).",
          "- Model selection for weakly-supervised students uses teacher-labeled validation data "
          "(no ground truth) unless `train.selection` says otherwise.", "",
          "## Failed / excluded runs", "", failures_md(results), ""]
    (ctx.out_dir / "summary.md").write_text("\n".join(L))
