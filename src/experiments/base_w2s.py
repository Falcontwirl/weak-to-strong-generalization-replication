"""Stage 1: the base weak-to-strong experiment (Burns et al., 2023).

For a pair (weak W, strong S):
  weak performance     W trained on GT labels of `weak_train`, evaluated on test
  W2S performance      S trained on `strong_train` labeled by that W, evaluated on test
  strong ceiling       S trained on GT labels of `strong_train`, evaluated on test
  PGR = (W2S - weak) / (ceiling - weak)
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from .. import metrics as M
from .. import plotting as P
from ..analysis import aggregate, failures_md, md_table, ok, pm, pred_metrics, student_vs_teacher
from ..jobs import Job, JobResult, RunContext

NAME = "base_w2s"
REQUIRED_SPLITS = ("val", "weak_train", "strong_train")


def required_splits(cfg: dict) -> tuple[str, ...]:
    return REQUIRED_SPLITS


def pairs(cfg: dict) -> list[tuple[str, str]]:
    names = [m["name"] for m in cfg["models"]]
    spec = cfg["base_w2s"]["pairs"]
    if spec == "all":
        return list(itertools.combinations(names, 2))
    out = []
    for w, s in spec:
        if names.index(w) >= names.index(s):
            raise ValueError(f"pair ({w}, {s}): weak model must come before strong model in the ladder")
        out.append((w, s))
    return out


def teacher_name(seed, w):
    return f"s{seed}/teacher_gt/{w}"


def ceiling_name(seed, s):
    return f"s{seed}/ceiling_gt/{s}"


def w2s_name(seed, w, s):
    return f"s{seed}/w2s/{w}->{s}"


def zs_name(m):
    return f"zero_shot/{m}"


def build_jobs(cfg: dict, seed: int, first: bool) -> list[Job]:
    names = [m["name"] for m in cfg["models"]]
    gen = {n: i + 1 for i, n in enumerate(names)}
    ps = pairs(cfg)
    jobs: list[Job] = []
    for w in dict.fromkeys(w for w, _ in ps):
        jobs.append(Job(teacher_name(seed, w), w, seed, train_split="weak_train", condition="teacher_gt",
                        generation=gen[w]))
    for s in dict.fromkeys(s for _, s in ps):
        jobs.append(Job(ceiling_name(seed, s), s, seed, train_split="strong_train", condition="ceiling_gt",
                        generation=gen[s]))
    for w, s in ps:
        jobs.append(Job(w2s_name(seed, w, s), s, seed, train_split="strong_train",
                        label_source=teacher_name(seed, w), condition="w2s", generation=gen[s],
                        meta={"teacher": w}))
    if first and cfg["zero_shot"]["enabled"]:
        # Zero-shot is deterministic and seed-independent: run once.
        jobs += [Job(zs_name(m), m, 0, kind="zero_shot", condition="zero_shot") for m in names]
    return jobs


def analyze(ctx: RunContext, results: dict[str, JobResult]) -> None:
    cfg, out = ctx.cfg, ctx.out_dir
    acfg = cfg["analysis"]
    eb = acfg["ece_bins"]
    rows, boots, bins = [], [], []
    preds_cache: dict[str, pd.DataFrame] = {}

    def preds(name):
        if name not in preds_cache:
            preds_cache[name] = results[name].test_preds()
        return preds_cache[name]

    names = [m["name"] for m in cfg["models"]]
    for m in names:
        if ok(results, zs_name(m)):
            rows.append({"seed": 0, "experiment": NAME, "condition": "zero_shot", "model": m, "teacher": "",
                         "generation": 0, **pred_metrics(preds(zs_name(m)), eb)})

    for seed in cfg["seeds"]:
        ps = pairs(cfg)
        for w in dict.fromkeys(w for w, _ in ps):
            tn = teacher_name(seed, w)
            if not ok(results, tn):
                continue
            lab = results[tn].labels("strong_train", ctx.task.label_names)
            rows.append({"seed": seed, "experiment": NAME, "condition": "teacher_gt", "model": w, "teacher": "",
                         "generation": names.index(w) + 1, **pred_metrics(preds(tn), eb),
                         "teacher_label_acc": float(lab["teacher_correct"].mean())})
        for s in dict.fromkeys(s for _, s in ps):
            if ok(results, ceiling_name(seed, s)):
                rows.append({"seed": seed, "experiment": NAME, "condition": "ceiling_gt", "model": s, "teacher": "",
                             "generation": names.index(s) + 1, **pred_metrics(preds(ceiling_name(seed, s)), eb)})
        for w, s in ps:
            tn, cn, sn = teacher_name(seed, w), ceiling_name(seed, s), w2s_name(seed, w, s)
            if not ok(results, tn, sn):
                continue
            sp, tp = preds(sn), preds(tn)
            r = {"seed": seed, "experiment": NAME, "condition": "w2s", "model": s, "teacher": w,
                 "generation": names.index(s) + 1, **pred_metrics(sp, eb), **student_vs_teacher(sp, tp)}
            r["teacher_test_acc"] = M.accuracy(tp["pred"], tp["label"])
            r["teacher_label_acc"] = float(results[tn].labels("strong_train", ctx.task.label_names)["teacher_correct"].mean())
            r["w2s_gain"] = r["test_acc"] - r["teacher_test_acc"]
            if ok(results, cn):
                cp = preds(cn)
                r["ceiling_acc"] = M.accuracy(cp["pred"], cp["label"])
                r["pgr"], valid = M.pgr(r["teacher_test_acc"], r["test_acc"], r["ceiling_acc"])
                r["pgr_valid"] = int(valid)
                b = M.paired_bootstrap(sp["correct"], cp["correct"], acfg["n_bootstrap"], seed)
                boots.append({"seed": seed, "comparison": f"w2s({w}->{s}) - ceiling({s})", **b})
            rows.append(r)
            b = M.paired_bootstrap(sp["correct"], tp["correct"], acfg["n_bootstrap"], seed)
            boots.append({"seed": seed, "comparison": f"w2s({w}->{s}) - weak({w})", **b})
            p = tp[[c for c in tp.columns if c.startswith("prob_")]].to_numpy()
            for row in M.confidence_bins(p, tp["pred"], tp["label"], sp["pred"], acfg["n_conf_bins"]):
                bins.append({"seed": seed, "pair": f"{w}->{s}", **row})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(boots).to_csv(out / "bootstrap.csv", index=False)
    summary = aggregate(metrics) if len(metrics) else pd.DataFrame()
    summary.to_csv(out / "metrics_summary.csv", index=False)
    binsdf = pd.DataFrame(bins)
    if len(binsdf):
        pooled = (binsdf.assign(corr=binsdf.error_correction * binsdf.n_teacher_errors)
                  .groupby(["pair", "bin", "conf_low", "conf_high"], as_index=False)
                  .agg(n=("n", "sum"), n_teacher_errors=("n_teacher_errors", "sum"), corr=("corr", "sum")))
        pooled["error_correction"] = pooled["corr"] / pooled["n_teacher_errors"].where(pooled.n_teacher_errors > 0)
        pooled.drop(columns="corr").to_csv(out / "confidence_bins.csv", index=False)
        P.plot_confidence_bins(pooled, out / "confidence_bins.png", "pair",
                               "Teacher errors corrected, by teacher confidence")
    if len(summary) and (summary.condition == "w2s").any() and (summary.condition == "ceiling_gt").any():
        P.plot_w2s_accuracy(summary, cfg["models"], out / "w2s_accuracy.png")
        if "pgr_mean" in summary:
            P.plot_pgr_grid(summary, cfg["models"], out / "pgr_grid.png")
    _write_summary(ctx, summary, pd.DataFrame(boots), results)


def _write_summary(ctx: RunContext, summary: pd.DataFrame, boots: pd.DataFrame, results) -> None:
    cfg = ctx.cfg
    n_seeds = len(cfg["seeds"])
    L = [f"# Weak-to-strong generalization: {cfg['name']}", "",
         f"Dataset: `{cfg['dataset']['path']}/{cfg['dataset'].get('name')}`; "
         f"test n = {len(ctx.task.splits['test'])}; seeds = {cfg['seeds']}; loss = `{cfg['train']['loss']}`; "
         f"selection = `{cfg['train']['selection']}`.", "",
         "Ladder: " + ", ".join(f"`{m['name']}` ({m['id']})" for m in cfg["models"]), ""]
    if n_seeds == 1:
        L += ["> Single seed: differences below are not evidence of significance on their own.", ""]
    if len(summary):
        def get(cond, model, teacher=""):
            s = summary[(summary.condition == cond) & (summary.model == model) & (summary.teacher == teacher)]
            return s.iloc[0] if len(s) else None

        L += ["## Per-model reference points", ""]
        ref = []
        for m in cfg["models"]:
            n = m["name"]
            t, c, z = get("teacher_gt", n), get("ceiling_gt", n), get("zero_shot", n)
            ref.append({"model": n,
                        "zero-shot": pm(z.test_acc_mean, None) if z is not None else "n/a",
                        "GT on weak_train (as supervisor)": pm(t.test_acc_mean, t.test_acc_std) if t is not None else "–",
                        "GT on strong_train (ceiling)": pm(c.test_acc_mean, c.test_acc_std) if c is not None else "–"})
        L += [md_table(ref, list(ref[0].keys())), ""]
        L += ["## Weak-to-strong pairs", "",
              "PGR = (W2S − weak) / (ceiling − weak). Pairs whose ceiling does not exceed the weak model "
              "are flagged: they have no real capability gap and their PGR is not meaningful.", ""]
        pr = []
        flagged = []
        for _, r in summary[summary.condition == "w2s"].iterrows():
            gap = r.get("ceiling_acc_mean", float("nan")) - r.get("teacher_test_acc_mean", float("nan"))
            if not (gap > 0):
                flagged.append(f"{r.teacher}->{r.model}")
            pr.append({"weak→strong": f"{r.teacher}→{r.model}",
                       "weak": pm(r.teacher_test_acc_mean, r.teacher_test_acc_std),
                       "W2S": pm(r.test_acc_mean, r.test_acc_std),
                       "ceiling": pm(r.get("ceiling_acc_mean"), r.get("ceiling_acc_std")),
                       "PGR": pm(r.get("pgr_mean"), r.get("pgr_std"), 2),
                       "agree": pm(r.agreement_mean, None),
                       "err. corrected": pm(r.error_correction_mean, None),
                       "gap≤0": "⚠" if not (gap > 0) else ""})
        if pr:
            L += [md_table(pr, list(pr[0].keys())), ""]
        if flagged:
            L += [f"**No capability gap (excluded from PGR interpretation):** {', '.join(flagged)}", ""]
    if len(boots):
        L += ["## Paired bootstrap (95% CI on test accuracy differences, per seed)", ""]
        br = [{"seed": r.seed, "comparison": r.comparison, "diff": f"{r['diff']:+.4f}",
               "95% CI": f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"} for _, r in boots.iterrows()]
        L += [md_table(br, ["seed", "comparison", "diff", "95% CI"]), ""]
    L += ["## Figures", "", "![w2s accuracy](w2s_accuracy.png)", "", "![pgr](pgr_grid.png)", "",
          "![confidence bins](confidence_bins.png)", "", "## Failed / excluded runs", "", failures_md(results), ""]
    (ctx.out_dir / "summary.md").write_text("\n".join(L))
