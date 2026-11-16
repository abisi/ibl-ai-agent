"""Old DLC shuffles 3 / 5 / 6 (Tracking2D-Axel-2024-08-10, one fine-tuning chain) vs the merged model(s) on the mice
never seen by the old chain (training_plan.md section 6): AB151-155, MH015, MH062, MH064-065, MH067-070.

Strict set = frames of these mice in the merged TEST split: no model trained on them, and the old chain never saw
these mice (the merged model saw other frames of the same mice: it is a "seen mouse, unseen frame" test for the
merged model and an "unseen mouse" test for the old ones, so it checks that the merged model is not worse and how
much it gains, not the merged model's generalisation to new mice). Reference set = the other (merged TRAIN) frames of
these mice, where only the old models are held out.
Same rules as the summary (019): merged ground truth for all models, keypoints of the frame's view only, spout and
top tongue excluded, error over all predictions whatever their likelihood, likelihood cutoff 0.5 for derived
quantities (compute_kinematics_alt whisker angle, jaw angle, 4-point pupil area).
Inputs: combined_results_ks4/ssl-dlc-kuma/shuffle_bench/pred_s{3,5,6}.csv (007, 2026-11-04), merged predictions
(best snapshot of every evaluated merged architecture, 019 inputs).
Outputs: DLCProjects/merged_training_2026-11-09/unseen_mice_benchmark/ (figure, errors_long.csv, summary.csv)
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 021_unseen_mice_benchmark.py
"""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

spec = importlib.util.spec_from_file_location("ts", Path(__file__).with_name("019_training_summary.py"))
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)
ps = ts.ps

BENCH = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma/shuffle_bench")
OUT = ts.SIDE / "unseen_mice_benchmark"
UNSEEN = ["AB151", "AB152", "AB153", "AB154", "AB155", "MH015", "MH062", "MH064", "MH065", "MH067", "MH068", "MH069",
          "MH070"]
OLD = {3: "Old s3", 5: "Old s5", 6: "Old s6"}
OLD_COLORS = {"Old s3": "#bdbdbd", "Old s5": "#969696", "Old s6": "#636363"}


def old_long(shuffle, frames):
    p = pd.read_csv(BENCH / f"pred_s{shuffle}.csv")
    p = p.merge(frames[["frame", "folder", "image"]].assign(image=frames.image.str.split("/").str[-1]),
                on=["folder", "image"])
    long = p.melt(id_vars=["frame"], value_vars=[c for c in p.columns if c.endswith(("_x", "_y", "_likelihood"))])
    long[["bodypart", "coord"]] = long.variable.str.rsplit("_", n=1, expand=True)
    return long.pivot_table(index=["frame", "bodypart"], columns="coord", values="value").reset_index()


def model_errors(pred_long, frames, gt_long):
    """errors() of 019 for a prediction long table (frame, bodypart, x, y, likelihood)."""
    e = gt_long.merge(pred_long, on=["frame", "bodypart"])
    e["labelled"] = e[["x_gt", "y_gt"]].notna().all(axis=1)
    e["err_px"] = np.hypot(e.x - e.x_gt, e.y - e.y_gt)
    e = e.merge(frames[["frame", "split", "view", "mouse", "mm_per_px"]], on="frame")
    e["err_mm"] = e.err_px * e.mm_per_px
    return ts.in_view(e)


def merged_models(frames, gt_long):
    snap = pd.read_csv(ts.OUT / "errors_per_snapshot.csv")
    t = snap[(snap.split == "test") & (snap.view == "all")]
    out = {}
    for s, g in t.groupby("shuffle"):
        ep = int(g.loc[g.mean_err_px.idxmin(), "epoch"])
        out[f"Merged {ts.MODELS[s]}"] = (ts.errors(ts.snapshot_files(s)[ep], frames, gt_long), ts.MODEL_COLORS[s], ep)
    return out


def boot_ci(values, groups, n=2000, seed=0):
    """95% CI of the mean by resampling mice (groups)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": values, "g": groups})
    gm = df.groupby("g").v.agg(["sum", "count"])
    ids = gm.index.to_numpy()
    bs = []
    for _ in range(n):
        pick = gm.loc[rng.choice(ids, len(ids))]
        bs.append(pick["sum"].sum() / pick["count"].sum())
    return np.percentile(bs, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ps.set_nature_rc()
    frames, gt_long = ts.load_ground_truth()
    frames["unseen"] = frames.mouse.isin(UNSEEN)
    models = {name: (model_errors(old_long(s, frames), frames, gt_long), OLD_COLORS[name], None)
              for s, name in OLD.items()}
    models.update(merged_models(frames, gt_long))
    sets = {"strict": frames[frames.unseen & (frames.split == "test")].frame,
            "reference": frames[frames.unseen & (frames.split == "train")].frame}
    rows, per_frame = [], []
    for name, (e, _, ep) in models.items():
        lab = e[e.labelled]
        for set_name, fr in sets.items():
            d = lab[lab.frame.isin(fr)]
            for view, dv in [("all", d)] + list(d.groupby("view")):
                lo, hi = boot_ci(dv.err_px.to_numpy(), dv.mouse.to_numpy())
                rows.append({"model": name, "epoch": ep, "set": set_name, "view": view, "n_frames": dv.frame.nunique(),
                             "n_mice": dv.mouse.nunique(), "n_keypoints": len(dv), "mean_err_px": dv.err_px.mean(),
                             "ci_lo": lo, "ci_hi": hi, "median_err_px": dv.err_px.median(),
                             "mean_err_mm": dv.err_mm.mean()})
            if set_name == "strict":
                per_frame.append(d.groupby(["frame", "view", "mouse"]).err_px.mean().rename(name))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    pf = pd.concat(per_frame, axis=1).reset_index()
    pf = pf[pf.frame.isin(sets["strict"])]
    pf.to_csv(OUT / "errors_per_frame_strict.csv", index=False)

    names = list(models)
    merged = [n for n in names if n.startswith("Merged")]
    colors = {n: c for n, (_, c, _) in models.items()}

    # paired tests (per frame, strict set): each merged model vs each old shuffle
    tests = []
    for m in merged:
        for o in OLD.values():
            for view in ("side", "top"):
                d = pf[pf.view == view].dropna(subset=[m, o])
                w = stats.wilcoxon(d[m], d[o]) if len(d) > 5 else None
                tests.append({"merged": m, "old": o, "view": view, "n_frames": len(d),
                              "median_diff_px": float((d[m] - d[o]).median()),
                              "wilcoxon_p": float(w.pvalue) if w else np.nan})
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / "paired_tests_strict.csv", index=False)

    fig = plt.figure(figsize=(183 * ps.MM, 130 * ps.MM))
    gs = fig.add_gridspec(2, 4, left=0.07, right=0.97, top=0.92, bottom=0.12, wspace=0.6, hspace=0.75)
    # a: mean error per view, strict set, with mouse-bootstrap CI
    for j, view in enumerate(ts.VIEWS):
        ax = fig.add_subplot(gs[0, j]); ts.panel_label(ax, "a" if j == 0 else "")
        s = summary[(summary.set == "strict") & (summary.view == view)].set_index("model").reindex(names)
        x = np.arange(len(names))
        ax.bar(x, s.mean_err_px, color=[colors[n] for n in names], width=0.7)
        ax.errorbar(x, s.mean_err_px, yerr=[s.mean_err_px - s.ci_lo, s.ci_hi - s.mean_err_px], fmt="none", ecolor="k",
                    elinewidth=0.6, capsize=1.5)
        ax.set_xticks(x, [n.replace("Merged ", "") for n in names], rotation=45, ha="right")
        ax.set_ylabel("Mean test error (px)" if j == 0 else "")
        ax.set_title(f"{view.capitalize()} view: {int(s.n_frames.iloc[0])} frames, {int(s.n_mice.iloc[0])} mice", pad=2)
        ts.mm_axis(ax, frames[frames.view == view].mm_per_px.median())
    # b: per-frame paired comparison, best merged vs old s6 (the old model behind most MH h5s)
    best = summary[(summary.set == "strict") & (summary.view == "all") & summary.model.isin(merged)]
    best = best.loc[best.mean_err_px.idxmin(), "model"]
    ax = fig.add_subplot(gs[0, 2]); ts.panel_label(ax, "b")
    for view in ts.VIEWS:
        d = pf[pf.view == view]
        for o, mk in (("Old s3", "o"), ("Old s6", "^")):
            ax.scatter(d[o], d[best], s=5, marker=mk, color=ts.VIEW_COLORS[view], edgecolors="none",
                       label=f"{view}, vs {o.replace('Old ', '')}")
    lim = [0.5, max(pf[[best, "Old s3", "Old s6"]].max().max() * 1.2, 20)]
    ax.plot(lim, lim, color="0.5", lw=0.5, ls=":"); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Old model, frame error (px)"); ax.set_ylabel(f"{best.replace('Merged ', 'Merged ')} (px)")
    ax.legend(loc="upper left", fontsize=4.5, handletextpad=0.1)
    # c: per-mouse mean error, old s3 / s6 vs best merged
    ax = fig.add_subplot(gs[0, 3]); ts.panel_label(ax, "c")
    pm = pf.groupby(["mouse", "view"])[names].mean().reset_index()
    for _, r in pm.iterrows():
        ax.plot([0, 1, 2], [r["Old s3"], r["Old s6"], r[best]], color=ts.VIEW_COLORS[r.view], lw=0.5, marker="o", ms=1.5)
    ax.set_xticks([0, 1, 2], ["Old s3", "Old s6", best.replace("Merged ", "")], rotation=45, ha="right")
    ax.set_yscale("log"); ax.set_ylabel("Mean error per mouse (px)")
    # d: per keypoint, strict set
    k = []
    for name, (e, _, _) in models.items():
        d = e[e.labelled & e.frame.isin(sets["strict"])]
        k.append(d.groupby(["view", "bodypart"]).err_px.mean().rename(name))
    k = pd.concat(k, axis=1).reset_index()
    k.to_csv(OUT / "per_keypoint_strict.csv", index=False)
    sub = gs[1, 0:3].subgridspec(1, 2, width_ratios=[len(ts.VIEW_KEYPOINTS["side"]), len(ts.VIEW_KEYPOINTS["top"])],
                                 wspace=0.25)
    for j, view in enumerate(ts.VIEWS):
        ax = fig.add_subplot(sub[0, j])
        if j == 0:
            ts.panel_label(ax, "d")
        kv = k[k.view == view].set_index("bodypart").reindex(ts.VIEW_KEYPOINTS[view])
        w = 0.8 / len(names)
        for i, n in enumerate(names):
            ax.bar(np.arange(len(kv)) + (i - (len(names) - 1) / 2) * w, kv[n], width=w, color=colors[n], label=n)
        ax.set_xticks(np.arange(len(kv)), [b.replace("_", " ") for b in kv.index], rotation=60, ha="right")
        ax.set_yscale("log"); ax.set_title(f"{view.capitalize()} view", pad=2)
        if j == 0:
            ax.set_ylabel("Mean test error (px)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.12), ncol=len(names), fontsize=5)
    # e: derived quantities (whisker angle, jaw angle, pupil area), median absolute error, strict set
    ax = fig.add_subplot(gs[1, 3]); ts.panel_label(ax, "e")
    no_human = ts.human_reference().iloc[0:0]
    strict_frames = frames.assign(split=np.where(frames.frame.isin(sets["strict"]), "test", "other"))
    der = []
    for name, (e, _, _) in models.items():
        eb = e.assign(split=np.where(e.frame.isin(sets["strict"]), "test", "other"))
        d = ts.derived_errors(eb, strict_frames, gt_long, no_human).dropna(subset=["label", "model"])
        diff = d["model"] - d["label"]
        d["abs_err"] = np.where(d.quantity == "pupil_area", (100 * diff / d["label"]).abs(), diff.abs())
        der.append(d.assign(model_name=name))
    der = pd.concat(der)
    der.to_csv(OUT / "derived_strict.csv", index=False)
    med = der.groupby(["quantity", "model_name"]).abs_err.median().unstack().reindex(columns=names)
    labels = {"whisker_angle": "Whisker\nangle (deg)", "jaw_angle": "Jaw\nangle (deg)", "pupil_area": "Pupil\narea (%)"}
    qs = [q for q in labels if q in med.index]
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(np.arange(len(qs)) + (i - (len(names) - 1) / 2) * w, med.loc[qs, n], width=w, color=colors[n])
    ax.set_xticks(np.arange(len(qs)), [labels[q] for q in qs])
    ax.set_yscale("log"); ax.set_ylabel("Median absolute error")
    n_der = der.groupby("quantity").frame.nunique().to_dict()
    ax.set_title(", ".join(f"{q.split('_')[0]} n = {n_der.get(q, 0)}" for q in qs), pad=2, fontsize=5)
    med.to_csv(OUT / "derived_median_strict.csv")
    fig.savefig(OUT / "unseen_mice_benchmark.png", dpi=300)
    fig.savefig(OUT / "unseen_mice_benchmark.pdf")
    print(summary[summary.set == "strict"].round(2).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(med.round(2).to_string())


if __name__ == "__main__":
    main()
