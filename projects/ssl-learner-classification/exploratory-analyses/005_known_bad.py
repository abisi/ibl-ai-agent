"""005 -- What isolates the R+ mice the user knows are non-learners? (user 2026-11-23: "how to isolate the few mice in R+ that
I know are bad?"). Known bad = the fixed non-learners (MH020, MH013, AB142, MH068, AB151, AB117, AB107); the flexible sheet-bad
mice (AB141, MH007, AB104, MH026, MH009) are shown separately and NOT used as either class.

Per day-0 feature (001 per-mouse table, 004 whisker-only statistics, 037 learning-trial definitions as "LT defined"):
AUC of known bad vs the other R+ mice (good / moderate; direction-free), the best threshold (balanced accuracy), how many
good / moderate mice fall on the bad side (confusers, named), and whether the feature separates perfectly. Then the best
pair of features (2-feature threshold AND / OR rules), leave-one-out.
Outputs: tables/005_known_bad_features.csv, tables/005_known_bad_pairs.csv, figures/005_known_bad.png|pdf.
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-learner-classification/exploratory-analyses/005_known_bad.py
"""
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

RES = axel_bisi_root() / "combined_results_ks4" / "ssl-learner-classification"
LT = REPO / "projects" / "ssl-learning-trial-identification" / "artifacts" / "037_methods_lt.csv"
FIXED_BAD = ["MH020", "MH013", "AB142", "MH068", "AB151", "AB117", "AB107"]
FLEX = ["AB141", "MH007", "AB104", "MH026", "MH009"]
FEATS = {"hit_w": "W P(lick) day 0", "fa": "FA P(lick) day 0", "dprime": "d' day 0", "gap_mean": "W - chance curve, mean",
         "gap_max": "W - chance curve, max", "gap_last20": "W - chance, last 20", "frac_above": "Frac. W trials > chance",
         "run_above": "Longest run > chance", "half2_gap": "W - FA, 2nd half", "late30_gap": "W - FA, last 30",
         "p_whisker_max": "Max W curve", "peak_dprime": "Peak d'", "end_dprime": "d', last 20",
         "w_half2": "W P(lick), 2nd half", "w_sustained": "Sustained W P(lick) (20)", "frac_w_high": "Frac. W > R- level",
         "first_hit": "W trials before 1st hit", "n_whisker": "W trials", "rt_w": "W RT", "hit_a": "A P(lick) day 0",
         "consensus": "9-criteria consensus"}


def best_threshold(x, y):
    """y True = known bad; best balanced accuracy over thresholds and both sides"""
    best = (-1, None, None)
    for th in np.unique(x):
        for side in ("<=", ">="):
            pred = x <= th if side == "<=" else x >= th
            ba = 0.5 * (pred[y].mean() + (~pred[~y]).mean())
            if ba > best[0]:
                best = (ba, th, side)
    return best


def main():
    d = pd.read_csv(RES / "tables" / "004_rminus_rule_per_mouse.csv")   # R+ mice, 001 features + 004 statistics
    lt = pd.read_csv(LT) if LT.exists() else None
    feats = dict(FEATS)
    if lt is not None:
        for k in [c for c in lt.columns if c not in ("session_id", "cohort", "learning_category", "n_whisker")]:
            d = d.merge(lt[["session_id", k]].rename(columns={k: f"LT_{k}"}), on="session_id", how="left")
            d[f"has_{k}"] = d[f"LT_{k}"].notna().astype(float)
            feats[f"has_{k}"] = f"LT defined: {k}"
    d["group"] = np.where(d.mouse_id.isin(FIXED_BAD), "known bad", np.where(d.mouse_id.isin(FLEX), "flexible",
                                                                          d.learning_category))
    t = d[d.group != "flexible"].copy()
    y = (t.group == "known bad").to_numpy()
    rows = []
    for f, lab in feats.items():
        x = t[f].to_numpy(float)
        ok = np.isfinite(x)
        if ok.sum() < 10 or y[ok].sum() < 3:
            continue
        auc = roc_auc_score(y[ok], x[ok])
        ba, th, side = best_threshold(x[ok], y[ok])
        pred = (x <= th) if side == "<=" else (x >= th)
        conf = t[ok & pred & ~y].mouse_id.tolist()
        missed = t[ok & ~pred & y].mouse_id.tolist()
        xb, xo = x[ok & y], x[ok & ~y]
        perfect = (xb.max() < xo.min()) or (xb.min() > xo.max())
        rows.append(dict(feature=f, label=lab, auc=max(auc, 1 - auc), direction="low" if auc < 0.5 else "high",
                         threshold=th, side=side, balanced_acc=ba, n_confusers=len(conf), confusers=",".join(conf),
                         missed=",".join(missed), perfect=perfect, n=int(ok.sum())))
    F = pd.DataFrame(rows).sort_values(["balanced_acc", "auc"], ascending=False)
    F.to_csv(RES / "tables" / "005_known_bad_features.csv", index=False)
    # pairs of the top continuous features: AND / OR of the two single thresholds, leave-one-out re-fit
    top = [f for f in F[~F.feature.str.startswith("has_")].feature.head(10)]
    prow = []
    for a, b in itertools.combinations(top, 2):
        xa, xb_ = t[a].to_numpy(float), t[b].to_numpy(float)
        ok = np.isfinite(xa) & np.isfinite(xb_)
        for op in ("and", "or"):
            def fit(idx):
                ra, rb = best_threshold(xa[idx], y[idx]), best_threshold(xb_[idx], y[idx])
                return ra, rb

            def pred_of(ra, rb, i):
                pa = xa[i] <= ra[1] if ra[2] == "<=" else xa[i] >= ra[1]
                pb = xb_[i] <= rb[1] if rb[2] == "<=" else xb_[i] >= rb[1]
                return (pa & pb) if op == "and" else (pa | pb)
            idx = np.flatnonzero(ok)
            ra, rb = fit(idx)
            p = pred_of(ra, rb, idx)
            ba = 0.5 * (p[y[idx]].mean() + (~p[~y[idx]]).mean())
            loo = []
            for i in idx:
                j = idx[idx != i]
                ra_, rb_ = fit(j)
                loo.append(pred_of(ra_, rb_, np.array([i]))[0] == y[i])
            prow.append(dict(a=a, b=b, op=op, balanced_acc=ba, loo_acc=float(np.mean(loo)),
                             confusers=",".join(t.mouse_id.to_numpy()[idx][p & ~y[idx]]),
                             missed=",".join(t.mouse_id.to_numpy()[idx][~p & y[idx]])))
    Pp = pd.DataFrame(prow).sort_values(["balanced_acc", "loo_acc"], ascending=False)
    Pp.to_csv(RES / "tables" / "005_known_bad_pairs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(F.drop(columns=["label"]).head(25).round(3).to_string(index=False))
    print(Pp.head(12).round(3).to_string(index=False))
    # flexible mice on the best features
    show = F[~F.feature.str.startswith("has_")].feature.head(6).tolist()
    print(d[d.group == "flexible"][["mouse_id"] + show].round(2).to_string(index=False))

    fig, axs = plt.subplots(1, len(show), figsize=(7.4, 2.3), gridspec_kw=dict(wspace=0.7))
    col = {"known bad": "k", "flexible": "tab:orange", "moderate": "#7fd17f", "good": "#00B400"}
    for ax, f in zip(axs, show):
        for i, g in enumerate(["good", "moderate", "flexible", "known bad"]):
            v = d[d.group == g]
            x = i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
            ax.scatter(x, v[f], s=6, color=col[g], lw=0)
            if g in ("known bad", "flexible"):
                for xx, yy, m in zip(x, v[f], v.mouse_id):
                    ax.annotate(m, (xx, yy), fontsize=3.5, xytext=(2, 0), textcoords="offset points")
        r = F[F.feature == f].iloc[0]
        ax.axhline(r.threshold, color="tab:red", lw=0.6, ls="--")
        ax.set_xticks(range(4), ["good", "mod.", "flex.", "bad"], fontsize=5)
        ax.set_title(f"{FEATS.get(f, f)}\nAUC {r.auc:.2f}, BA {r.balanced_acc:.2f}", fontsize=5.5)
        ax.tick_params(labelsize=5)
    fig.suptitle("R+ day 0: features that isolate the known non-learners (black) from good / moderate mice; flexible "
                 "mice (orange) not used in the fit; dashed = best threshold.", fontsize=6, y=1.1)
    for ext in ("png", "pdf"):
        fig.savefig(RES / "figures" / f"005_known_bad.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
