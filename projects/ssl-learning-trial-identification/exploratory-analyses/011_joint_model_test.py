"""011 -- Test of the joint whisker + false-alarm learning models (lt_joint.py) on a few
sessions, with annotated figures (user request 2026-09-25: "Make a test for a few sessions of
that with all details and annotations on figures").

Models (see lt_joint.py docstring):
  CONTINUOUS  logit P(no-stim lick) = g_t;  logit P(whisker lick) = g_t + d_t;
              g (general lick propensity) and d (whisker-specific drive) independent random walks,
              step sizes integrated over a grid; exact 2-D grid forward-backward.
  SWITCHING   g random walk; d jumps once from d_naive to d_learned (uniform prior on the switch
              trial; R+ d rises, R- d falls, by >= 0.5 logit); exact grid over (g, state) with
              d_naive, d_learned, sigma_g integrated on grids; Bayes factor vs no switch.
Learning trials derived (whisker-trial index, same as the stored LT):
  LT_cont    R+: first whisker trial with P(d > DELTA) >= 0.9 on >= 16 of the next 20 whisker
             trials; R-: after the first whisker trial with P(d > DELTA) >= 0.9, first with
             P(d > DELTA) <= 0.5 on >= 16 of the next 20. DELTA = 1 logit (whisker licking at
             least e^1 = 2.7x the odds of an FA lick).
  LT_switch  posterior median of the switch trial (first whisker trial at/after the switch),
             90% credible interval, log10 Bayes factor vs no switch.
Sessions: the same 6 examples as 008 (one per outcome type).
Validation of the grid continuous model against PyMC: 011b_joint_pymc_check.py.
Outputs: artifacts/011_joint_fits.pkl, exploratory-analyses/011_joint_model_test.png/.pdf,
         011_joint_model_summary.csv
Run on haas (6 sessions in parallel, ~1-2 min each).
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lt_joint as J  # noqa: E402

ART = HERE.parent / "artifacts"
EXAMPLES = [("AB119", "R+ learner (stored 14 -> L6 49)"), ("MH028", "R+ learns by FA dropping"),
            ("AB082", "R+ early peak then decline"), ("AB107", "R+ no learning event"),
            ("AB085", "R- gradual learner"), ("AB139", "R- never licked above FA")]
DELTA = 1.0
DP_LEARNED, DP_UNLEARNED = 0.2, 0.1   # probability-scale discrimination thresholds (whisker - FA lick prob)
W, K = 20, 16
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def build_sequence(d):
    t = np.r_[d["w_start"], d["n_start"]]
    y = np.r_[d["w_outcomes"], d["n_outcomes"]].astype(int)
    is_w = np.r_[np.ones(len(d["w_start"]), bool), np.zeros(len(d["n_start"]), bool)]
    o = np.argsort(t, kind="stable")
    return t[o], y[o], is_w[o]


def lt_from_d(p_above_w, rg):
    n = len(p_above_w)
    if rg == 1:
        for t in range(n - W + 1):
            if p_above_w[t] >= 0.9 and (p_above_w[t:t + W] >= 0.9).sum() >= K:
                return float(t)
        return np.nan
    first = np.where(p_above_w >= 0.9)[0]
    if len(first) == 0:
        return np.nan
    for t in range(first[0], n - W + 1):
        if (p_above_w[t:t + W] <= 0.5).sum() >= K:
            return float(t)
    return np.nan


def lt_from_dp(marg_dp_w, rg):
    """Probability-scale version: R+ first whisker trial with P(dp > 0.2) >= 0.9 sustained 16/20;
    R- after the first with P(dp > 0.2) >= 0.9, first with P(dp > 0.1) <= 0.5 sustained 16/20."""
    p_hi = marg_dp_w[:, J.DP_GRID > DP_LEARNED].sum(1)
    p_lo = marg_dp_w[:, J.DP_GRID > DP_UNLEARNED].sum(1)
    n = len(p_hi)
    if rg == 1:
        for t in range(n - W + 1):
            if p_hi[t] >= 0.9 and (p_hi[t:t + W] >= 0.9).sum() >= K:
                return float(t), p_hi
        return np.nan, p_hi
    first = np.where(p_hi >= 0.9)[0]
    if len(first) == 0:
        return np.nan, p_lo
    for t in range(first[0], n - W + 1):
        if (p_lo[t:t + W] <= 0.5).sum() >= K:
            return float(t), p_lo
    return np.nan, p_lo


def process(item):
    sid, d = item
    rg = 1 if d["reward_group"] == "R+" else 0
    t, y, is_w = build_sequence(d)
    cont = J.fit_continuous(y, is_w)
    sw = J.fit_switching(y, is_w, rg)
    return sid, dict(t=t, y=y, is_w=is_w, cont=cont, sw=sw, rg=rg)


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    lt = pd.read_csv(ART / "007_learning_trials_v2.csv").set_index("session_id")
    sel = [(next(k for k in inputs if k.startswith(p)), lab) for p, lab in EXAMPLES]
    out_path = ART / "011_joint_fits.pkl"
    if out_path.exists():
        fits = pickle.load(open(out_path, "rb"))
    else:
        with ProcessPoolExecutor(max_workers=len(sel)) as ex:
            fits = dict(ex.map(process, [(s, inputs[s]) for s, _ in sel]))
        pickle.dump(fits, open(out_path, "wb"))

    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    rows = []
    fig, axes = plt.subplots(len(sel), 5, figsize=(29, 4.0 * len(sel)), constrained_layout=True, squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.15, 1.15, 0.55]))
    for r, (sid, lab) in enumerate(sel):
        f, d = fits[sid], inputs[sid]
        rg_name = d["reward_group"]
        col = COHORT_COLOR[rg_name]
        is_w, y = f["is_w"], f["y"]
        xw = np.cumsum(is_w) - 1.0                     # whisker-trial index axis for all trials
        xw[~is_w] += 0.5                               # no-stim trials sit between whisker trials
        wi = np.where(is_w)[0]
        (g10, g50, g90), _ = J.marg_summary(f["cont"]["marg_g"], J.G_GRID)
        (d10, d50, d90), dmean = J.marg_summary(f["cont"]["marg_d"], J.D_GRID)
        (s10, s50, s90), _ = J.marg_summary(f["cont"]["marg_s"], J.S_GRID)
        p_above = f["cont"]["marg_d"][:, J.D_GRID > DELTA].sum(1)
        lt_cont = lt_from_d(p_above[wi], f["rg"])
        lt_cont_p, p_dp = lt_from_dp(f["cont"]["marg_dp"][wi], f["rg"])
        (dp10, dp50, dp90), _ = J.marg_summary(f["cont"]["marg_dp"], J.DP_GRID)
        psw = f["sw"]["p_switch"]
        cdf = np.cumsum(psw) / max(psw.sum(), 1e-12)
        k_med, k_lo, k_hi = (int(np.searchsorted(cdf, q)) for q in (0.5, 0.05, 0.95))
        to_w = lambda k: float(np.searchsorted(wi, k))  # noqa: E731  first whisker trial at/after all-trial index k
        lt_sw, lt_sw_lo, lt_sw_hi = to_w(k_med), to_w(k_lo), to_w(k_hi)
        row = lt.loc[sid]
        rows.append(dict(session_id=sid, reward_group=rg_name, label=lab, n_whisker=int(is_w.sum()), n_nostim=int((~is_w).sum()),
                         stored_lt=row.L0_stored, L6=row.L6, lt_lenient_clean=row.lt_lenient_clean, lt_cont=lt_cont, lt_cont_prob=lt_cont_p,
                         lt_switch=lt_sw, lt_switch_ci05=lt_sw_lo, lt_switch_ci95=lt_sw_hi, switch_log10_bf=f["sw"]["log10_bf"],
                         d_naive=f["sw"]["d_naive"], d_learned=f["sw"]["d_learned"], sigma_g=f["cont"]["sigma_g_mean"],
                         sigma_d=f["cont"]["sigma_d_mean"]))
        lts = [(row.L0_stored, "#d62728", "-", "stored"), (row.L6, "#1f77b4", "-", "L6 (step, joint CP)"),
               (row.lt_lenient_clean, "#ff7f0e", "--", "lenient clean"), (lt_cont, "#2ca02c", "-.", "joint cont. (logit d)"),
               (lt_cont_p, "#17becf", "-", "joint cont. (prob. dp)"),
               (lt_sw, "#9467bd", "-", "joint switching")]

        # A raw data
        ax = axes[r, 0]
        for mask, c_, lab_, y0 in ((is_w, col, "whisker", 1.08), (~is_w, "#666666", "no-stim (FA)", -0.08)):
            xs, ys = xw[mask], y[mask]
            ax.scatter(xs[ys == 1], np.full((ys == 1).sum(), y0), s=5, marker="|", color=c_)
            run = pd.Series(ys.astype(float)).rolling(10, center=True, min_periods=4).mean()
            ax.plot(xs, run, color=c_, lw=1.3, label=f"{lab_} lick rate (running 10 trials)")
        for val, c_, ls, name in lts:
            if not pd.isna(val):
                ax.axvline(val, color=c_, ls=ls, lw=1.3)
        ax.set_ylim(-0.15, 1.15)
        ax.set_ylabel(f"{sid[:14]}\n{rg_name} -- {lab}", fontsize=8)
        ax.set_title(f"A. RAW DATA: licks per trial type (ticks), running rates\n{int(is_w.sum())} whisker, "
                     f"{int((~is_w).sum())} no-stim trials; vertical lines = learning trials (legend in E)", fontsize=8, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="center right")

        # B model curves vs separate fits
        ax = axes[r, 1]
        ax.fill_between(xw, expit(g10), expit(g90), color="#666666", alpha=0.2, lw=0)
        ax.plot(xw, expit(g50), color="#333333", lw=1.5, label="P(FA lick) = logistic(g), joint model")
        ax.fill_between(xw[wi], expit(s10[wi]), expit(s90[wi]), color=col, alpha=0.25, lw=0)
        ax.plot(xw[wi], expit(s50[wi]), color=col, lw=1.8, label="P(whisker lick) = logistic(g + d), joint model")
        e = curves[sid]["eb"]
        ax.plot(np.arange(len(e["p_mean"])), e["p_mean"], color=col, lw=0.8, ls=":", label="whisker, separate fit (002)")
        ax.plot(np.arange(len(e["fa_time"])), e["fa_time"], color="#333333", lw=0.8, ls=":", label="FA, separate fit (002)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"B. JOINT CONTINUOUS MODEL, lick probabilities (80% bands)\nsigma_g = {f['cont']['sigma_g_mean']:.2f}, "
                     f"sigma_d = {f['cont']['sigma_d_mean']:.2f} logit/trial (posterior means); dotted = separate fits",
                     fontsize=8, loc="left")
        ax.legend(fontsize=6.3, frameon=False, loc="best")

        # C d posterior (logit) and dp posterior (probability scale)
        ax = axes[r, 2]
        ax.fill_between(xw[wi], dp10[wi], dp90[wi], color="#17becf", alpha=0.2, lw=0)
        ax.plot(xw[wi], dp50[wi], color="#17becf", lw=1.4, label="dp = P(whisker) - P(FA) (median, 80%)")
        ax.axhline(DP_LEARNED, color="#17becf", ls=":", lw=0.8)
        if not pd.isna(lt_cont_p):
            ax.axvline(lt_cont_p, color="#17becf", lw=1.6)
        ax.fill_between(xw, d10, d90, color="#8c564b", alpha=0.25, lw=0)
        ax.plot(xw, d50, color="#8c564b", lw=1.6, label="d = whisker-specific drive (median, 80% band)")
        ax.axhline(DELTA, color="#8c564b", ls="--", lw=0.8)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_ylabel("d (logit)  /  dp (probability)", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(xw[wi], p_above[wi], color="#2ca02c", lw=1.1, label=f"P(d > {DELTA:g})")
        ax2.axhline(0.9, color="#2ca02c", ls=":", lw=0.8)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel(f"P(d > {DELTA:g})", fontsize=8, color="#2ca02c")
        if not pd.isna(lt_cont):
            ax.axvline(lt_cont, color="#2ca02c", ls="-.", lw=1.6)
        rule_rm = "" if f["rg"] == 1 else f"; R-: then P(dp>{DP_UNLEARNED:g})<=0.5"
        ax.set_title(f"C. WHISKER-SPECIFIC DRIVE: d (logit, brown) and dp = P(whisker)-P(FA) (cyan)\nLT from d: {lt_cont:.0f} "
                     f"(P(d>{DELTA:g})>=0.9); LT from dp: {lt_cont_p:.0f} (P(dp>{DP_LEARNED:g})>=0.9{rule_rm}); sustained 16/20"
                     .replace("nan", "none"), fontsize=8, loc="left")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper left")

        # D switching
        ax = axes[r, 3]
        ax.plot(xw, f["sw"]["p_learned"], color="#9467bd", lw=1.6, label="P(learned state)")
        ax.bar(xw, psw / max(psw.max(), 1e-12), width=1.0, color="#9467bd", alpha=0.3, label="switch-trial posterior (scaled)")
        ax.axvspan(lt_sw_lo, lt_sw_hi, color="#9467bd", alpha=0.08)
        ax.axvline(lt_sw, color="#9467bd", lw=1.6)
        ax.set_ylim(-0.02, 1.05)
        ev = "strong" if f["sw"]["log10_bf"] > 1 else ("substantial" if f["sw"]["log10_bf"] > 0.5 else "weak / none")
        ax.set_title(f"D. JOINT SWITCHING MODEL: d jumps once, g drifts\nswitch = {lt_sw:.0f} [{lt_sw_lo:.0f}, {lt_sw_hi:.0f}], "
                     f"log10 BF vs no switch = {f['sw']['log10_bf']:+.1f} ({ev}); d {f['sw']['d_naive']:+.1f} -> "
                     f"{f['sw']['d_learned']:+.1f}", fontsize=8, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="center right")

        # E sigma posterior + legend of LTs
        ax = axes[r, 4]
        im = ax.imshow(f["cont"]["sigma_weights"], origin="lower", cmap="Greys", aspect="auto")
        tick = [f"{s:.2f}" for s in J.SIGMAS]
        ax.set_xticks(range(len(J.SIGMAS)), tick, fontsize=6, rotation=45)
        ax.set_yticks(range(len(J.SIGMAS)), tick, fontsize=6)
        ax.set_xlabel("sigma_d", fontsize=7)
        ax.set_ylabel("sigma_g", fontsize=7)
        ax.set_title("E. step-size posterior\n(continuous model)", fontsize=8, loc="left")
        txt = "\n".join(f"{name}: {val:.0f}".replace("nan", "none") for val, _, _, name in lts)
        ax.text(1.05, 0.5, "learning trials\n(whisker index)\n" + txt, transform=ax.transAxes, fontsize=7, va="center")
        for a in axes[r, :4]:
            a.set_xlim(-1, is_w.sum())
            a.tick_params(labelsize=7)
            a.spines[["top"]].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    for a in axes[-1, :4]:
        a.set_xlabel("whisker trial (index used by learning_trial); no-stim trials placed between", fontsize=8)
    fig.suptitle("Joint whisker + false-alarm models on 6 sessions: logit P(no-stim lick) = g (general lick propensity), "
                 "logit P(whisker lick) = g + d (d = whisker-specific drive). Learning = change in d only.\nLine colours: red "
                 "stored LT, blue L6 step change point, orange lenient clean LT, green joint-continuous LT, purple "
                 "joint-switching LT", fontsize=11)
    fig.savefig(HERE / "011_joint_model_test.png", dpi=160)
    fig.savefig(HERE / "011_joint_model_test.pdf")
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "011_joint_model_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
