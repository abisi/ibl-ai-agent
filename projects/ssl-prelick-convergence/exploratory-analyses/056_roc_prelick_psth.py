"""Average first-lick-aligned PSTHs of pre-lick-ROC significant vs non-significant neurons, learning vs expert.

Inputs: 051 per-session _roc_prelick_trials.npz (PSTH per unit x class, 10 ms bins, -600..+400 ms around the corrected
first lick) and 053 prelick_units.parquet (trial variant "all"; tested units: min FR 0.1 Hz, >= 3 trials per class).
Per unit: rate minus its mean rate in [-600, -400] ms (all classes pooled), 3-bin boxcar. Mean +- SEM over neurons.
Groups: two-class types: significant positive, significant negative, non-significant (tested);
        three-class: significant preferring WH / AH / FA (one-vs-rest), non-significant.
Figures -> combined_results_ks4/ssl-prelick-convergence/across_days/fa/psth/
  wholebrain_<type>.png  rows = cohort x stage, columns = groups; traces = WH (blue), AH (red), FA (grey)
  areas_<type>.png       rows = area groups, columns = cohort x stage; two-class: class-2 minus class-1 difference
                         trace for sig. positive (solid), sig. negative (dashed) and non-sig. (grey) neurons;
                         three-class: WH / AH / FA traces of significant (solid) and non-significant (thin dashed)
                         neurons. Panels with < MIN_N neurons left empty.
"""
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
OUT = m51.OUTROOT / "psth"
CCOL = {"WH": "#1f77b4", "AH": "#d62728", "FA": "#7f7f7f"}
COH = {"R+": "#00B400", "R-": "#C800C8"}
ROWS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
TYPES = {"wh_vs_aud_hit_prelick": ("WH", "AH", "WH vs AH (+ = AH > WH)"),
         "whisker_hit_vs_fa_prelick": ("FA", "WH", "FA vs WH (+ = WH > FA)"),
         "auditory_hit_vs_fa_prelick": ("FA", "AH", "FA vs AH (+ = AH > FA)")}
BASEW = (-0.6, -0.4)
MIN_N = 10


def load():
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    P, keys, edges = [], [], None
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        edges = z["psth_edges"]
        P.append(z["psth"].astype(np.float32))
        keys.append(pd.DataFrame(dict(session_id=r.session_id, electrode_group=z["electrode_group"].astype(str),
                                      cluster_id=z["cluster_id"].astype(str))))
    P = np.concatenate(P); K = pd.concat(keys, ignore_index=True); K["row"] = np.arange(len(K))
    tc = (edges[:-1] + np.diff(edges) / 2)
    b = (tc >= BASEW[0]) & (tc < BASEW[1])
    P = P - P[:, :, b].mean((1, 2), keepdims=True)
    P = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3) / 3, "same"), 2, P)
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    W = W.merge(K, on=["session_id", "electrode_group", "cluster_id"], how="inner")
    return W, P, tc * 1e3


def groups(W, t):
    s = W[f"sig:{t}@all"]
    tested = s.notna()
    sig = tested & (s.astype(float) == 1)
    if t == "three_class_prelick":
        pc = W["preferred_class@all"]
        return {"sig, prefers WH": sig & (pc == "WH"), "sig, prefers AH": sig & (pc == "AH"),
                "sig, prefers FA": sig & (pc == "FA"), "non-significant": tested & ~sig}
    sel = W[f"sel:{t}@all"]
    return {"sig, positive": sig & (sel > 0), "sig, negative": sig & (sel < 0), "non-significant": tested & ~sig}


def mean_sem(A):
    return A.mean(0), A.std(0) / np.sqrt(max(len(A), 1))


def deco(ax, tc):
    ax.axvspan(m51.PRELICK[0] * 1e3, m51.PRELICK[1] * 1e3, color="#FDD49E", alpha=0.55, lw=0)
    ax.axvline(0, color="k", lw=0.4); ax.axhline(0, color="0.75", lw=0.3)
    ax.set_xlim(tc[0], tc[-1])


def wholebrain(plt, W, P, tc, t):
    G = groups(W, t)
    fig, axs = plt.subplots(4, len(G), figsize=(7.4, 7.2), sharex=True, gridspec_kw=dict(hspace=0.5, wspace=0.3))
    for i, (c, s_) in enumerate(ROWS):
        m0 = ((W.cohort == c) & (W.stage == s_)).to_numpy()
        for j, (gname, gm) in enumerate(G.items()):
            ax = axs[i, j]; idx = W.row.to_numpy()[m0 & gm.to_numpy()]
            deco(ax, tc)
            if len(idx) >= MIN_N:
                for k, cl in enumerate(m51.CLASSES):
                    mu, se = mean_sem(P[idx, k])
                    ax.fill_between(tc, mu - se, mu + se, color=CCOL[cl], alpha=0.25, lw=0)
                    ax.plot(tc, mu, color=CCOL[cl], lw=0.9, label=cl)
            tot = (m0 & np.isin(np.arange(len(W)), np.where(sum(g.to_numpy() for g in G.values()))[0])).sum()
            ax.set_title(f"{gname}: n={len(idx)} ({100 * len(idx) / max(tot, 1):.0f}%)", fontsize=5.5)
            if j == 0:
                ax.set_ylabel(f"{c.replace('-', '−')} {s_}\nΔ rate (Hz)", color=COH[c], fontsize=6)
            if i == 3:
                ax.set_xlabel("ms from first lick")
    axs[0, 0].legend(frameon=False, fontsize=5, loc="upper left")
    fig.suptitle(f"{TYPES[t][2]}: first-lick PSTH of significant vs non-significant neurons (all units tested, all trials; "
                 "shaded = ROC window)", fontsize=6.8)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.07)
    fig.savefig(OUT / f"wholebrain_{t}.png", dpi=220); fig.savefig(OUT / f"wholebrain_{t}.pdf"); plt.close(fig)


def areas(plt, W, P, tc, t):
    import ephys_utilities.allen_utils.allen_utils as au
    order = [g for g in au.get_area_group_custom_order() if g in set(W.area_group.dropna())]
    G = groups(W, t)
    c1, c2, lab = TYPES[t]
    fig, axs = plt.subplots(len(order), 4, figsize=(7.4, 1.0 * len(order) + 0.8), sharex=True,
                            gridspec_kw=dict(hspace=0.55, wspace=0.3))
    for i, ag in enumerate(order):
        for j, (c, s_) in enumerate(ROWS):
            ax = axs[i, j]; deco(ax, tc)
            m0 = ((W.cohort == c) & (W.stage == s_) & (W.area_group == ag)).to_numpy()
            txt = []
            if c1 is not None:
                k1, k2 = m51.CLASSES.index(c1), m51.CLASSES.index(c2)
                for gname, ls, col in [("sig, positive", "-", COH[c]), ("sig, negative", "--", COH[c]), ("non-significant", "-", "0.55")]:
                    idx = W.row.to_numpy()[m0 & G[gname].to_numpy()]
                    if len(idx) >= MIN_N:
                        mu, se = mean_sem(P[idx, k2] - P[idx, k1])
                        ax.fill_between(tc, mu - se, mu + se, color=col, alpha=0.2, lw=0)
                        ax.plot(tc, mu, ls=ls, color=col, lw=0.8)
                    txt.append(f"{gname.split(', ')[-1][:3]} {len(idx)}")
            else:
                sig = m0 & sum(G[g].to_numpy() for g in G if g != "non-significant").astype(bool)
                for gm, ls, lw in [(sig, "-", 0.9), (m0 & G["non-significant"].to_numpy(), "--", 0.5)]:
                    idx = W.row.to_numpy()[gm]
                    if len(idx) >= MIN_N:
                        for k, cl in enumerate(m51.CLASSES):
                            ax.plot(tc, P[idx, k].mean(0), ls=ls, lw=lw, color=CCOL[cl])
                    txt.append(f"{'sig' if lw > 0.6 else 'n.s.'} {len(idx)}")
            ax.set_title(", ".join(txt), fontsize=4.4)
            ax.tick_params(labelsize=4.5)
            if j == 0:
                ax.set_ylabel(ag.replace(" areas", "").replace("Somatosensory", "SS").replace("Lateral septal complex", "LSX")
                              .replace("Amygdala and hypothalamus", "Amyg./hypoth."), fontsize=5, rotation=0, ha="right",
                              va="center")
            if i == 0:
                ax.text(0.5, 1.45, f"{c.replace('-', '−')} {s_}", transform=ax.transAxes, ha="center", color=COH[c],
                        fontsize=6.5, weight="bold")
            if i == len(order) - 1:
                ax.set_xlabel("ms from first lick", fontsize=5)
    if c1 is not None:
        sub = f"{c2} − {c1} rate difference (Hz): solid = sig. positive, dashed = sig. negative, grey = non-significant"
    else:
        sub = "WH (blue), AH (red), FA (grey) Δ rate: solid = three-class significant, dashed = non-significant"
    fig.suptitle(f"{lab} by area group\n{sub}", fontsize=6.5, y=0.998)
    fig.subplots_adjust(left=0.13, right=0.98, top=1 - 1.05 / (1.0 * len(order) + 0.8), bottom=0.04)
    fig.savefig(OUT / f"areas_{t}.png", dpi=220); fig.savefig(OUT / f"areas_{t}.pdf"); plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                         "xtick.labelsize": 5, "ytick.labelsize": 5, "pdf.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    W, P, tc = load()
    print(f"{len(W)} units with PSTH", flush=True)
    for t in TYPES:
        wholebrain(plt, W, P, tc, t)
        areas(plt, W, P, tc, t)
        print("saved", t, flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
