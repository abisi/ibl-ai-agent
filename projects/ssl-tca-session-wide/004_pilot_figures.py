"""Pilot figures: rank selection (train error, held-out error, replicate similarity) per epoch x normalisation, and
per-session factor pages (neuron factor by area, time factor, trial factor over the session with trial type, lick and
the learning curves). Also a first per-component table: Spearman correlation of the active whisker-trial factor with
the learning curve and with trial index (no null yet; that is 005).

Run: python 004_pilot_figures.py [--rank 6] [--norm soft]
     -> OUT/figures/pilot/rank_selection.png|pdf, OUT/figures/pilot/factors_<norm>_<epoch>_r<rank>.pdf,
        OUT/tables/pilot_component_corr_<norm>_r<rank>.csv
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import stats  # noqa: E402

import tca_lib as L  # noqa: E402

F = importlib.import_module("003_tca_fit")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False})
TT_COL = {"whisker_trial": "#f7b519", "auditory_trial": "#2c2cdb", "no_stim_trial": "#262626"}
COH_COL = {"R+": "#00B400", "R-": "#C800C8"}
FIG = L.OUT / "figures" / "pilot"
JAW = ["jaw", "jaw_all", "jaw_b50"]            # 003b_tca_jaw (bsub only)
ALL_EPOCHS = list(F.EPOCHS) + JAW


def area_colours(areas):
    from ephys_utilities.allen_utils import allen_utils as A
    grp = {k.strip(): v for k, v in A.get_custom_area_groups_from_name().items()}
    col = A.get_custom_area_groups_colors()
    order = A.get_area_group_custom_order()
    g = [grp.get(a, None) for a in areas]
    missing = sorted({a for a, x in zip(areas, g) if x is None})
    if missing:
        raise KeyError(f"areas without group: {missing}")
    return [col[x] for x in g], [order.index(x) for x in g], g


def shade_passive(ax, t, x):
    """grey span per contiguous run of passive trials (passive_pre / passive_post)"""
    m = t.epoch.isin(["passive_pre", "passive_post"]).to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    for a0, a1 in zip(edges[::2], edges[1::2]):
        ax.axvspan(x[a0] - 0.5, x[a1 - 1] + 0.5, color="0.92", lw=0, edgecolor="none", zorder=-1)


def load_fit(norm, epoch, sid):
    f = F.TCA_DIR / norm / epoch / f"{sid}.npz"
    return dict(np.load(f)) if f.exists() else None


def rank_selection(sessions):
    fig, ax = plt.subplots(3 * len(F.NORMS), len(ALL_EPOCHS), figsize=(9.0, 2.0 * 3 * len(F.NORMS) / 1.6),
                           squeeze=False)
    for i, norm in enumerate(F.NORMS):
        for j, epoch in enumerate(ALL_EPOCHS):
            rows = {"train": [], "test": [], "sim": []}
            for r in sessions.itertuples():
                d = load_fit(norm, epoch, r.session_id)
                if d is None:
                    continue
                c = COH_COL[r.cohort]
                tr = [d[f"r{k}_obj"].min() for k in F.RANKS]
                te = [d[f"r{k}_cv"][:, 0].mean() for k in F.RANKS]
                si = [d[f"r{k}_sim"].mean() if len(d[f"r{k}_sim"]) else 1 for k in F.RANKS]
                for k, v in zip(("train", "test", "sim"), (tr, te, si)):
                    ax[3 * i + ["train", "test", "sim"].index(k), j].plot(list(F.RANKS), v, color=c, lw=0.5, alpha=0.6,
                                                                         ls="-" if r.stage == "learning" else ":")
                    rows[k].append(v)
            for k, lab in zip(("train", "test", "sim"),
                              ("Relative error, all entries", "Relative error, held-out fibres",
                               "Similarity to best fit")):
                a = ax[3 * i + ["train", "test", "sim"].index(k), j]
                if rows[k]:
                    a.plot(list(F.RANKS), np.nanmean(rows[k], 0), color="k", lw=1.2)
                a.set_xlabel("Rank (components)")
                a.set_ylabel(lab)
                a.set_title(f"{epoch}, {norm}" if k == "train" else "", fontsize=6)
    ax[0, 0].plot([], [], color=COH_COL["R+"], label="R+")
    ax[0, 0].plot([], [], color=COH_COL["R-"], label="R-")
    ax[0, 0].plot([], [], color="grey", ls=":", label="expert session")
    ax[0, 0].plot([], [], color="k", lw=1.2, label="mean")
    ax[0, 0].legend(frameon=False, fontsize=5)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"rank_selection.{ext}", dpi=300)
    plt.close(fig)


def factor_page(pdf, r, norm, epoch, rank, rows_out):
    d = load_fit(norm, epoch, r.session_id)
    if d is None:
        return
    t = pd.read_parquet(L.TENSORS / f"{r.session_id}_trials.parquet")
    if "trials_kept" in d:   # jaw-aligned fits use a subset of trials
        t = t.set_index("trial_index").loc[d["trials_kept"]].reset_index()
        t["jaw_rt_ms"] = d["jaw_onset"] * 1e3
    u = pd.read_parquet(L.TENSORS / f"{r.session_id}_units.parquet").iloc[d["units_kept"]].reset_index(drop=True)
    A, B, C = d[f"r{rank}_f0"], d[f"r{rank}_f1"], d[f"r{rank}_f2"]   # trials, units, time
    ucol, uord, _ = area_colours(u.area_acronym_custom.str.strip().tolist())
    o = np.lexsort((u.area_acronym_custom.to_numpy(), np.array(uord)))
    x = t.trial_index.to_numpy()
    fig, ax = plt.subplots(rank + 1, 3, figsize=(7.4, 1.0 * (rank + 1)), gridspec_kw={"width_ratios": [1.2, 0.8, 3]})
    # behaviour row
    b = ax[0, 2]
    for c, lab, col in (("p_wh_i", "whisker", TT_COL["whisker_trial"]), ("p_au_i", "auditory", TT_COL["auditory_trial"]),
                        ("p_ns_i", "no stim (false alarm)", TT_COL["no_stim_trial"])):
        if c in t:
            b.plot(x, t[c], color=col, lw=1, label=lab)
    shade_passive(b, t, x)
    b.set_ylabel("P(lick), curve")
    b.set_ylim(-0.02, 1.02)
    b.legend(frameon=False, fontsize=5, ncol=3, loc="lower right")
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")
    ax[0, 0].text(0, 0.5, f"{r.session_id}\n{r.cohort}, {r.stage}, {r.learning_category}\n{epoch}, {norm}, rank {rank}\n"
                  f"{len(t)} trials, {len(u)} units", fontsize=6, va="center")
    tc = d["t_centres"]
    act_w = ((t.epoch == "active") & (t.trial_type == "whisker_trial")).to_numpy()
    for k in range(rank):
        a = ax[k + 1, 0]
        a.bar(np.arange(len(o)), B[o, k], color=np.array(ucol)[o], width=1.0, lw=0)
        a.set_ylabel(f"C{k + 1} unit weight")
        a.set_xticks([])
        a = ax[k + 1, 1]
        a.plot(tc * 1e3, C[:, k], color="k", lw=0.8)
        a.axvline(0, color="grey", lw=0.4, ls="--")
        if "jaw_onset" in d:   # median stimulus time relative to jaw onset
            a.axvline(-np.median(d["jaw_onset"]) * 1e3, color="tab:red", lw=0.4, ls=":")
        a.set_ylabel("Time weight")
        a = ax[k + 1, 2]
        for tt_, col in TT_COL.items():
            m = (t.trial_type == tt_).to_numpy()
            lk = t.lick_flag.to_numpy() == 1
            a.scatter(x[m & lk], A[m & lk, k], s=3, color=col, lw=0)
            a.scatter(x[m & ~lk], A[m & ~lk, k], s=3, facecolor="none", edgecolor=col, lw=0.4)
        shade_passive(a, t, x)
        a.axhline(0, color="grey", lw=0.3)
        a.set_ylabel("Trial weight")
        # quick stats on active whisker trials
        if act_w.sum() > 10 and t.loc[act_w, "p_wh"].notna().sum() > 10:
            ok = act_w & t.p_wh.notna().to_numpy()
            rc = stats.spearmanr(A[ok, k], t.p_wh.to_numpy()[ok])[0]
            ri = stats.spearmanr(A[ok, k], x[ok])[0]
            res_a = A[ok, k] - np.polyval(np.polyfit(x[ok], A[ok, k], 1), x[ok])
            res_p = t.p_wh.to_numpy()[ok] - np.polyval(np.polyfit(x[ok], t.p_wh.to_numpy()[ok], 1), x[ok])
            rp = stats.spearmanr(res_a, res_p)[0]
            a.set_title(f"whisker active: rho(curve) {rc:.2f}, rho(trial) {ri:.2f}, partial {rp:.2f}", fontsize=5,
                        loc="right")
            rows_out.append(dict(session_id=r.session_id, cohort=r.cohort, stage=r.stage, learner=r.learner,
                                 epoch=epoch, norm=norm, rank=rank, comp=k + 1, rho_curve=rc, rho_trial=ri,
                                 rho_partial=rp, peak_ms=float(tc[np.argmax(C[:, k])] * 1e3)))
    ax[-1, 0].set_xlabel("Units (sorted by area)")
    ax[-1, 1].set_xlabel("Time from jaw onset (ms)" if epoch in JAW else "Time from stimulus (ms)")
    ax[-1, 2].set_xlabel("Trial in session (passive blocks shaded)")
    ax[1, 2].scatter([], [], s=3, color="k", label="lick")
    ax[1, 2].scatter([], [], s=3, facecolor="none", edgecolor="k", lw=0.4, label="no lick")
    ax[1, 2].legend(frameon=False, fontsize=5, loc="upper right")
    fig.tight_layout()
    pdf.savefig(fig, dpi=200)
    png = FIG / "png" / f"factors_{norm}_{epoch}_r{rank}"
    png.mkdir(parents=True, exist_ok=True)
    fig.savefig(png / f"{r.cohort.replace('+', 'plus').replace('-', 'minus')}_{r.stage}_{r.session_id}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--norm", default="bsub")
    ap.add_argument("--epochs", nargs="*")
    a = ap.parse_args()
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv")
    s = s[s.pilot].sort_values(["stage", "cohort", "learning_category"], ascending=[False, True, True])
    rank_selection(s)
    rows = []
    eps = a.epochs or [e for e in ALL_EPOCHS if not (a.norm == "bsub" and e == "baseline")]
    for epoch in eps:
        with PdfPages(FIG / f"factors_{a.norm}_{epoch}_r{a.rank}.pdf") as pdf:
            for r in s.itertuples():
                factor_page(pdf, r, a.norm, epoch, a.rank, rows)
    pd.DataFrame(rows).to_csv(L.OUT / "tables" / f"pilot_component_corr_{a.norm}_r{a.rank}.csv", index=False)


if __name__ == "__main__":
    main()
