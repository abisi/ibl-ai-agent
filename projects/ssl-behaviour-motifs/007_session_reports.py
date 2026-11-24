"""007 -- one PDF report per pilot session (haas): both views side by side, three methods.

Pages: (1) session info + DLC likelihood; (2) whole-session ethograms (method x view) with trial events, piezo licks and
learning trial; (3) state usage by trial epoch; (4) usage across the session vs whisker P(lick); (5) peri-event state
probability around the corrected first lick of hit trials and around jaw onsets; (7-12) state atlas per method x view:
fingerprints, motion-energy maps and keypoint trajectories of the top states (bm_atlas, 009, Kuma 011). State sequences are on the side 200 Hz grid (bm_states, v2). Usage = fraction of frames.
Learning trial: ssl-learning-trial-identification 017 `lt_final` (whisker-trial index of the stored day-0 curve), placed
at the kept whisker trial with that index (approximate if the curve's trial set differs).
Output: reports/<sid>_behaviour_motifs.pdf, tables/007_usage_<sid>.csv
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402
import bm_states as BS  # noqa: E402
import bm_atlas as BA  # noqa: E402

plt.rcParams.update({"font.size": 6, "font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})
NTOP = 8
EPOCHS = ("pre", "resp", "iti")
EPOCH_LABEL = {"pre": "pre-stim (-2-0 s)", "resp": "response window", "iti": "ITI"}
ROWS = [(m, v) for m in BS.METHODS for v in L.VIEWS]
LT_TABLE = L.ROOT / "combined_results_ks4" / "ssl-learning-trial-identification" / "artifacts" / "017_learning_trials_final.csv"


def state_colors(n):
    base = list(plt.cm.tab10.colors) + list(plt.cm.Set3.colors)
    return base[:n]


def top_states(z, method):
    if method == "lightning_action":
        return list(range(1, len(BS.LA_CLASSES)))
    v = z[z >= 0]
    return list(np.argsort(-np.bincount(v))[:NTOP])


def recode(z, top):
    """top states -> 0..n-1, other valid -> n, missing -> -1"""
    out = np.full(len(z), len(top))
    for i, s in enumerate(top):
        out[z == s] = i
    out[z < 0] = -1
    return out


def names(method, top):
    if method == "lightning_action":
        return [BS.LA_CLASSES[s] for s in top]
    return [f"{'S' if method == 'kpms' else 'A'}{s}" for s in top]


def learning_time(sid, trials):
    if not LT_TABLE.exists():
        return None, "learning-trial table not found"
    t = pd.read_csv(LT_TABLE)
    r = t[t.session_id == sid]
    if not len(r) or pd.isna(r.lt_final.iloc[0]):
        cat = r.category.iloc[0] if len(r) else "not in table"
        return None, f"no learning trial (017 category: {cat})"
    lt = int(r.lt_final.iloc[0])
    w = trials[trials.keep & (trials.stim == "whisker")]
    if lt >= len(w):
        return None, "learning trial index beyond kept whisker trials"
    return float(w.start_time.iloc[lt]), f"learning trial = whisker trial {lt} (017 lt_final, {r.category.iloc[0]})"


def page_info(pdf, sid, sel, trials, lik, lt_note, meta002):
    fig = plt.figure(figsize=(7.4, 9.5))
    r = sel.set_index("session_id").loc[sid]
    k = trials[trials.keep]
    lines = [f"Session {sid}", f"mouse {r.mouse_id} | cohort {r.reward_group} (reference sheet) | "
             f"learning_category {r.learning_category} -> {r.learner}",
             f"kept active trials {len(k)}: whisker {int((k.stim == 'whisker').sum())}, auditory "
             f"{int((k.stim == 'auditory').sum())}, no-stim {int((k.stim == 'no_stim').sum())}",
             f"P(lick): whisker {k[k.stim == 'whisker'].lick_flag.mean():.2f}, auditory {k[k.stim == 'auditory'].lick_flag.mean():.2f}, "
             f"no-stim (FA) {k[k.stim == 'no_stim'].lick_flag.mean():.2f}",
             f"cameras: side {r.rate_side:.1f} Hz, top {r.rate_top:.1f} Hz -> analysed at {L.FS_DS:.0f} Hz (native, keypoints from NWB_ks4)",
             f"alignment (mm, translation to the common spout-tip position): side via {meta002['side_ref']}; top via {meta002['top_ref']}",
             f"{lt_note}",
             "heuristic labels (pooled thresholds): " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in meta002["label_frac"].items()),
             "trials: active context, perf != 6, auditory warm-up cut; first lick = start + lick_time - response_window_start"]
    fig.text(0.05, 0.97, "\n".join(lines), va="top", fontsize=7, linespacing=1.6)
    for i, view in enumerate(L.VIEWS):
        ax = fig.add_axes([0.12 + 0.47 * i, 0.38, 0.34, 0.33])
        d = lik[(lik.session_id == sid) & (lik.view == view)].sort_values("frac_low")
        col = ["#444444" if b in L.KP_MODEL[view] else "#9ecae1" for b in d.bodypart]
        ax.barh([f"{b} (<{c:g})" for b, c in zip(d.bodypart, d.cutoff)], d.frac_low, color=col, lw=0)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("fraction of frames below the likelihood cutoff")
        ax.set_title(f"{view} view (dark = modelled, blue = reference / other)")
    ax = fig.add_axes([0.1, 0.06, 0.83, 0.22])
    w = k[k.stim == "whisker"]
    ax.plot(w.start_time / 60, w.lick_flag.rolling(10, min_periods=3, center=True).mean(), color=L.TASK_COL["whisker"], label="whisker")
    n = k[k.stim == "no_stim"]
    ax.plot(n.start_time / 60, n.lick_flag.rolling(10, min_periods=3, center=True).mean(), color=L.TASK_COL["no_stim"], label="no-stim (FA)")
    a = k[k.stim == "auditory"]
    ax.plot(a.start_time / 60, a.lick_flag.rolling(10, min_periods=3, center=True).mean(), color=L.TASK_COL["auditory"], label="auditory")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("session time (min)")
    ax.set_ylabel("P(lick), 10-trial running mean")
    ax.legend(frameon=False)
    pdf.savefig(fig)
    plt.close(fig)


def page_ethogram(pdf, sid, Z, sg, trials, lt_time, xlim=None):
    t = sg["t"] / 60
    if xlim is not None:
        keep = (t >= xlim[0]) & (t <= xlim[1])
        t = t[keep]
        Z = {k: v[keep] for k, v in Z.items()}
    fig, axs = plt.subplots(len(ROWS) + 2, 1, figsize=(7.4, 9.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[1] * len(ROWS) + [1.2, 0.6]))
    for ax, (m, v) in zip(axs, ROWS):
        z = Z.get((m, v))
        if z is None:
            ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center")
            ax.set_yticks([])
            ax.set_ylabel(f"{BS.METHOD_LABEL[m]}\n{v}", rotation=0, ha="right", va="center")
            continue
        top = top_states(z, m)
        rc = recode(z, top).astype(float)
        rc[rc < 0] = np.nan
        cols = state_colors(len(top)) + [(0.85, 0.85, 0.85)]
        ax.imshow(rc[None, :], aspect="auto", cmap=ListedColormap(cols), vmin=-0.5, vmax=len(top) + 0.5,
                  extent=[t[0], t[-1], 0, 1], interpolation="nearest", rasterized=True)
        ax.set_yticks([])
        ax.set_ylabel(f"{BS.METHOD_LABEL[m]}\n{v}", rotation=0, ha="right", va="center")
        hs = [plt.Rectangle((0, 0), 1, 1, color=c, lw=0) for c in cols]
        ax.legend(hs, names(m, top) + ["other"], ncol=len(hs), fontsize=4.5, frameon=False, loc="lower left",
                  bbox_to_anchor=(0, 1.0), handlelength=0.8, columnspacing=0.6)
    ax = axs[-2]
    k = trials[trials.keep]
    for j, stim in enumerate(("whisker", "auditory", "no_stim")):
        d = k[k.stim == stim]
        for lick, mk in ((1, "|"), (0, "|")):
            dd = d[d.lick_flag == lick]
            ax.scatter(dd.start_time / 60, np.full(len(dd), 2 * j + lick), marker=mk, s=12,
                       color=L.TASK_COL[stim], alpha=1.0 if lick else 0.35, lw=0.6)
    ax.set_yticks(range(6))
    ax.set_yticklabels(["whisker miss", "whisker hit", "auditory miss", "auditory hit", "no-stim CR", "no-stim FA"], fontsize=5)
    ax = axs[-1]
    ax.eventplot([sg["licks"] / 60], colors="k", linelengths=0.8, lw=0.3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_yticks([0])
    ax.set_yticklabels(["piezo licks"])
    ax.set_xlabel("session time (min)")
    for a in axs:
        if lt_time is not None:
            a.axvline(lt_time / 60, color="r", lw=0.8, ls="--")
    what = "whole-session ethograms" if xlim is None else f"zoom {xlim[0]:.1f}-{xlim[1]:.1f} min"
    fig.suptitle(f"{sid}: {what} (8 most used states coloured, grey = other; red dashed = learning trial)", fontsize=7)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.05, hspace=0.75)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def page_epoch(pdf, sid, Z, sg, usage_rows):
    ep = sg["epoch"]
    fig, axs = plt.subplots(3, 2, figsize=(7.4, 8))
    for ax, (m, v) in zip(axs.ravel(), ROWS):
        z = Z.get((m, v))
        ax.set_title(f"{BS.METHOD_LABEL[m]}, {v} view")
        if z is None:
            continue
        top = top_states(z, m)
        rc = recode(z, top)
        nm = names(m, top) + ["other"]
        w = 0.27
        for i, e in enumerate(EPOCHS):
            sel = (ep == e) & (rc >= 0)
            u = np.bincount(rc[sel], minlength=len(top) + 1) / max(sel.sum(), 1)
            ax.bar(np.arange(len(nm)) + (i - 1) * w, u, w, color=["#7f7f7f", "#d62728", "#1f77b4"][i], lw=0,
                   label=EPOCH_LABEL[e])
            for s, uu in zip(nm, u):
                usage_rows.append(dict(session_id=sid, method=m, view=v, state=s, epoch=e, usage=uu))
        ax.set_xticks(range(len(nm)))
        ax.set_xticklabels(nm, rotation=45, fontsize=5)
        ax.set_ylabel("fraction of frames in epoch")
        ax.legend(frameon=False, fontsize=5)
    fig.suptitle(f"{sid}: state usage by trial epoch (kept active trials)", fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_across(pdf, sid, Z, sg, trials):
    t = sg["t"]
    k = trials[trials.keep]
    w = k[k.stim == "whisker"]
    act = sg["epoch"] != "other"
    centers = np.arange(k.start_time.min() + 150, k.stop_time.max() - 150, 60)
    fig, axs = plt.subplots(3, 2, figsize=(7.4, 8))
    for ax, (m, v) in zip(axs.ravel(), ROWS):
        z = Z.get((m, v))
        ax.set_title(f"{BS.METHOD_LABEL[m]}, {v} view")
        if z is None:
            continue
        top = top_states(z, m)[:5]
        rc = recode(z, top)
        cols = state_colors(len(top))
        U = []
        for c in centers:
            s = (t >= c - 150) & (t < c + 150) & act & (rc >= 0)
            U.append(np.bincount(rc[s], minlength=len(top) + 1)[:len(top)] / max(s.sum(), 1))
        U = np.array(U)
        for i, nm in enumerate(names(m, top)):
            ax.plot(centers / 60, U[:, i], color=cols[i], lw=0.9, label=nm)
        ax.set_ylabel("usage (5-min window, active block)")
        ax.set_xlabel("session time (min)")
        ax2 = ax.twinx()
        hr = [w[(w.start_time >= c - 150) & (w.start_time < c + 150)].lick_flag.mean() for c in centers]
        ax2.plot(centers / 60, hr, color=L.TASK_COL["whisker"], lw=1.5, ls="--")
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel("whisker P(lick) (dashed)", color="#b07d00")
        ax2.spines["right"].set_visible(True)
        ax.legend(frameon=False, fontsize=4.5, ncol=3, loc="upper left")
    fig.suptitle(f"{sid}: usage of the 5 most used states across the session vs whisker P(lick)", fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def peri(z, t, events, top, win=(-1.0, 1.5)):
    lags = np.arange(int(win[0] * L.FS_DS), int(win[1] * L.FS_DS) + 1)
    idx = np.searchsorted(t, events)
    idx = idx[(idx + lags[0] >= 0) & (idx + lags[-1] < len(t))]
    rc = recode(z, top)
    M = rc[idx[:, None] + lags[None, :]]
    P = np.array([(M == i).sum(0) / np.maximum((M >= 0).sum(0), 1) for i in range(len(top))])
    return lags / L.FS_DS, P, len(idx)


def page_peri(pdf, sid, Z, sg, trials, jo):
    t = sg["t"]
    k = trials[trials.keep]
    hits = k[(k.lick_flag == 1) & k.stim.isin(["whisker", "auditory"]) & k.first_lick_time.notna()].first_lick_time.to_numpy()
    jon = jo[jo.jaw_onset_abs.notna() & (jo.context != "passive")].jaw_onset_abs.to_numpy()
    fig, axs = plt.subplots(6, 2, figsize=(7.4, 9.8))
    for r, (m, v) in enumerate(ROWS):
        z = Z.get((m, v))
        for c, (ev, lab) in enumerate(((hits, "first lick of hit trials (corrected)"), (jon, "jaw-opening onsets"))):
            ax = axs[r, c]
            if z is None or len(ev) == 0:
                ax.axis("off")
                continue
            top = top_states(z, m)
            x, P, n = peri(z, t, ev, top)
            im = ax.imshow(P, aspect="auto", extent=[x[0], x[-1], len(top) - 0.5, -0.5], cmap="magma", vmin=0,
                           vmax=max(0.05, np.nanmax(P)), interpolation="nearest")
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(names(m, top), fontsize=4.5)
            ax.axvline(0, color="w", lw=0.6, ls="--")
            ax.set_title(f"{BS.METHOD_LABEL[m]} {v}: {lab} (n={n})", fontsize=5.5)
            ax.set_xlabel("time from event (s)")
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01).set_label("P(state)", fontsize=5)
    fig.suptitle(f"{sid}: peri-event state probability (sanity checks)", fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_frames(pdf, sid):
    fr = BS.KR / "frames"
    for m in BS.METHODS:
        fs = [fr / f"{sid}_{m}_{v}.png" for v in L.VIEWS]
        if not any(f.exists() for f in fs):
            continue
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 9.5))
        for ax, f, v in zip(axs, fs, L.VIEWS):
            ax.axis("off")
            if f.exists():
                ax.imshow(plt.imread(f))
            ax.set_title(f"{BS.METHOD_LABEL[m]}, {v} view", fontsize=7)
        fig.suptitle(f"{sid}: raw-video frames at the middle of example bouts of the most used states", fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig, dpi=200)
        plt.close(fig)


def main():
    full = BS.SET == "full"
    sel = pd.read_csv(L.TABLES / ("full_sessions.csv" if full else "001_selected_sessions.csv"))
    lik = pd.read_csv(L.TABLES / ("full_dropout.csv" if full else "002_dropout.csv"))
    fr = pd.read_csv(L.TABLES / ("full_label_fractions.csv" if full else "002_label_fractions.csv")).set_index("session_id")
    ref = pd.read_csv(L.TABLES / ("full_alignment_reference.csv" if full else "002_alignment_reference.csv"))
    labs = [c for c in fr.columns if c not in ("rest_frac", "side_ref", "top_ref")]
    meta = {s: dict(label_frac={c: float(fr.loc[s, c]) for c in labs},
                    side_ref=ref[(ref.session_id == s) & (ref.view == "side")].method.iloc[0],
                    top_ref=ref[(ref.session_id == s) & (ref.view == "top")].method.iloc[0]) for s in sel.session_id}
    outdir = L.REPORTS / ("full_sessions" if full else "")
    outdir.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or list(sel.session_id)
    for sid in only:
        sg = BS.session_grid(sid)
        trials = pd.read_parquet(BS.INP / f"{sid}_trials.parquet")
        jf = BS.INP / f"{sid}_jaw_onsets.parquet"
        jo = pd.read_parquet(jf) if jf.exists() else pd.DataFrame(columns=["jaw_onset_abs", "context"])
        Z = {(m, v): BS.load_states(sid, m, v) for m, v in ROWS}
        Z = {k: v for k, v in Z.items() if v is not None}
        lt_time, lt_note = learning_time(sid, trials)
        rows = []
        final = outdir / f"{sid}_behaviour_motifs.pdf"
        tmp = outdir / f".{sid}_behaviour_motifs.tmp.pdf"
        with PdfPages(tmp) as pdf:
            page_info(pdf, sid, sel, trials, lik, lt_note, meta[sid])
            page_ethogram(pdf, sid, Z, sg, trials, lt_time)
            k = trials[trials.keep]
            mid = k.start_time.median() / 60
            page_ethogram(pdf, sid, Z, sg, trials, lt_time, xlim=(mid, mid + 2.0))
            page_epoch(pdf, sid, Z, sg, rows)
            page_across(pdf, sid, Z, sg, trials)
            page_peri(pdf, sid, Z, sg, trials, jo)
            T = BA.tables()
            for m, v in ROWS:  # state atlas (009 fingerprints, 011 motion-energy maps, keypoint trajectories)
                BA.page_atlas(pdf, sid, m, v, T)
        try:  # atomic replace; a PDF open in a viewer on Windows can lock the NAS file
            os.replace(tmp, final)
        except OSError:
            os.replace(tmp, final.with_name(final.stem + "_new.pdf"))
            print("locked, written as", final.stem + "_new.pdf", flush=True)
        if not full:
            pd.DataFrame(rows).to_csv(L.TABLES / f"007_usage_epoch_{sid}.csv", index=False)
        print("report", sid, sorted(Z), flush=True)
    json.dump(dict(script="007_session_reports.py", sessions=only, cohort_source="reference sheet (share_internal)",
                   learning_trial_table=str(LT_TABLE), kpms_seed_shown=0, n_top_states=NTOP,
                   epochs={"pre": "[-2, 0) s before stim", "resp": "stim to response_window_stop", "iti": "rest of active block"},
                   preprocessing="v3: mm, spout-tip aligned; kpms / lightning-action on deviations from the resting pose; "
                                 "AR-HMM features in physical units scaled with pooled statistics; labels with pooled thresholds",
                   kuma_meta={f"{m}_{v}": BS.kuma_meta(m, v) for m in ("arhmm",) for v in L.VIEWS},
                   versions=L.versions()), open(L.RES / "provenance" / ("007_full.json" if full else "007.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
