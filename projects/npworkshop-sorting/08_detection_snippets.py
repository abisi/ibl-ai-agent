"""08 -- Example snippets of the shared input with every sorter's detected spikes (user 2026-11-17): 100 ms x 24
channels (depth order), so that individual spikes can be seen and compared across sorters.

Extraction (Kuma, pipeline container, --rec): shared drift-corrected input + 300 Hz high-pass for display. Candidate
100 ms windows drawn at random (seed 0) in the central 80 % of the recording; each unit's extremum channel = channel of
the most negative trough of its mean waveform over up to N_WF spikes (+-1 ms); each spike is drawn on its unit's
extremum channel. Per window, the 24-channel block (step 2) with the most spikes; the N_EX windows with the most spikes
from >= 2 sorters and at least one spike from a sorter-only unit (07 tables, n_sorters == 1) are kept.
Outputs tables/08_snippets_<rec>.npz (traces uV, channel ids / depths, spikes), copied to the NAS by sbatch/snippets.sbatch.
Plot (here or haas, --plot): figures/fig7_snippets_<rec>_<k>.{pdf,png}; markers = spikes, one slot per sorter just
above the extremum channel's trace; filled = unit also found by another sorter, open = found by this sorter only.
Usage: python 08_detection_snippets.py --rec <rec_id> | --plot
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import npw  # noqa: E402

L = npw.SORTER_LABEL
WIN_MS, N_CH, N_CAND, N_EX, N_WF = 100, 24, 60, 4, 30


def traces_uv(rec, f0, f1):
    try:
        return rec.get_traces(start_frame=f0, end_frame=f1, return_in_uV=rec.has_scaleable_traces()).astype(np.float32)
    except TypeError:
        return rec.get_traces(start_frame=f0, end_frame=f1, return_scaled=rec.has_scaleable_traces()).astype(np.float32)


def extract(cfg, rec_id):
    import spikeinterface.preprocessing as spre
    rec = spre.highpass_filter(npw.load_input(cfg, rec_id), freq_min=300.0)
    fs = rec.sampling_frequency
    loc = rec.get_channel_locations()
    order = np.lexsort((loc[:, 0], loc[:, 1]))  # depth, then x
    names = [s for s in cfg["sorters"] if (npw.sorting_dir(cfg, rec_id, s) / "sorting").exists()]
    sortings = {s: npw.load_sorting(cfg, rec_id, s) for s in names}
    ut = pd.read_csv(npw.out(cfg, "tables", f"07_units_{rec_id}.csv"))
    k_of = {(r.sorter, str(r.unit_id)): int(r.n_sorters) for r in ut.itertuples()}
    rng = np.random.default_rng(0)
    n = rec.get_num_frames()
    w = int(WIN_MS * fs / 1000)
    starts = np.sort(rng.integers(int(0.1 * n), int(0.9 * n) - w, N_CAND))
    vec = {s: sortings[s].to_spike_vector() for s in names}
    uid = {s: np.asarray(sortings[s].unit_ids) for s in names}
    half = int(fs / 1000)
    peak = {}

    def peak_ch(s, ui):
        if (s, ui) not in peak:
            fr = vec[s]["sample_index"][vec[s]["unit_index"] == ui]
            fr = fr[(fr > half) & (fr < n - half)]
            fr = rng.choice(fr, min(N_WF, len(fr)), replace=False) if len(fr) else fr
            wf = np.mean([traces_uv(rec, f - half, f + half) for f in fr], axis=0) if len(fr) else np.zeros((1, len(loc)))
            peak[(s, ui)] = int(np.argmin(wf.min(axis=0)))
        return peak[(s, ui)]

    cands = []
    for f0 in starts:
        sp = []
        for s in names:
            v = vec[s]
            i0, i1 = np.searchsorted(v["sample_index"], [f0, f0 + w])
            for fr, ui in zip(v["sample_index"][i0:i1], v["unit_index"][i0:i1]):
                u = uid[s][ui]
                sp.append((s, str(u), int(fr - f0), peak_ch(s, int(ui)), k_of.get((s, str(u)), 1)))
        if not sp:
            continue
        sp = pd.DataFrame(sp, columns=["sorter", "unit_id", "frame", "ch", "n_sorters"])
        rank = np.empty(len(order), int)
        rank[order] = np.arange(len(order))
        sp["depth_rank"] = rank[sp.ch.to_numpy()]
        best = max(range(0, len(order) - N_CH + 1, 2),
                   key=lambda b: ((sp.depth_rank >= b) & (sp.depth_rank < b + N_CH)).sum())
        blk = sp[(sp.depth_rank >= best) & (sp.depth_rank < best + N_CH)]
        n_multi = blk.sorter.nunique()
        cands.append(dict(f0=int(f0), block=best, n=len(blk), n_sorters=n_multi, n_only=int((blk.n_sorters == 1).sum()),
                          spikes=blk))
        print(rec_id, f0, len(blk), n_multi, flush=True)
    ok = [c for c in cands if c["n_sorters"] >= 2 and c["n_only"] > 0]
    ex = sorted(ok, key=lambda c: -c["n"])[:N_EX]
    out = dict(rec_id=rec_id, fs=fs, sorters=np.array(names), win_ms=WIN_MS, in_uv=rec.has_scaleable_traces())
    for k, c in enumerate(ex):
        chans = order[c["block"]:c["block"] + N_CH]
        tr = traces_uv(rec, c["f0"], c["f0"] + w)[:, chans].T
        sp = c["spikes"].copy()
        sp["row"] = sp.depth_rank - c["block"]
        out[f"ex{k}_traces"] = tr
        out[f"ex{k}_t0_s"] = c["f0"] / fs
        out[f"ex{k}_depth_um"] = loc[chans, 1]
        out[f"ex{k}_x_um"] = loc[chans, 0]
        out[f"ex{k}_channel_ids"] = np.asarray(rec.channel_ids)[chans].astype(str)
        for col in ("sorter", "unit_id"):
            out[f"ex{k}_sp_{col}"] = sp[col].to_numpy().astype(str)
        for col in ("frame", "row", "n_sorters"):
            out[f"ex{k}_sp_{col}"] = sp[col].to_numpy().astype(int)
    out["n_ex"] = len(ex)
    np.savez_compressed(npw.out(cfg, "tables", f"08_snippets_{rec_id}.npz"), **out)
    print(rec_id, "examples", [(c["f0"] / fs, c["n"], c["n_sorters"], c["n_only"]) for c in ex])


def plot(cfg):
    col = cfg["colors"]
    for rec_id in cfg["recordings"]:
        p = npw.out(cfg, "tables", f"08_snippets_{rec_id}.npz")
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=False)
        names = [str(s) for s in d["sorters"]]
        missing = [L[s] for s in cfg["sorters"] if s not in names]
        fs = float(d["fs"])
        for k in range(int(d["n_ex"])):
            tr = d[f"ex{k}_traces"]
            t = np.arange(tr.shape[1]) / fs * 1000
            gap = max(np.percentile(np.abs(tr), 99.5) * 1.6, 60.0)
            fig, ax = plt.subplots(figsize=cfg["figures"]["size_in"])
            fig.subplots_adjust(left=0.1, right=0.78, top=0.86, bottom=0.1)
            for r in range(tr.shape[0]):
                ax.plot(t, tr[r] + r * gap, color="0.35", lw=0.6)
            slot = gap * 0.36 / max(len(names), 1)
            for si_, s in enumerate(names):
                m = d[f"ex{k}_sp_sorter"] == s
                fr, row, kk = d[f"ex{k}_sp_frame"][m], d[f"ex{k}_sp_row"][m], d[f"ex{k}_sp_n_sorters"][m]
                y = row * gap + gap * 0.12 + si_ * slot  # above the trace baseline (troughs point down)
                ax.scatter(fr[kk > 1] / fs * 1000, y[kk > 1], s=26, color=col[s], zorder=3)
                ax.scatter(fr[kk == 1] / fs * 1000, y[kk == 1], s=26, facecolor="white", edgecolor=col[s], lw=1.4,
                           zorder=3)
            dep = d[f"ex{k}_depth_um"]
            ax.set_yticks(np.arange(tr.shape[0])[::2] * gap, [f"{v:.0f}" for v in dep[::2]], fontsize=11)
            ax.set_ylabel("distance from probe tip (um)")
            ax.set_xlabel("time (ms)")
            ax.set_xlim(0, t[-1])
            ax.set_ylim(-gap, tr.shape[0] * gap)
            ax.spines[["left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            # scale bar: 100 uV, 5 ms
            x0 = t[-1] * 1.01
            ax.plot([x0, x0], [-gap * 0.6, -gap * 0.6 + 100], color="k", lw=2, clip_on=False)
            ax.text(x0 + t[-1] * 0.008, -gap * 0.6 + 50, "100 uV" if bool(d["in_uv"]) else "100 a.u.", va="center",
                    fontsize=11)
            handles = [plt.Line2D([], [], marker="o", ls="", color=col[s], ms=8) for s in names]
            handles += [plt.Line2D([], [], marker="o", ls="", color="0.4", ms=8),
                        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="0.4", ms=8)]
            labels = [L[s] for s in names] + ["unit also found by\nanother sorter", "unit found by\nthis sorter only"]
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.8, 0.5), frameon=False, fontsize=13)
            sub = "300 Hz high-pass of the shared drift-corrected input; each spike on its unit's extremum channel"
            if missing:
                sub += f"; not yet sorted: {', '.join(missing)}"
            fig.suptitle(f"{rec_id}, t = {float(d[f'ex{k}_t0_s']):.1f} s: spikes detected by each sorter", fontsize=18,
                         x=0.1, ha="left", y=0.97)
            fig.text(0.1, 0.905, sub, fontsize=12, color="0.35")
            npw.save_fig(cfg, fig, f"fig7_snippets_{rec_id}_{k + 1}")
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    cfg = npw.load_config()
    if a.rec:
        extract(cfg, a.rec)
    if a.plot:
        npw.style(cfg)
        plot(cfg)


if __name__ == "__main__":
    main()
