"""Hits whose onset is already after the stimulus and moves with the prominent-peak rule (009, f = 0.5): how many shifts
skip a real first opening vs a small twitch? (user, 2026-11-12: "show me the bad examples; are there more or less bad
than good shifts?")

Per changed trial (|new - current| > 10 ms, whisker and auditory hits, 200 Hz): skipped opening = maximum jaw distance
(baseline SD) between the current onset and the new onset; also as a fraction of the trial maximum before the lick, and
the latencies. Classes by the skipped opening (the class bounds are a choice, to look at, not a rule):
  small   < 10 SD (twitch / drift: the shift looks right)
  medium  10-20 SD
  large   >= 20 SD (a clear first opening: the shift looks wrong)
Separate first opening = after the skipped peak the jaw closes at least halfway before the new onset (otherwise the
new onset is later on the same, possibly stepwise, rise: less conservative, not a different movement).
Figure: histograms of the skipped opening and of the shift, then 6 random trials each: continuous rise, separate opening
< 10 SD, separate opening >= 10 SD (current onset purple, new onset green, first lick pink).
Output: combined_results_ks4/ssl-dlc-jaw-onset/figures/010_poststim_shift_review.png, tables/010_poststim_shift_review.csv
Run on haas after 009: ~/code/ibl-ai-agent/.venv/bin/python 010_poststim_shift_review.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

R = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
OUT = R / "ssl-dlc-jaw-onset"
T0, FS = -1.0, 200.0
CLASSES = [("small", 0, 10), ("medium", 10, 20), ("large", 20, np.inf)]
TYPES = {"whisker_trial": "Whisker hit", "auditory_trial": "Auditory hit"}


def traces(mouse, keys):
    d = pd.read_pickle(R / mouse / "dlc_jaw_onset_times.pkl")
    d = d.set_index(["session_id", "trial_id"])
    out = {}
    for k in keys:
        r = d.loc[k]
        out[k] = np.asarray(r.jaw_dlc_pos, float) / (r.threshold / 5.0)
    return out


def main():
    v = pd.read_csv(OUT / "tables" / "009_onset_variants.csv")
    ch = v[v.trial_type.isin(TYPES) & (v.onset_pkl >= 0) & ((v["onset_prom0.5"] - v.onset_pkl).abs() > 0.01)].copy()
    ch = ch.rename(columns={"onset_prom0.5": "onset_new"})
    groups = {m: list(zip(g.session_id, g.trial_id)) for m, g in ch.groupby("mouse_id")}
    res = Parallel(n_jobs=48)(delayed(traces)(m, ks) for m, ks in groups.items())
    tr = {k: x for r in res for k, x in r.items()}
    rows = []
    for r in ch.itertuples():
        x = tr[(r.session_id, r.trial_id)]
        if len(x) != 621:  # 200 Hz grid only
            continue
        a, b = int(round((r.onset_pkl - T0) * FS)), int(round((r.onset_new - T0) * FS))
        stop = int(round((r.lick - T0) * FS)) if np.isfinite(r.lick) else len(x)
        top = np.nanmax(x[int(-T0 * FS):max(stop, int(-T0 * FS) + 1)])
        skipped = np.nanmax(x[a:b]) if b > a else np.nan
        # separate first opening: after its peak, the jaw closes at least halfway (towards baseline) before the new onset
        if b > a + 1:
            pk = a + int(np.nanargmax(x[a:b]))
            closes = np.nanmin(x[pk:b + 1]) <= 0.5 * x[pk] if pk < b else False
        else:
            closes = False
        rows.append(dict(session_id=r.session_id, trial_id=r.trial_id, trial_type=r.trial_type, onset=r.onset_pkl,
                         onset_new=r.onset_new, lick=r.lick, skipped_sd=skipped,
                         skipped_frac_of_max=skipped / top if top > 0 else np.nan, separate_opening=bool(closes)))
    t = pd.DataFrame(rows)
    t["shift_ms"] = 1000 * (t.onset_new - t.onset)
    t["cls"] = pd.cut(t.skipped_sd, [c[1] for c in CLASSES] + [np.inf], labels=[c[0] for c in CLASSES], right=False)
    t.to_csv(OUT / "tables" / "010_poststim_shift_review.csv", index=False)
    n_hits = int((v.trial_type.isin(TYPES) & (v.onset_pkl >= 0)).sum())
    print(f"{len(t)} changed post-stimulus hit onsets (200 Hz) of {n_hits}")
    print(t.groupby("cls", observed=False).agg(n=("shift_ms", "size"), median_shift_ms=("shift_ms", "median"),
                                               median_skipped_frac=("skipped_frac_of_max", "median")).round(2))
    print(t.groupby(["trial_type", "cls"], observed=False).size().unstack())
    print("separate first opening (jaw closes >= halfway before the new onset):",
          int(t.separate_opening.sum()), f"({100 * t.separate_opening.mean():.0f}%)",
          "| by class:", t.groupby("cls", observed=False).separate_opening.mean().mul(100).round(0).to_dict(),
          "| by type:", t.groupby("trial_type").separate_opening.mean().mul(100).round(0).to_dict())
    print("separate openings, skipped peak (SD) quartiles:",
          t[t.separate_opening].skipped_sd.quantile([.25, .5, .75]).round(0).to_dict())

    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 6)
    ax = fig.add_subplot(gs[0, :2])
    ax.hist(t.skipped_sd.clip(upper=60), bins=np.arange(0, 61, 1), color="0.4")
    for c in (10, 20):
        ax.axvline(c, color="r", lw=0.8, ls="--")
    ax.set_xlabel("Skipped first opening: max jaw distance between current and new onset (baseline SD)")
    ax.set_ylabel("Trials")
    counts = t.cls.value_counts()
    ax.set_title(f"{len(t)} changed post-stimulus hit onsets of {n_hits}: "
                 + ", ".join(f"{c} {counts.get(c, 0)} ({100 * counts.get(c, 0) / len(t):.0f}%)" for c, _, _ in CLASSES),
                 fontsize=9)
    ax = fig.add_subplot(gs[0, 2:4])
    ax.hist(t.shift_ms, bins=np.arange(10, 400, 10), color="0.4")
    ax.set_xlabel("Onset shift (ms, new - current)")
    slots = [(r_, c) for r_ in (1, 2, 3) for c in range(6)]
    ex = [("continuous rise", t[~t.separate_opening]),
          ("separate opening < 10 SD", t[t.separate_opening & (t.skipped_sd < 10)]),
          ("separate opening >= 10 SD", t[t.separate_opening & (t.skipped_sd >= 10)])]
    for ci, (cls, sel) in enumerate(ex):
        pick = sel.sample(min(6, len(sel)), random_state=rng.integers(1e9))
        for j, r in enumerate(pick.itertuples()):
            ax = fig.add_subplot(gs[ci + 1, j])
            x = tr[(r.session_id, r.trial_id)]
            tt = T0 + np.arange(len(x)) / FS
            ax.plot(tt, x, color="k", lw=0.8)
            ax.axhline(5, color="r", lw=0.5, ls=":")
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.axvline(r.onset, color="purple", lw=1.2)
            ax.axvline(r.onset_new, color="g", lw=1.2)
            ax.axvline(r.lick, color="#e7298a", lw=1)
            ax.set_xlim(-0.2, 0.7)
            ax.set_title(f"{cls}: {TYPES[r.trial_type]} {r.session_id[:14]} t{r.trial_id} "
                         f"(skipped {r.skipped_sd:.0f} SD)", fontsize=7)
    fig.suptitle("Post-stimulus hit onsets moved by the prominent-peak rule (f = 0.5): continuous rise vs separate first "
                 f"opening (jaw closes >= halfway before the new onset: {int(t.separate_opening.sum())} of {len(t)}); "
                 "purple current, green new, pink first lick; 6 random per row", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"010_poststim_shift_review.{ext}", dpi=110)


if __name__ == "__main__":
    main()
