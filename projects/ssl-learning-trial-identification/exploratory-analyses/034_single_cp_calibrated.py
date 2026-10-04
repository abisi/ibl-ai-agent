"""034 -- ONE learning-trial definition (no cascade): extended joint change-point model, shuffle-calibrated (user 2026-10-01:
"Ideally I don't want a cascade but a single metric"; AB118 should learn after its hit rate recovers, MH070 around trial 30
when whisker and FA separate; "would it make sense to use the whole spontaneous lick rate instead [of FA trials]?").
Model (raw outcomes, Beta(1,1) segment rates integrated out, segments >= MIN_SEG whisker trials), two streams: whisker
outcomes (per whisker trial) and a "baseline licking" stream assigned to segments by time:
  R+ change: naive -> [optional LAPSE: whisker rate below both naive and learned, <= MAX_LAPSE trials] -> learned
             (discrimination whisker - baseline higher than naive) -> [optional end decline: whisker rate below learned];
             prior: 1/2 no-lapse family, 1/2 lapse family, uniform within each.
  R+ null:   flat | flat -> decline | flat -> lapse -> flat (same rates either side of the lapse).
  R- change: generalising -> learned (discrimination lower); null: flat.
  No "high_from_start" / "never_licked" gates: a mouse that starts high is a learner only if its discrimination still rises
  (relative to its own start); a mouse that never licked cannot show a fall.
  LT = posterior median of the first trial of the learned segment (90% CI reported).
Calibration (replaces the fixed log10 BF 0.5 / 0): per session, N_SHUF shuffles of the whisker outcomes and of the
baseline stream (each permuted within itself; keeps the rates, destroys time structure); p = P(BF_shuffle >= BF_real);
learner iff p < ALPHA. Expected false-positive rate = ALPHA by construction.
Baseline streams (two versions of the same model):
  FA     no-stim trial outcomes (as L6);
  SPONT  spontaneous licking: one 1-s pseudo-trial per trial (any type, active context, perf != 6, curve set) = the second
         before the trial's abort (quiet) window starts, kept only if it starts >= CONSUME s after the previous trial's
         response-window end (excludes reward consumption); y = any piezo lick in it (NWB_ks4 piezo_lick_times).
         ~3x more samples than no-stim trials, one per inter-trial interval (lick bouts not counted as independent).
Outputs: artifacts/034_single_cp.csv (per session x stream: BF, p, LT, CI, learner), 034_spont_vs_fa.csv (validity of the
spontaneous stream), figures 034_single_cp_review_<cohort>_<stream>.png and 034_single_cp_summary.png.
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/034_single_cp_calibrated.py
"""

from __future__ import annotations

import os
import zlib
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
MIN_SEG, MAX_LAPSE = 5, 30
N_SHUF, ALPHA = 50, 0.05
CONSUME, BIN = 4.0, 1.0
N_WORKERS = int(os.environ.get("SSL_N_WORKERS", "40"))


class Streams:
    """Whisker outcomes o (n) + baseline stream (times bt, outcomes by) assigned to whisker-trial segments by time."""

    def __init__(self, o, wt, by, bt):
        self.n = len(o)
        self.cw = np.r_[0, np.cumsum(o)]
        idx = np.searchsorted(wt, bt, side="right") - 1          # baseline sample belongs to the whisker trial before it
        idx = np.clip(idx, 0, self.n - 1)
        self.cb_s = np.r_[0, np.cumsum(np.bincount(idx, weights=by, minlength=self.n))]
        self.cb_m = np.r_[0, np.cumsum(np.bincount(idx, minlength=self.n))]

    def counts(self, a, b):
        return self.cw[b] - self.cw[a], b - a, self.cb_s[b] - self.cb_s[a], self.cb_m[b] - self.cb_m[a]

    def ml(self, a, b):
        sw, mw, sb, mb = self.counts(a, b)
        return betaln(sw + 1, mw - sw + 1) + betaln(sb + 1, mb - sb + 1)

    def ml_pooled(self, a1, b1, a2, b2):
        s1, s2 = self.counts(a1, b1), self.counts(a2, b2)
        sw, mw, sb, mb = (x + y for x, y in zip(s1, s2))
        return betaln(sw + 1, mw - sw + 1) + betaln(sb + 1, mb - sb + 1)

    def wrate(self, a, b):
        return (self.cw[b] - self.cw[a]) / max(b - a, 1)

    def disc(self, a, b):
        sw, mw, sb, mb = self.counts(a, b)
        return sw / max(mw, 1) - (sb / mb if mb > 0 else 0.0)


def fit(S, rg):
    """Returns log10 BF and the posterior over the learned-segment start (pk, length n)."""
    n = S.n
    fam_no, fam_lapse, l_no, l_lapse = [], [], [], []
    if rg == "R+":
        for k in range(MIN_SEG, n - MIN_SEG + 1):          # learned starts at k (no lapse)
            dn = S.disc(0, k)
            for j in list(range(k + MIN_SEG, n - MIN_SEG + 1)) + [n]:
                if S.disc(k, j) <= dn or (j < n and S.wrate(j, n) >= S.wrate(k, j)):
                    continue
                fam_no.append(S.ml(0, k) + S.ml(k, j) + (S.ml(j, n) if j < n else 0.0))
                l_no.append(k)
        for k in range(MIN_SEG, n - 2 * MIN_SEG + 1):      # naive [0,k) lapse [k,l) learned [l,j) [decline]
            dn, wn = S.disc(0, k), S.wrate(0, k)
            for l in range(k + MIN_SEG, min(k + MAX_LAPSE, n - MIN_SEG) + 1):
                wl = S.wrate(k, l)
                if wl >= wn:
                    continue
                for j in list(range(l + MIN_SEG, n - MIN_SEG + 1)) + [n]:
                    if S.disc(l, j) <= dn or wl >= S.wrate(l, j) or (j < n and S.wrate(j, n) >= S.wrate(l, j)):
                        continue
                    fam_lapse.append(S.ml(0, k) + S.ml(k, l) + S.ml(l, j) + (S.ml(j, n) if j < n else 0.0))
                    l_lapse.append(l)
        null = [S.ml(0, n)] + [S.ml(0, j) + S.ml(j, n) for j in range(MIN_SEG, n - MIN_SEG + 1)
                               if S.wrate(j, n) < S.wrate(0, j)]
        null_lapse = []
        for k in range(MIN_SEG, n - 2 * MIN_SEG + 1):
            for l in range(k + MIN_SEG, min(k + MAX_LAPSE, n - MIN_SEG) + 1):
                if S.wrate(k, l) < S.wrate(0, k):
                    null_lapse.append(S.ml_pooled(0, k, l, n) + S.ml(k, l))
        lnull = [logsumexp(null) - np.log(len(null))]
        if null_lapse:
            lnull.append(logsumexp(null_lapse) - np.log(len(null_lapse)))
        log_null = logsumexp(lnull) - np.log(len(lnull))
    else:
        for k in range(MIN_SEG, n - MIN_SEG + 1):
            if S.disc(k, n) >= S.disc(0, k):
                continue
            fam_no.append(S.ml(0, k) + S.ml(k, n))
            l_no.append(k)
        log_null = S.ml(0, n)
    fams = []
    for logs, ls in ((fam_no, l_no), (fam_lapse, l_lapse)):
        if logs:
            fams.append((np.asarray(logs), np.asarray(ls)))
    if not fams:
        return -np.inf, None
    fam_ev = [logsumexp(lg) - np.log(len(lg)) for lg, _ in fams]
    log_change = logsumexp(fam_ev) - np.log(len(fam_ev))
    pk = np.zeros(n)
    for (lg, ls), ev in zip(fams, fam_ev):
        w = np.exp(lg - logsumexp(lg)) * np.exp(ev - logsumexp(fam_ev))   # within-family posterior x family posterior
        pk += np.bincount(ls, weights=w, minlength=n)
    return float((log_change - log_null) / np.log(10)), pk


def summarize_pk(pk):
    cdf = np.cumsum(pk) / pk.sum()
    return float(np.searchsorted(cdf, 0.5)), float(np.searchsorted(cdf, 0.05)), float(np.searchsorted(cdf, 0.95))


def spont_stream(sid, nwb_dir):
    from pynwb import NWBHDF5IO
    from ssl_timeresolved_decoding import _active_trials_for_curve_untrimmed as U
    with NWBHDF5IO(str(nwb_dir / f"{sid}.nwb"), "r") as io:
        n = io.read()
        t = n.trials.to_dataframe().assign(session_id=sid)
        licks = np.asarray(n.processing["behavior"].data_interfaces["BehavioralEvents"].time_series["piezo_lick_times"].timestamps[:])
    allt = t.sort_values("start_time").reset_index(drop=True)
    act = U(sid, t)
    keep = set(act.start_time.round(4))
    prev_end = np.r_[-np.inf, allt.response_window_stop_time.to_numpy()[:-1]]
    bt, by = [], []
    licks = np.sort(licks)
    for i, r in allt.iterrows():
        if round(r.start_time, 4) not in keep:
            continue
        a = r.abort_window_start_time
        if not np.isfinite(a):
            continue
        b0 = a - BIN
        if b0 < prev_end[i] + CONSUME:
            continue
        nl = np.searchsorted(licks, a) - np.searchsorted(licks, b0)
        bt.append(b0)
        by.append(int(nl > 0))
    return np.asarray(bt, float), np.asarray(by, float)


def one(args):
    sid, d, rg, nwb_dir = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(REPO / "scripts"))
    o, wt = np.asarray(d["w_outcomes"], float), np.asarray(d["w_start"], float)
    streams = {"FA": (np.asarray(d["n_start"], float), np.asarray(d["n_outcomes"], float))}
    try:
        streams["SPONT"] = spont_stream(sid, nwb_dir)
    except Exception as e:  # noqa: BLE001
        streams["SPONT"] = None
        err = repr(e)[:120]
    rng = np.random.default_rng(zlib.crc32(sid.encode()))   # reproducible (hash() is salted per process)
    out = []
    for name, st in streams.items():
        if st is None:
            out.append(dict(session_id=sid, stream=name, error=err))
            continue
        bt, by = st
        bf, pk = fit(Streams(o, wt, by, bt), rg)
        null = []
        for _ in range(N_SHUF):
            b, _ = fit(Streams(rng.permutation(o), wt, rng.permutation(by), bt), rg)
            null.append(b)
        null = np.asarray(null)
        p = float((np.sum(null >= bf) + 1) / (N_SHUF + 1))
        lt, lo, hi = summarize_pk(pk) if pk is not None and pk.sum() > 0 else (np.nan, np.nan, np.nan)
        out.append(dict(session_id=sid, mouse_id=d["mouse_id"], reward_group=rg, learning_category=d["learning_category"],
                        stream=name, n_whisker=len(o), n_baseline=len(by), baseline_rate=float(by.mean()) if len(by) else np.nan,
                        log10_bf=bf, p_shuffle=p, null_bf_95=float(np.quantile(null, 0.95)), LT=lt, LT_ci05=lo, LT_ci95=hi,
                        learner=bool(p < ALPHA), learner_p10=bool(p < 0.10), pk=pk))
    return out


def main():
    os.chdir(REPO)
    from axel_bisi_paths import axel_bisi_root
    nwb_dir = axel_bisi_root() / "NWB_ks4"
    inp = pickle.load(open(ART / "028_chain_all" / "001_inputs.pkl", "rb"))
    only = set(sys.argv[1:])
    jobs = [(sid, d, d["reward_group"], nwb_dir) for sid, d in inp.items() if not only or d["mouse_id"] in only]
    rows = []
    with ProcessPoolExecutor(min(N_WORKERS, len(jobs))) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % 10 == 0:
                print(i, "/", len(jobs), flush=True)
    R = pd.DataFrame(rows)
    pk = {(r.session_id, r.stream): r.pk for r in R.itertuples() if hasattr(r, "pk")}
    R.drop(columns=["pk"]).to_csv(ART / f"034_single_cp{'_' + '_'.join(sorted(only)) if only else ''}.csv", index=False)
    pickle.dump(pk, open(ART / f"034_single_cp_pk{'_' + '_'.join(sorted(only)) if only else ''}.pkl", "wb"))
    lt = pd.read_csv(ART / "028_learning_trials_all_methods_all_mice.csv").set_index("session_id")
    R["L6 lenient"] = R.session_id.map(lambda s: lt.loc[s, "L6 lenient"] if s in lt.index else np.nan)
    pd.set_option("display.width", 220)
    for (st, rg), g in R.groupby(["stream", "reward_group"]):
        both = g[g.learner & g["L6 lenient"].notna()]
        print(st, rg, "n", len(g), "learners p<.05", int(g.learner.sum()), "p<.10", int(g.learner_p10.sum()),
              "| L6 lenient", int(g["L6 lenient"].notna().sum()), "| median |LT - L6len|",
              float(np.median(np.abs(both.LT - both["L6 lenient"]))) if len(both) else np.nan)
    print(R[R.mouse_id.isin(["AB118", "MH070"])][["mouse_id", "stream", "log10_bf", "p_shuffle", "LT", "LT_ci05", "LT_ci95",
                                                  "baseline_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
