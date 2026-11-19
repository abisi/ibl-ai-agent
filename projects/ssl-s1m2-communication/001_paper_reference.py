"""Paper reference values from the authors' Zenodo CCV maps (10.5281/zenodo.21451128): per session, the readout of
fig_3defjkl.m (mean CCV over delays 0..+15 ms, minus -250..0 ms baseline, mean over 0..100 ms) and its feedback
counterpart (delays -15..0 ms), plus the shuffle-map readouts and the grand-average time courses.

Run on haas: python 001_paper_reference.py  ->  OUT/paper_reference/{per_session.parquet, traces.npz, config.json}
"""
import json

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

import s1m2_lib as L

SRC = L.OUT / "paper_zenodo" / "data"
DST = L.OUT / "paper_reference"
FILES = {(corr, res, cond): SRC / f"CorrMapFin_{corr}CCA_{res}_Dim1_{cond}.mat"
         for corr in ("Total", "Noise") for res in ("5msBin",)
         for cond in ("All", "Go", "NoGo", "hit", "miss", "CR", "FA", "Response")}


def _frac(E, k, s):
    return float(np.ravel(getattr(E, k))[s]) if hasattr(E, k) else np.nan


def main():
    DST.mkdir(parents=True, exist_ok=True)
    dprime = np.ravel(sio.loadmat(SRC / "dPrime.mat", squeeze_me=True)["dPrime"]).astype(float)
    rows, traces = [], {}
    for (corr, res, cond), f in FILES.items():
        if not f.exists():
            continue
        E = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["Edata"]
        C, S, d, t = E.CorMapAll, E.shflCorMap, np.asarray(E.delays, float), np.asarray(E.t_corr, float)
        # times are nominal (-0.459 ... ); the scripts index the bin nearest 0 as stimulus time
        t = t - t[np.argmin(np.abs(t))]
        for direction, rng in (("FF", (0.0, 0.015)), ("FB", (-0.015, 0.0))):
            tr_all, trs_all, trc_all = [], [], []
            for s in range(C.shape[2]):
                v, tr, _ = L.readout(C[:, :, s], d, t, rng)
                vs, trs, _ = L.readout(S[:, :, s], d, t, rng)
                vc, trc, _ = L.readout_clean(C[:, :, s], d, t, rng)  # same windows as the SSL option-a readout
                tr_all.append(tr); trs_all.append(trs); trc_all.append(trc)
                rows.append(dict(corr=corr, condition=cond, direction=direction, session_idx=s, value=v, value_clean=vc,
                                 shuffle_value=vs, dprime=dprime[s] if s < len(dprime) else np.nan,
                                 fracSig_S1=_frac(E, "fracSig_S1", s), fracSig_M2=_frac(E, "fracSig_M2", s)))
            traces[f"{corr}_{cond}_{direction}"] = np.array(tr_all)
            traces[f"{corr}_{cond}_{direction}_shuffle"] = np.array(trs_all)
            traces[f"{corr}_{cond}_{direction}_clean"] = np.array(trc_all)
        traces[f"{corr}_{cond}_map_mean"] = np.nanmean(C, axis=2)
        traces[f"{corr}_{cond}_t"], traces["delays"] = t, d
    df = pd.DataFrame(rows)
    df.to_parquet(DST / "per_session.parquet")
    np.savez_compressed(DST / "traces.npz", **traces)
    summ = []
    for (corr, cond, direction), g in df.groupby(["corr", "condition", "direction"]):
        v = g.value.dropna()
        w = stats.wilcoxon(v) if len(v) > 5 else None
        ok = g.dropna(subset=["value", "dprime"])
        r = stats.pearsonr(ok.value, ok.dprime) if len(ok) > 5 else (np.nan, np.nan)
        summ.append(dict(corr=corr, condition=cond, direction=direction, n=len(v), mean=v.mean(), median=v.median(),
                         wilcoxon_p=w.pvalue if w else np.nan, r_dprime=r[0], p_dprime=r[1]))
    summ = pd.DataFrame(summ)
    summ.to_csv(DST / "summary.csv", index=False)
    (DST / "config.json").write_text(json.dumps({"source": str(SRC), "readout": L.CONFIG["readout"]}, indent=1))
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
