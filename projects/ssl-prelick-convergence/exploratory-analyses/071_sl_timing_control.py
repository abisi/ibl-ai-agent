"""Control: are spontaneous licks too close to task events? lambda (057 definition) with spontaneous licks restricted by
their time since the previous trial start / since the previous rewarded lick (spontaneous-lick reference, whole brain).

Timing per spontaneous lick (bout onset, piezo clock = trial clock): time since the previous trial start, time since
the previous rewarded first lick (R+: whisker and auditory hits; R-: auditory hits), from trials.parquet.
Exclusions tested: none (current), >= 5 s after the previous trial start, >= 10 s after the previous trial start,
>= 5 s after the previous rewarded lick, >= 10 s after the previous rewarded lick.
lambda: 057 (z-scored good + mua units with mean raw pre-lick rate >= 0.1 Hz over the used events, >= 5 units,
>= 4 events per class, cross-validated over 50 split halves, AH-reference axis >= 0.01 per unit, clipped [-1, 2]).
Statistics: 062 group_stats (MWU, mouse-level permutation interaction), all mice.
Output: combined_results_ks4/_roc_prelick_sl/sl_timing_control/
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
m57 = importlib.import_module("057_roc_prelick_lambda")
m62 = importlib.import_module("062_pub_convergence_figures")
assert m51.REF == "sl"
OUT = m51.OUTROOT / "sl_timing_control"
TRIALS = pathlib.Path.home() / "code/ibl-ai-agent/reports/datasets/ssl_ephys/1.0.0/metadata/trials.parquet"
RULES = [("none", None, None), ("trial >= 5 s", "since_trial", 5), ("trial >= 10 s", "since_trial", 10),
         ("reward >= 5 s", "since_reward", 5), ("reward >= 10 s", "since_reward", 10)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = pd.read_parquet(TRIALS)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    rng = np.random.default_rng(0)
    rows = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        t = T[T.session_id == r.session_id].sort_values("start_time")
        if not f.exists() or t.empty:
            continue
        z = np.load(f, allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        K = K.merge(W[W.session_id == r.session_id][["electrode_group", "cluster_id", "cohort", "stage", "mouse_id",
                                                      "quality_label"]], on=["electrode_group", "cluster_id"], how="left")
        if K.cohort.isna().all():
            continue
        coh = K.cohort.dropna().iloc[0]
        meta = dict(session_id=r.session_id, mouse_id=K.mouse_id.dropna().iloc[0], cohort=coh, stage=K.stage.dropna().iloc[0])
        X, raw, lab, ev = z["rates"].astype(float), z["raw"].astype(float), z["cls"], z["trial_start"]
        st = t.start_time.to_numpy()
        rew = t[(t.lick_flag == 1) & t.trial_type.isin(["auditory_trial", "whisker_trial"] if coh == "R+" else ["auditory_trial"])]
        rl = np.sort((rew.start_time + rew.lick_time - rew.response_window_start_time).dropna().to_numpy())
        i = np.searchsorted(st, ev) - 1
        since_trial = np.where(i >= 0, ev - st[np.clip(i, 0, None)], np.inf)
        j = np.searchsorted(rl, ev) - 1
        since_rew = np.where(j >= 0, ev - rl[np.clip(j, 0, None)], np.inf) if len(rl) else np.full(len(ev), np.inf)
        timing = {"since_trial": since_trial, "since_reward": since_rew}
        for name, col, thr in RULES:
            keep = np.ones(len(lab), bool)
            if col:
                keep = (lab != "FA") | (timing[col] >= thr)
            idx = np.where(keep)[0]
            l = lab[idx]
            if min((l == c).sum() for c in m51.CLASSES) < 4:
                continue
            ok = (raw[:, idx].mean(1) >= m51.MIN_FR) & K.quality_label.isin(m57.UNIT_SET).to_numpy()
            Z = X[:, idx]; sd = Z.std(1); ok &= sd > 0
            Z = (Z - Z.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1)[:, None]
            if ok.sum() < m57.MIN_UNITS:
                continue
            res = m57.lam(Z[ok], l, rng)
            if res:
                lam = res["lam"] if res["d_AH_FA"] >= m57.AXIS_MIN else np.nan
                rows.append(dict(meta, rule=name, n_SL=int((l == "FA").sum()), lam=float(np.clip(lam, *m57.LAM_CLIP))
                                 if np.isfinite(lam) else np.nan, dd=res["d_WH_FA"] - res["d_WH_AH"]))
    S = pd.DataFrame(rows); S.to_csv(OUT / "sessions.csv", index=False)
    st_ = []
    for name, _, _ in RULES:
        d = S[S.rule == name]
        for col in ["lam", "dd"]:
            G = m62.group_stats(d, col, f"{name} {col}", rng, "sl-timing")
            st_.append(dict(rule=name, measure=col, **{f"mean {m62.GLAB[k]}": G["means"][k] for k in m62.GROUPS},
                            **{f"n {m62.GLAB[k]}": G["n"][k] for k in m62.GROUPS},
                            median_SL_per_session=float(d.n_SL.median()),
                            p_Rplus_change=G.get("R+ L vs E", (np.nan,))[0],
                            p_expert=G.get("expert R+ vs R-", (np.nan,))[0], p_interaction=G["interaction"][1]))
    R = pd.DataFrame(st_); R.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
