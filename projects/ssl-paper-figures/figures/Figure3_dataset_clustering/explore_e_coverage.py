"""Diagnostic (2026-11-26): why only ~26k of the 43k rastermap neurons carry a decoder weight for detection / movement
(Figure 3 E). Per area group: units in the decoding unit table (units_stable.parquet, 126 sessions), rastermap neurons,
rastermap neurons with >= 1 weight sample at N = 200 (area-group runs), n_samples distribution, and the coverage expected
if units were simply never drawn (100 iterations x 20 sessions drawn with replacement from the eligible sessions x N/20 =
10 units per session with replacement). Key checks: rastermap keys absent from the unit table, area-group label mismatch.
Also: movement (jaw FA vs CR) per metacluster summary from the E stats + area-curve peaks.
Output: printed + Figures/_explore/Figure3_E_proposal_2026-11-26/coverage_*.csv
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. .venv/bin/python <this>
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute as C  # noqa: E402

KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
U = pd.read_parquet(C.RERUN / "units" / "units_stable.parquet").astype({"cluster_id": str, "electrode_group": str})
U = U[U.quality_label.isin(["good", "mua"])].drop_duplicates(KEYS)
fam, size, md, pms = C.families()
R = pms.load_rastermap(md)
neu = pd.read_csv(md / "neuron_cluster_labels_cv.csv").astype({"cluster_id": str, "electrode_group": str})
neu["area_group_rm"] = R["area_group_arr"]
OUT = C.E_SRC


def main():
    # key checks
    m = neu.merge(U[KEYS + ["area_group", "stable"]], on=KEYS, how="left", indicator=True)
    print(f"rastermap neurons {len(neu)}; in decoding unit table: {(m._merge == 'both').sum()}; "
          f"area_group label mismatch among those: {((m._merge == 'both') & (m.area_group != m.area_group_rm)).sum()}")
    print("cluster_id dtype rastermap", neu.cluster_id.dtype, "example", neu.cluster_id.iloc[:3].tolist(),
          "| unit table", U.cluster_id.iloc[:3].tolist())
    rows = []
    for key in ("wdet", "adet", "move"):
        folder = C.PSEUDO_SRC[key]
        wf = sorted((C.RERUN / folder / "weights").glob("area_group__*__N200.parquet"))
        W = pd.concat([pd.read_parquet(f, columns=KEYS + ["area", "n_samples"]) for f in wf], ignore_index=True)
        W = W.astype({"cluster_id": str, "electrode_group": str})
        rf = {f.name.split("__")[1].replace("_", " "): f for f in (C.RERUN / folder / "raw").glob("area_group__*__N200.parquet")}
        mm = m.merge(W[KEYS + ["n_samples"]], on=KEYS, how="left")
        for ag, g in mm[mm._merge == "both"].groupby("area_group"):
            n_tab = int((U.area_group == ag).sum())
            sess_units = U[U.area_group == ag].groupby("session_id").size()
            if ag in rf:
                r = pd.read_parquet(rf[ag], columns=["n_eligible_sessions", "sessions"])
                E = int(r.n_eligible_sessions.iloc[0])
                elig = set(np.concatenate(r.sessions.dropna().map(list).to_numpy()))
                n_iter = len(r)
            else:
                E, elig, n_iter = 0, set(), 0
            # expected P(never drawn) per neuron: [(1 - 1/E + (1/E) (1 - 1/n_s)^10)^20]^n_iter, sessions seen in draws
            if E:
                ns = g.session_id.map(sess_units).to_numpy(float)
                pnever = np.where(g.session_id.isin(elig), ((1 - 1 / E + (1 / E) * (1 - 1 / ns) ** 10) ** 20) ** n_iter, 1.0)
                exp_cov = float(np.mean(1 - pnever))
            else:
                exp_cov = 0.0
            got = g.n_samples.notna()
            rows.append(dict(decoder=key, area_group=ag, units_table=n_tab, sessions_with_units=int(len(sess_units)),
                             eligible_sessions=E, iterations=n_iter, rastermap_neurons=len(g),
                             rm_in_sessions_ever_drawn=int(g.session_id.isin(elig).sum()),
                             rm_with_weight=int(got.sum()), frac=float(got.mean()), expected_frac=exp_cov,
                             median_samples=float(g.n_samples[got].median()) if got.any() else np.nan))
        d = pd.DataFrame([r for r in rows if r["decoder"] == key])
        print(f"\n{key}: rastermap neurons with a weight {d.rm_with_weight.sum()} / {d.rastermap_neurons.sum()} "
              f"(expected if never drawn {np.sum(d.expected_frac * d.rastermap_neurons):.0f}); weight rows {len(W)}; "
              f"n_samples per weighted neuron median {W.n_samples.median():.0f}, IQR {W.n_samples.quantile(.25):.0f}-"
              f"{W.n_samples.quantile(.75):.0f}")
        print(d.drop(columns="decoder").round(3).sort_values("rastermap_neurons", ascending=False).to_string(index=False))
    pd.DataFrame(rows).to_csv(OUT / "coverage_by_area.csv", index=False)

    # movement sanity
    st = pd.read_csv(C.E_SRC / "option_e_cluster_stats.csv")
    mv = st[st.decoder == "move"]
    s = mv.groupby("metacluster").agg(clusters=("cluster", "size"), sig_z3=("sig_z3", "sum"), sig_fdr=("sig_fdr", "sum"),
                                     max_z=("peak_z", "max"), median_peak_t_ms=("peak_t_s", lambda x: 1000 * x.median()))
    print("\nmovement (jaw) per metacluster:\n", s.round(1).to_string())
    for key in ("wdet", "move"):
        oth = st[st.decoder == key].groupby("metacluster").sig_z3.sum()
        print(key, "sig z>=3 in MC2/3/4:", int(oth.reindex([2, 3, 4]).fillna(0).sum()), "of", int(oth.sum()))
    cur = C._move_jaw_curves()
    pk = cur[cur.t.between(-0.5, 0.3)].groupby("area_group").apply(
        lambda g: pd.Series(dict(peak=g.acc_minus_null.max(), t_peak_ms=1000 * g.t.iloc[int(np.argmax(g.acc_minus_null))],
                                 at_0=float(g.acc_minus_null[np.isclose(g.t, 0)].iloc[0]))))
    print("\nmovement area curves (accuracy - null, -0.5..0.3 s):\n", pk.sort_values("peak", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
