"""Decoder transfer sensitivity sweep (whole brain, all trials): which unit selection / readout makes the AH-vs-FA
decoder transfer to whisker hits most sensitive?

Unit sets (pre-lick 100 ms window, rates baseline-corrected): base = good + mua, mean raw rate >= 0.1 Hz (current);
good = quality good only (passes all metrics incl. the joint drift check, i.e. stable units); fr1 / fr2 = good + mua with
mean raw pre-lick rate >= 1 / 2 Hz; good_fr1 = good and >= 1 Hz.
Readouts (stratified k-fold over AH / FA trials, k = min(5, smallest class); units z-scored on training folds; L2
logistic regression C = 0.05, balanced classes; WH trials never used for fitting):
  bin   fraction of WH decoded as AH, normalised (WH - FPR) / (TPR - FPR) (current "transfer_n")
  prob  continuous: (mean P(AH | WH) - mean P(AH | FA_test)) / (mean P(AH | AH_test) - mean P(AH | FA_test))
  pca   prob readout after within-fold PCA (10 components fitted on training trials)
  topk  prob readout after within-fold selection of the 100 units with the largest |t| (AH vs FA, training trials)
Values clipped to [-1, 2]; kept if >= 5 units and held-out balanced accuracy >= 0.55 (bin / prob share the fit).
Statistics: 062 group_stats (session = unit; MWU, Welch; interaction by mouse-level permutation).
Output: combined_results_ks4/ssl-prelick-convergence/across_days/fa/decoder_sweep/ (sweep_sessions.csv, sweep_stats.csv, sweep_summary.png)
Note: this is a sensitivity analysis; choosing the best variant post hoc inflates false positives.
"""
import importlib
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m62 = importlib.import_module("062_pub_convergence_figures")
OUT = m51.OUTROOT / "decoder_sweep"
UNIT_SETS = {"base": (("good", "mua"), 0.1), "good": (("good",), 0.1), "fr1": (("good", "mua"), 1.0),
             "fr2": (("good", "mua"), 2.0), "good_fr1": (("good",), 1.0)}
READOUTS = ["bin", "prob", "pca", "topk"]


def decode(Z, lab, rng):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    m = np.isin(lab, ["AH", "FA"]); w = lab == "WH"; y = (lab[m] == "AH").astype(int)
    k = min(5, int(y.sum()), int((1 - y).sum()))
    if k < 3 or w.sum() < 3:
        return None
    Xa, Xw = Z[m], Z[w]
    out = {}
    for ro in ["base", "pca", "topk"]:
        pred = np.zeros(len(y)); prob = np.zeros(len(y)); wb, wp = [], []
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=int(rng.integers(1e9))).split(Xa, y):
            mu, sd = Xa[tr].mean(0), Xa[tr].std(0); sd[sd == 0] = 1
            A, B, C = (Xa[tr] - mu) / sd, (Xa[te] - mu) / sd, (Xw - mu) / sd
            if ro == "pca":
                pc = PCA(min(10, A.shape[1], A.shape[0] - 1)).fit(A); A, B, C = pc.transform(A), pc.transform(B), pc.transform(C)
            elif ro == "topk" and A.shape[1] > 100:
                a1, a0 = A[y[tr] == 1], A[y[tr] == 0]
                t = (a1.mean(0) - a0.mean(0)) / np.sqrt(a1.var(0) / len(a1) + a0.var(0) / len(a0) + 1e-9)
                sel = np.argsort(-np.abs(t))[:100]; A, B, C = A[:, sel], B[:, sel], C[:, sel]
            clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000).fit(A, y[tr])
            pred[te] = clf.predict(B); prob[te] = clf.predict_proba(B)[:, 1]
            wb.append(clf.predict(C).mean()); wp.append(clf.predict_proba(C)[:, 1].mean())
        tpr, fpr = pred[y == 1].mean(), pred[y == 0].mean(); bacc = (tpr + 1 - fpr) / 2
        pa, pf = prob[y == 1].mean(), prob[y == 0].mean()
        ok = bacc >= 0.55
        if ro == "base":
            out["bacc"] = bacc
            out["bin"] = float(np.clip((np.mean(wb) - fpr) / (tpr - fpr), -1, 2)) if ok and tpr - fpr > 0 else np.nan
            out["prob"] = float(np.clip((np.mean(wp) - pf) / (pa - pf), -1, 2)) if ok and pa - pf > 0 else np.nan
        else:
            out[f"bacc_{ro}"] = bacc
            out[ro] = float(np.clip((np.mean(wp) - pf) / (pa - pf), -1, 2)) if ok and pa - pf > 0 else np.nan
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    f = OUT / "sweep_sessions.csv"
    if f.exists():
        R = pd.read_csv(f)
    else:
        rng = np.random.default_rng(0); rows = []
        for i, r in enumerate(ss.itertuples()):
            z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
            K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
            K = K.merge(W[W.session_id == r.session_id][["electrode_group", "cluster_id", "quality_label", "cohort", "stage",
                                                          "mouse_id"]], on=["electrode_group", "cluster_id"], how="left")
            if K.cohort.isna().all():
                continue
            meta = dict(session_id=r.session_id, mouse_id=K.mouse_id.dropna().iloc[0], cohort=K.cohort.dropna().iloc[0],
                        stage=K.stage.dropna().iloc[0])
            X, raw, lab = z["rates"].astype(float), z["raw"].astype(float), z["cls"]
            fr = raw.mean(1)
            for name, (qs, frmin) in UNIT_SETS.items():
                u = K.quality_label.isin(qs).to_numpy() & (fr >= frmin)
                if u.sum() < 5:
                    continue
                d = decode(X[u].T, lab, rng)
                if d:
                    rows.append(dict(meta, unit_set=name, n_units=int(u.sum()), **d))
            if i % 20 == 0:
                print(f"[063] {i + 1}/{len(ss)}", flush=True)
        R = pd.DataFrame(rows); R.to_csv(f, index=False)
    rng = np.random.default_rng(1)
    st = []
    for name in UNIT_SETS:
        for ro in READOUTS + ["bacc"]:
            d = R[R.unit_set == name]
            S = m62.group_stats(d, ro, f"{name}:{ro}", rng, "sweep")
            st.append(dict(unit_set=name, readout=ro, **{f"mean {m62.GLAB[k]}": S["means"][k] for k in m62.GROUPS},
                           **{f"n {m62.GLAB[k]}": S["n"][k] for k in m62.GROUPS},
                           p_Rp_LE=S.get("R+ L vs E", (np.nan,))[0], p_Rm_LE=S.get("R- L vs E", (np.nan,))[0],
                           p_expert=S.get("expert R+ vs R-", (np.nan,))[0], p_interaction=S["interaction"][1],
                           median_units=d.n_units.median()))
    T = pd.DataFrame(st); T.to_csv(OUT / "sweep_stats.csv", index=False)
    plt = m62.setup()
    fig, axs = plt.subplots(1, 4, figsize=(m62.W_IN, 2.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    T2 = T[T.readout != "bacc"]
    labels = [f"{u} · {r}" for u, r in zip(T2.unit_set, T2.readout)]
    y = np.arange(len(T2))
    for ax, (col, ttl) in zip(axs, [("p_Rp_LE", "R+ learning vs expert"), ("p_Rm_LE", "R− learning vs expert"),
                                    ("p_expert", "Expert R+ vs R−"), ("p_interaction", "Interaction")]):
        v = -np.log10(T2[col].astype(float).to_numpy())
        ax.barh(y, v, color=["#2c2cdb" if r != "bin" else "0.6" for r in T2.readout], height=0.7)
        ax.axvline(-np.log10(0.05), color="k", lw=0.6, ls=(0, (2, 2)))
        ax.set_title(ttl); ax.set_xlabel("−log10 p")
    axs[0].set_yticks(y, labels, fontsize=4.8); axs[0].invert_yaxis()
    fig.suptitle("Decoder transfer sensitivity (whole brain; grey: current binary readout; dashed: p = 0.05)", fontsize=7,
                 x=0.12, ha="left")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.17)
    m62.save(fig, OUT, "sweep_summary"); plt.close(fig)
    json.dump(dict(script="063_decoder_sweep.py", unit_sets=UNIT_SETS, readouts=READOUTS,
                   note="sensitivity analysis; best variant chosen post hoc would inflate false positives"),
              open(OUT / "provenance.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
