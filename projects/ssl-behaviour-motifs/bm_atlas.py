"""State atlas pages (per-session reports, 007) and mapping / meta-state pages (summary, 008), from 009 tables and the
Kuma video pass (011: mean frame, per-state motion-energy maps) and grid movies (012)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

import bm_lib as L
import bm_states as BS

FEATS_LABEL = {"whisker_setpoint": "whisker set-point", "whisking_amp": "whisking amplitude", "top_nose_dist": "nose pos (top)",
               "top_nose_speed": "nose speed (top)", "me_top_snout": "ME snout (top)", "me_top_whisker_pad": "ME whisker pad",
               "me_top_whiskers": "ME whiskers", "jaw_open": "jaw opening", "tongue_out": "tongue out",
               "nose_dist": "nose pos (side)", "nose_speed": "nose speed (side)", "me_side_snout": "ME snout (side)",
               "me_side_mouth": "ME mouth", "me_side_cheek": "ME cheek"}
ROWS = [(m, v) for m in BS.METHODS for v in L.VIEWS]
SHORT = {"kpms": "kpms", "arhmm": "AR-HMM", "lightning_action": "l-action"}
TRAJ_S, TRAJ_GAIN = 0.5, 5.0


def tables():
    t = {k: pd.read_csv(L.TABLES / f"009_{k}.csv") for k in ("fingerprints_pooled", "fingerprints_session", "metastates",
                                                            "contingency", "hungarian", "agreement_meta", "state_sessions")}
    t["feats"] = [c for c in t["metastates"].columns if c in FEATS_LABEL]
    return t


def state_label(method, s):
    return BS.LA_CLASSES[s] if method == "lightning_action" else f"{'S' if method == 'kpms' else 'A'}{s}"


def fp_heatmap(ax, M, feats, ylabels, title):
    v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -max(v, 0.5), max(v, 0.5)), aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([FEATS_LABEL[f] for f in feats], rotation=50, ha="right", fontsize=5)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=5)
    ax.set_title(title, fontsize=6, loc="left")
    return im


def trajectories(sid, view, z, top):
    k = np.load(L.CACHE / "inputs" / f"{sid}_{view}_keypoints.npz")
    raw = (k["xy"].astype(float) - k["shift"]) / float(k["mm_per_px"])  # v3: aligned mm -> raw pixels
    n = int(TRAJ_S * L.FS_DS)
    out = {}
    ch = np.flatnonzero(np.diff(z) != 0) + 1
    st = np.r_[0, ch]
    ln = np.diff(np.r_[st, len(z)])
    for s in top:
        on = st[(z[st] == s) & (ln >= 20) & (st + n < len(raw)) & (st > 0)]  # bouts >= 100 ms
        if len(on) == 0:
            continue
        seg = np.stack([raw[o:o + n] - raw[o] for o in on[:300]])  # displacement from onset
        out[s] = (np.nanmedian(raw, 0), np.nanmean(seg, 0), len(on))
    return out, [str(b) for b in k["bodyparts"]]


def page_atlas(pdf, sid, method, view, T):
    z = BS.load_states_native(sid, method, view)
    vf = L.RES / "kuma" / "video" / f"{sid}_{view}.npz"
    if z is None or not vf.exists():
        return
    v = np.load(vf)
    keys = [str(q) for q in v["me_keys"]]
    cnt = np.bincount(z[z >= 0])
    top = list(range(1, len(BS.LA_CLASSES))) if method == "lightning_action" else [s for s in np.argsort(-cnt)[:10] if cnt[s] / cnt.sum() >= 0.005]
    S = T["fingerprints_session"]
    S = S[(S.session_id == sid) & (S.method == method) & (S.view == view)].set_index("state")
    names = {r.meta: r["name"] for _, r in T["metastates"].iterrows()}
    feats = T["feats"]
    fig = plt.figure(figsize=(7.4, 9.8))
    ax = fig.add_axes([0.26, 0.72, 0.6, 0.22])
    M = np.array([S.loc[s, feats].to_numpy(float) if s in S.index else np.full(len(feats), np.nan) for s in top])
    usage = np.bincount(z[z >= 0], minlength=max(top) + 1)[top] / max((z >= 0).sum(), 1)
    ylab = [f"{state_label(method, s)} ({100 * u:.0f}%) -> {names.get(int(S.loc[s, 'meta']), '?') if s in S.index else 'rare'}"
            for s, u in zip(top, usage)]
    im = fp_heatmap(ax, M, feats, ylab, f"{sid}: {BS.METHOD_LABEL[method]}, {view} view - state fingerprints "
                    "(mean within-session z of features; row label: state (usage) -> meta-state)")
    plt.colorbar(im, ax=ax, fraction=0.03).set_label("z", fontsize=5)
    tr, bps = trajectories(sid, view, z, top)
    mf = v["mean_frame"].astype(float)
    cols = plt.cm.viridis(np.linspace(0, 1, int(TRAJ_S * L.FS_DS)))
    for i, s in enumerate(top):
        ax = fig.add_axes([0.02 + (i % 5) * 0.196, 0.47 - (i // 5) * 0.2, 0.19, 0.16])
        ax.axis("off")
        key = f"{method}|{int(s)}"
        y0, y1, x0, x1 = [int(b) for b in v["boxes"][list(v["regions"]).index("face")]]
        ax.imshow(mf[y0:y1, x0:x1], cmap="gray")
        if key in keys:
            mm = v["me_maps"][keys.index(key)].astype(float)[y0:y1, x0:x1]
            ax.imshow(np.ma.masked_less(mm, np.percentile(mm, 80)), cmap="hot", alpha=0.6,
                      vmin=np.percentile(mm, 80), vmax=np.percentile(mm, 99.5))
        if s in tr:
            med, seg, nb = tr[s]
            for j in range(len(bps)):
                p = (med[j] + TRAJ_GAIN * seg[:, j]) / 2 - [x0, y0]
                ax.scatter(p[:, 0], p[:, 1], c=cols, s=3, lw=0)
        else:
            nb = 0
        ax.set_title(f"{state_label(method, s)}: {ylab[i].split('-> ')[-1][:28]}\n{nb} bouts", fontsize=5)
    fig.text(0.02, 0.06, "States shown: the 10 most used in this session with >= 0.5 % usage. Panels: mean frame (grey), motion-energy map of the state (hot: mean |frame_t - frame_t-1| over up to "
             "50 sampled bouts, top 20 % of pixels), mean keypoint\ndisplacement over the first 0.5 s after state onset "
             f"(x{TRAJ_GAIN:.0f}, colour = time, dark -> light). Grid movies: reports/movies/{method}_{view}_<state>.mp4",
             fontsize=5)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def summary_pages(pdf, sids, coh):
    T = tables()
    feats, M, P = T["feats"], T["metastates"], T["fingerprints_pooled"]
    # page 1: meta-states and how each method's states map onto them
    fig = plt.figure(figsize=(7.4, 9.8))
    ax = fig.add_axes([0.3, 0.68, 0.6, 0.26])
    im = fp_heatmap(ax, M[feats].to_numpy(float), feats, [f"{r['name']} ({100 * r.frac_frames:.0f}% of units' frames)"
                                                          for _, r in M.iterrows()],
                    "G  Meta-states: Ward clusters of session-state fingerprints of all methods and views (centroids)")
    plt.colorbar(im, ax=ax, fraction=0.03).set_label("z", fontsize=5)
    ax = fig.add_axes([0.08, 0.05, 0.88, 0.52])
    ax.axis("off")
    lines = ["H  Top states of each method x view (pooled over sessions): auto-name from its own fingerprint -> meta-state"]
    for m, v in ROWS:
        g = P[(P.method == m) & (P.view == v)].sort_values("usage", ascending=False).head(10)
        lines.append(f"{SHORT[m]} {v}:")
        for _, r in g.iterrows():
            lines.append(f"   {state_label(m, int(r.state)):<10} {100 * r.usage:4.1f}%  {r['name']:<34} -> {r.meta_name}")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=4.6)
    pdf.savefig(fig)
    plt.close(fig)
    # page 2: contingency matrices with Hungarian matches
    pairs = [(("kpms", "side"), ("kpms", "top")), (("arhmm", "side"), ("arhmm", "top")),
             (("lightning_action", "side"), ("lightning_action", "top")), (("kpms", "side"), ("arhmm", "side")),
             (("kpms", "top"), ("arhmm", "top")), (("kpms", "side"), ("lightning_action", "side"))]
    C, Hh = T["contingency"], T["hungarian"]
    fig, axs = plt.subplots(3, 2, figsize=(7.4, 9.8))
    for ax, (a, b) in zip(axs.ravel(), pairs):
        A, B = f"{a[0]} {a[1]}", f"{b[0]} {b[1]}"
        c = C[(C.A == A) & (C.B == B)]
        h = Hh[(Hh.A == A) & (Hh.B == B)]
        ra, rb = list(dict.fromkeys(c.state_A)), list(dict.fromkeys(c.state_B))
        hm = dict(zip(h.state_A, h.state_B))
        rb = [hm[s] for s in ra if s in hm] + [s for s in rb if s not in hm.values()]  # matched columns first
        mat = c.pivot(index="state_A", columns="state_B", values="p_B_given_A").loc[ra, rb].to_numpy()
        ax.imshow(mat, cmap="magma", vmin=0, vmax=1, aspect="auto")
        for i, s in enumerate(ra):
            if s in hm:
                ax.add_patch(plt.Rectangle((rb.index(hm[s]) - 0.5, i - 0.5), 1, 1, fill=False, ec="c", lw=0.8))
        nm = lambda m, v, s: P[(P.method == m) & (P.view == v) & (P.state == s)].meta.astype(str).tolist()  # noqa: E731
        ax.set_yticks(range(len(ra)))
        ax.set_yticklabels([f"{state_label(a[0], s)} M{(nm(*a, s) or ['?'])[0]}" for s in ra], fontsize=4.5)
        ax.set_xticks(range(len(rb)))
        ax.set_xticklabels([f"{state_label(b[0], s)} M{(nm(*b, s) or ['?'])[0]}" for s in rb], fontsize=4.5, rotation=90)
        ax.set_title(f"P({SHORT[b[0]]} {b[1]} | {SHORT[a[0]]} {a[1]}); cyan = Hungarian match", fontsize=5.5)
    fig.suptitle("I  Contingency between methods / views (synchronous frames, 8 sessions pooled; M = meta-state)", fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    # page 3: meta-state agreement, session-specific states, R+ vs R- in meta-state space
    fig = plt.figure(figsize=(7.4, 9.8))
    AG = T["agreement_meta"].groupby(["A", "B"])[["ami_states", "ami_meta", "frac_same_meta"]].mean().reset_index()
    ax = fig.add_axes([0.08, 0.72, 0.4, 0.22])
    x = np.arange(len(AG))
    ax.bar(x - 0.2, AG.ami_states, 0.4, color="#999999", lw=0, label="original states (AMI)")
    ax.bar(x + 0.2, AG.ami_meta, 0.4, color="#1f77b4", lw=0, label="meta-states (AMI)")
    ax.plot(x, AG.frac_same_meta, "k.", label="same meta-state (fraction of frames)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a.replace('lightning_action', 'l-act')} / {b.replace('lightning_action', 'l-act')}" for a, b in zip(AG.A, AG.B)],
                       rotation=90, fontsize=4)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=4.5)
    ax.set_title("J  Agreement before / after mapping to meta-states", fontsize=6, loc="left")
    S = T["fingerprints_session"]
    ax = fig.add_axes([0.58, 0.72, 0.38, 0.22])
    X = S[feats].fillna(0).to_numpy()
    U, sv, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    pc = U[:, :2] * sv[:2]
    mk = {"kpms": "o", "arhmm": "s", "lightning_action": "^"}
    for m in BS.METHODS:
        q = S.method == m
        ax.scatter(pc[q, 0], pc[q, 1], c=plt.cm.tab10(S.meta[q] % 10), marker=mk[m], s=8, lw=0, label=SHORT[m])
    ax.set_xlabel("fingerprint PC1")
    ax.set_ylabel("fingerprint PC2")
    ax.legend(frameon=False, fontsize=5)
    ax.set_title("K  Session-states (marker = method; colour = meta-state:\nM0 blue, M1 orange, M2 green, M3 red)", fontsize=6, loc="left")
    SP = T["state_sessions"]
    spec = SP[(SP.dominant_share > 0.8) | (SP.n_sessions == 1)]
    lines = ["L  Session-specific states (> 80 % of usage in one session, or present in one session only) and their meta-state:"]
    for (m, v), g in spec.groupby(["method", "view"]):
        bym = g.groupby("metas").apply(lambda q: ", ".join(f"{state_label(m, int(s))}[{ss}]" for s, ss in zip(q.state, q.sessions)))
        for meta, txt in bym.items():
            lines.append(f"  {SHORT[m]} {v:<5} M{meta:<6} {txt[:120]}")
    # R+ vs R- meta usage
    meta = np.load(L.CACHE / "009_meta.npz")
    rows = []
    for sid in sids:
        sg = BS.session_grid(sid)
        t, lk = sg["t"], sg["licks"]
        j = np.clip(np.searchsorted(lk, t), 1, len(lk) - 1)
        nolick = np.minimum(np.abs(t - lk[j - 1]), np.abs(lk[j] - t)) > 1.0
        act = sg["epoch"] != "other"
        for m, v in ROWS:
            zz = meta[f"{sid}|{m}|{v}"]
            for scope, msk in (("all", act & (zz >= 0)), ("no-lick", act & (zz >= 0) & nolick)):
                for k in M.meta:
                    rows.append(dict(session_id=sid, cohort=coh[sid], method=m, view=v, scope=scope, meta=k,
                                     usage=float(np.mean(zz[msk] == k))))
    R = pd.DataFrame(rows)
    R.to_csv(L.TABLES / "009_meta_usage_per_session.csv", index=False)
    tr = []
    for (m, v, sc, k), g in R.groupby(["method", "view", "scope", "meta"]):
        a, b = g[g.cohort == "R+"].usage, g[g.cohort == "R-"].usage
        tr.append(dict(method=m, view=v, scope=sc, meta=k, Rp=a.mean(), Rm=b.mean(), p_mw=stats.mannwhitneyu(a, b).pvalue,
                       p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue))
    TR = pd.DataFrame(tr)
    TR.to_csv(L.TABLES / "009_meta_rplus_rminus_exploratory.csv", index=False)
    best = TR.assign(p=TR[["p_mw", "p_welch"]].min(1)).sort_values("p").head(8)
    lines += ["", "M  EXPLORATORY R+ vs R- meta-state usage (active block; mouse level, n = 4 vs 4, underpowered; MW + Welch, uncorrected).",
              f"   {len(TR)} comparisons; 8 smallest p:"]
    for r in best.itertuples():
        lines.append(f"   {SHORT[r.method]} {r.view:<5} {r.scope:<8} {M.name[r.meta]:<32} R+ {r.Rp:.3f}  R- {r.Rm:.3f}  "
                     f"p_MW {r.p_mw:.3f}  p_Welch {r.p_welch:.3f}")
    ax = fig.add_axes([0.06, 0.03, 0.9, 0.58])
    ax.axis("off")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=4.6)
    pdf.savefig(fig)
    plt.close(fig)
    # page 4: movie index
    mv = sorted((L.REPORTS / "movies").glob("*.mp4")) if (L.REPORTS / "movies").exists() else []
    fig = plt.figure(figsize=(7.4, 9.8))
    ax = fig.add_axes([0.05, 0.03, 0.9, 0.93])
    ax.axis("off")
    lines = ["N  Grid movies (reports/movies/; one bout per session, 2 x 4 tiles, -0.25 to +1 s around onset, 0.5x speed):"]
    for f in mv:
        m, v, s = f.stem.rsplit("_", 2)
        r = P[(P.method == m) & (P.view == v) & (P.state == int(s))]
        nm = f"{r['name'].iloc[0]} -> {r.meta_name.iloc[0]}" if len(r) else ""
        lines.append(f"  {f.name:<34} {nm}")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=4.8)
    pdf.savefig(fig)
    plt.close(fig)
    return TR
