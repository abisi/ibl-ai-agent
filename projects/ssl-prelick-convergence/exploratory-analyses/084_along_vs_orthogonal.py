"""084 -- Do whisker hits move along the reward-lick axis, rotate around it, or grow orthogonally? Within day and across days.

User request (2026-10-08): one figure summarising whether R+ / R- whisker-hit (WH) pre-lick activity moves along the
SL -> AH axis and / or rotates (orthogonalises), with equations, schematics and results.

Geometry (both parts, per session, cross-validated, units z-scored): v = WH - SL, Delta = AH - SL (pre-lick population
means). Normalised by the axis length |Delta|:
  lambda = (v . Delta) / |Delta|^2 = rho cos(theta)        component along the axis (0 = SL, 1 = AH)
  omega  = sqrt(o^2) / |Delta| = rho sin(theta)            orthogonal component, o^2 = |v|^2 - (v . Delta)^2 / |Delta|^2
  rho    = |v| / |Delta|,   theta = atan2(omega, lambda)
Across days (Part I, 057 split-half cross-validated squared distances per unit, whole brain, variant all):
  v . Delta = (d(WH,SL) + d(AH,SL) - d(WH,AH)) / 2, |v|^2 = d(WH,SL), |Delta|^2 = d(AH,SL) (sessions with d(AH,SL) >= 0.01);
  stages: day 0 vs expert. o^2 < 0 (cross-validation noise) is clipped at 0 for omega and theta.
Within day (Part II, 025 halves, early tau < 0.5 vs late): lambda_h = rho_h cos(theta_h), omega_h = rho_h sin(theta_h), from
  the 025 cross-validated |v|, projection and |Delta| of each half (sessions with >= 3 WH per half).
Statistics (unit = session): within day, late - early per cohort (Wilcoxon), R+ vs R- (Mann-Whitney U, Welch t, and on
  day 0 the mouse-level cohort permutation); across days, expert - day 0 per cohort (MWU, Welch) and the cohort x stage
  interaction (mouse-level permutation, 001). Learners.
Output: across_days/<ref>/publication/learners/along_vs_orthogonal.{png,pdf,svg} + along_vs_orthogonal_stats.csv
"""
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
COH, CL = m62.COH, m62.CL
OUT = m51.OUTROOT / "publication" / "learners"
MEAS = [("lam", "Along λ = ρ cos θ"), ("omega", "Orthogonal ω = ρ sin θ"), ("rho", "Length ρ = |v|/|Δ|"), ("theta", "Angle θ (°)")]


def across_days():
    L = pd.read_csv(m51.OUTROOT / "learners" / "lambda_sessions_learners.csv")
    L = L[(L.level == "all") & (L.variant == "all") & (L.d_AH_FA >= 0.01)].copy()
    along = (L.d_WH_FA + L.d_AH_FA - L.d_WH_AH) / 2
    L["lam"] = along / L.d_AH_FA
    o2 = L.d_WH_FA - along ** 2 / L.d_AH_FA
    L["omega"] = np.sqrt(np.clip(o2, 0, None) / L.d_AH_FA)
    L["rho"] = np.sqrt(np.clip(L.d_WH_FA, 0, None) / L.d_AH_FA)
    L["theta"] = np.degrees(np.arctan2(L.omega, L.lam))
    return L


def within_day():
    D = pd.read_csv(m51.WITHIN / "cd_normalisation" / "sessions.csv")
    D = m61.learner_filter(D)
    for h in ("early", "late"):
        D[f"lam_{h}"] = D[f"rho_{h}"] * np.cos(np.radians(D[f"theta_{h}"]))
        D[f"omega_{h}"] = D[f"rho_{h}"] * np.sin(np.radians(D[f"theta_{h}"]))
    for q, _ in MEAS:
        D[f"{q}_change"] = D[f"{q}_late"] - D[f"{q}_early"]
    return D


def tests(L, D):
    rows = []
    for q, lab in MEAS:
        for st in ("learning", "expert"):                       # within day
            a = D[(D.cohort == "R+") & (D.stage == st)][f"{q}_change"].dropna()
            b = D[(D.cohort == "R-") & (D.stage == st)][f"{q}_change"].dropna()
            r = dict(part="within day (late - early)", stage=st, measure=q, n_Rp=len(a), n_Rm=len(b), mean_Rp=a.mean(),
                     sem_Rp=a.sem(), mean_Rm=b.mean(), sem_Rm=b.sem())
            if len(a) >= 3 and len(b) >= 3:
                r.update(p_Rp_vs0=stats.wilcoxon(a).pvalue, p_Rm_vs0=stats.wilcoxon(b).pvalue,
                         p_MWU=stats.mannwhitneyu(a, b).pvalue, p_Welch=stats.ttest_ind(a, b, equal_var=False).pvalue)
                if st == "learning":
                    r["diff"], r["p_perm"] = m001.perm_interaction(D.assign(_v=D[f"{q}_change"]), "_v", kind="learning")
            rows.append(r)
        r = dict(part="across days (expert - day 0)", stage="both", measure=q)  # across days
        for c, k in (("R+", "Rp"), ("R-", "Rm")):
            x = L[(L.cohort == c) & (L.stage == "learning")][q].dropna(); y = L[(L.cohort == c) & (L.stage == "expert")][q].dropna()
            r.update({f"day0_{k}": x.mean(), f"day0_sem_{k}": x.sem(), f"exp_{k}": y.mean(), f"exp_sem_{k}": y.sem(),
                      f"n_day0_{k}": len(x), f"n_exp_{k}": len(y), f"mean_{k}": y.mean() - x.mean(),
                      f"p_{k}_MWU": stats.mannwhitneyu(x, y).pvalue, f"p_{k}_Welch": stats.ttest_ind(x, y, equal_var=False).pvalue})
        r["diff"], r["p_perm"] = m001.perm_interaction(L, q, kind="stage")
        rows.append(r)
    return pd.DataFrame(rows)


def fp(p):
    return "n.s." if not np.isfinite(p) or p >= 0.05 else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


def plane(ax, pts, title, lim=(1.25, 1.1)):
    """pts: list of (cohort, x0, y0, x1, y1, sx0, sy0, sx1, sy1, label0, label1)"""
    ax.plot(0, 0, "o", color=CL["FA"], ms=5, zorder=5); ax.plot(1, 0, "o", color=CL["AH"], ms=5, zorder=5)
    ax.annotate("", (1, 0), (0, 0), arrowprops=dict(arrowstyle="->", color=CL["AH"], lw=0.9))
    ax.text(0, -0.07, "SL", ha="center", va="top", fontsize=5); ax.text(1, -0.07, "AH", ha="center", va="top", fontsize=5, color=CL["AH"])
    R_ = 0.92 * min(lim)
    for th in (30, 60):
        t = np.radians(th)
        ax.plot([0, 3 * np.cos(t)], [0, 3 * np.sin(t)], ":", color="0.85", lw=0.6)
        ax.text(R_ * np.cos(t), R_ * np.sin(t), f"{th}°", fontsize=4.2, color="0.6", ha="right")
    for c, x0, y0, x1, y1, sx0, sy0, sx1, sy1 in pts:
        col = COH[c]
        ax.errorbar(x0, y0, xerr=sx0, yerr=sy0, fmt="o", mfc="white", mec=col, ecolor=col, ms=3.6, lw=0.7, zorder=4)
        ax.errorbar(x1, y1, xerr=sx1, yerr=sy1, fmt="o", color=col, ms=3.6, lw=0.7, zorder=4)
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, shrinkA=3, shrinkB=3))
    ax.set_aspect("equal"); ax.set_xlim(-0.15, lim[0]); ax.set_ylim(-0.15, lim[1])
    ax.set_xlabel("along the axis  λ = ρ cos θ", fontsize=5.2); ax.set_ylabel("orthogonal  ω = ρ sin θ", fontsize=5.2)
    ax.set_title(title, fontsize=5.8); ax.tick_params(labelsize=4.6)


def change_panel(ax, T, part, stage, title):
    x = np.arange(len(MEAS))
    for j, (q, _) in enumerate(MEAS):
        r = T[(T.part.str.startswith(part)) & (T.stage == stage) & (T.measure == q)].iloc[0]
        scale = 1 / 45 if q == "theta" else 1.0                  # theta on its own scale (degrees / 45)
        for c, k, dx in (("R+", "Rp", -0.13), ("R-", "Rm", 0.13)):
            m, s = r[f"mean_{k}"], (r[f"sem_{k}"] if f"sem_{k}" in r and np.isfinite(r.get(f"sem_{k}", np.nan)) else np.nan)
            if part.startswith("across"):
                s = np.sqrt(r[f"day0_sem_{k}"] ** 2 + r[f"exp_sem_{k}"] ** 2)
                pv = r[f"p_{k}_MWU"]
            else:
                pv = r[f"p_{k}_vs0"]
            ax.errorbar(j + dx, m * scale, yerr=s * scale, fmt="o", color=COH[c], ms=3, lw=0.8, mfc=COH[c] if pv < 0.05 else "white")
        pt = r.get("p_perm", np.nan) if np.isfinite(r.get("p_perm", np.nan)) else r.get("p_MWU", np.nan)
        ax.text(j, 1.0, fp(pt), transform=ax.get_xaxis_transform(), fontsize=4.3, ha="center", va="bottom")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xticks(x, ["Δλ\nalong", "Δω\northogonal", "Δρ\nlength", "Δθ / 45°\nangle"], fontsize=4.8)
    ax.set_ylabel("change (axis units)", fontsize=5.2); ax.tick_params(labelsize=4.6)
    ax.set_title(title, fontsize=5.8, pad=9)


def figure(L, D, T):
    plt = m62.setup()
    fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 1.05))
    gs = fig.add_gridspec(4, 4, hspace=0.95, wspace=0.55, left=0.07, right=0.98, top=0.95, bottom=0.05,
                          height_ratios=[0.85, 1.1, 1.1, 0.25])
    # ---- row 1: schematics
    ax = fig.add_subplot(gs[0, 0])
    W = np.r_[0.55, 0.45]
    ax.annotate("", (1, 0), (0, 0), arrowprops=dict(arrowstyle="->", color=CL["AH"], lw=1))
    ax.annotate("", W, (0, 0), arrowprops=dict(arrowstyle="->", color=CL["WH"], lw=1.1))
    ax.plot([W[0], W[0]], [0, W[1]], ":", color="0.4", lw=0.7)
    ax.plot([0, W[0]], [-0.04, -0.04], color="#d94801", lw=2)
    a = np.linspace(0, np.arctan2(W[1], W[0]), 20); ax.plot(0.15 * np.cos(a), 0.15 * np.sin(a), color="0.3", lw=0.6)
    for P, c, t in (((0, 0), CL["FA"], "SL"), ((1, 0), CL["AH"], "AH"), (W, CL["WH"], "WH")):
        ax.plot(*P, "o", color=c, ms=4); ax.text(P[0], P[1] + 0.06, t, fontsize=5, ha="center", color=c if t != "WH" else "#b07e00")
    ax.text(0.17, 0.05, "θ", fontsize=5.5); ax.text(0.2, 0.3, "v", fontsize=5.5, color="#b07e00")
    ax.text(W[0] / 2, -0.1, "λ", fontsize=5.5, color="#d94801", ha="center", va="top")
    ax.text(W[0] + 0.03, W[1] / 2, "ω", fontsize=5.5, color="0.3")
    ax.text(0.75, 0.03, "Δ", fontsize=5.5, color=CL["AH"])
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.2, 0.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Geometry (axis length |Δ| = 1)", fontsize=5.8)
    eqs = [r"$v = \bar x_{WH} - \bar x_{SL},\ \ \Delta = \bar x_{AH} - \bar x_{SL}$",
           r"$\lambda = \frac{v\cdot\Delta}{|\Delta|^2} = \rho\cos\theta$   (along)",
           r"$\omega = \frac{\sqrt{|v|^2 - (v\cdot\Delta)^2/|\Delta|^2}}{|\Delta|} = \rho\sin\theta$",
           r"$\rho = |v|/|\Delta|,\quad \theta = \mathrm{atan2}(\omega, \lambda)$",
           "cross-validated dot products (independent trial sets)"]
    pos = [(0.33, 0.80), (0.33, 0.773), (0.66, 0.80), (0.66, 0.769), (0.33, 0.747)]
    for (x_, y_), e in zip(pos, eqs):
        fig.text(x_, y_, e, fontsize=5.6)
    sc = [("Move along the axis", [(0.25, 0.3), (0.6, 0.3)], "λ ↑, ω =, θ ↓, ρ ↑"),
          ("Rotate / orthogonalise", [(0.35, 0.25), (0.35, 0.6)], "λ =, ω ↑, θ ↑, ρ ↑\n(or θ ↓ toward AH)"),
          ("Amplify (scale)", [(0.25, 0.2), (0.55, 0.44)], "λ ↑, ω ↑, θ =, ρ ↑")]
    for j, (ttl, (E, Lt), sig) in enumerate(sc):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.annotate("", (1, 0), (0, 0), arrowprops=dict(arrowstyle="->", color=CL["AH"], lw=0.9))
        ax.plot(0, 0, "o", color=CL["FA"], ms=3.5); ax.plot(1, 0, "o", color=CL["AH"], ms=3.5)
        ax.annotate("", E, (0, 0), arrowprops=dict(arrowstyle="->", color=CL["WH"], lw=0.9, ls="--"))
        ax.annotate("", Lt, (0, 0), arrowprops=dict(arrowstyle="->", color=CL["WH"], lw=1.2))
        ax.plot(*E, "o", mfc="white", mec=CL["WH"], ms=3.5); ax.plot(*Lt, "o", color=CL["WH"], ms=3.5)
        ax.annotate("", Lt, E, arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.8, shrinkA=3, shrinkB=3))
        ax.text(0.5, -0.12, sig, ha="center", va="top", fontsize=5)
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.4, 0.75); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(ttl, fontsize=5.8)
    # ---- row 2: within day
    pts = {}
    for st in ("learning", "expert"):
        pts[st] = []
        for c in ("R+", "R-"):
            d = D[(D.cohort == c) & (D.stage == st)][["lam_early", "omega_early", "lam_late", "omega_late"]].dropna()
            m, s = d.mean(), d.sem()
            pts[st].append((c, m.lam_early, m.omega_early, m.lam_late, m.omega_late, s.lam_early, s.omega_early, s.lam_late, s.omega_late))
    pa = []
    for c in ("R+", "R-"):
        x = L[(L.cohort == c) & (L.stage == "learning")]; y = L[(L.cohort == c) & (L.stage == "expert")]
        pa.append((c, x.lam.mean(), x.omega.mean(), y.lam.mean(), y.omega.mean(), x.lam.sem(), x.omega.sem(), y.lam.sem(), y.omega.sem()))
    allp = pts["learning"] + pts["expert"] + pa
    ymax = max(max(q[2] + q[6], q[4] + q[8]) for q in allp) + 0.12
    xmax = max(1.25, max(max(q[1] + q[5], q[3] + q[7]) for q in allp) + 0.12)
    lim = (xmax, ymax)
    n = {(c, st): int(D[(D.cohort == c) & (D.stage == st)][["lam_early", "lam_late"]].dropna().shape[0])
         for c in ("R+", "R-") for st in ("learning", "expert")}
    plane(fig.add_subplot(gs[1, 0]), pts["learning"],
          f"Within day 0: early (open) → late\nR+ n={n[('R+', 'learning')]}, R− n={n[('R-', 'learning')]}", lim)
    plane(fig.add_subplot(gs[1, 1]), pts["expert"],
          f"Within expert sessions: early → late\nR+ n={n[('R+', 'expert')]}, R− n={n[('R-', 'expert')]}", lim)
    change_panel(fig.add_subplot(gs[1, 2]), T, "within", "learning", "Day 0: late − early\n(top: R+ vs R−)")
    change_panel(fig.add_subplot(gs[1, 3]), T, "within", "expert", "Expert: late − early\n(top: R+ vs R−)")
    # ---- row 3: across days
    nn = L.groupby(["cohort", "stage"]).size()
    plane(fig.add_subplot(gs[2, 0]), pa, f"Across days: day 0 (open) → expert\nR+ {nn[('R+', 'learning')]}→{nn[('R+', 'expert')]}, "
          f"R− {nn[('R-', 'learning')]}→{nn[('R-', 'expert')]} sessions", lim)
    change_panel(fig.add_subplot(gs[2, 1]), T, "across", "both", "Across days: expert − day 0\n(top: cohort × stage, perm)")
    # verdict table
    ax = fig.add_subplot(gs[2, 2:]); ax.axis("off")
    rows = [("", "Within day 0", "Within expert", "Across days"),
            ("R+", "moves along: λ ↑ (> R−),\nθ ↓; ω, ρ ↑ (as R−)", "λ ↑, θ ↓ (as R−)", "amplifies: λ ↑, ω ↑, ρ ↑,\nθ unchanged"),
            ("R−", "grows orthogonally:\nω ↑, λ flat, θ n.s.", "ω ↑, θ ↓ (as R+)", "no change")]
    tb = ax.table(cellText=[r[1:] for r in rows[1:]], rowLabels=[r[0] for r in rows[1:]], colLabels=rows[0][1:],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(4.7); tb.scale(1, 3.3)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("0.75"); cell.set_linewidth(0.4)
        if j == -1:
            cell.get_text().set_color(COH["R+" if i == 1 else "R-"]); cell.get_text().set_fontweight("bold")
    ax.set_title("Summary (filled dots: change ≠ 0, p < 0.05)", fontsize=5.8, pad=14)
    # ---- row 4: notes
    ax = fig.add_subplot(gs[3, :]); ax.axis("off")
    txt = ("Within day: 025 cross-validated halves (τ < 0.5 vs ≥ 0.5, ≥ 3 WH per half; 20 random A/B splits of AH and SL); dots: per-cohort "
           "change vs 0 (Wilcoxon; filled p < 0.05); top: R+ vs R− (day 0: mouse-level permutation; expert: Mann-Whitney U).\n"
           "Across days: 057 split-half cross-validated squared distances (whole brain); dots: expert − day 0 per cohort (filled: MWU p < 0.05); "
           "top: cohort × stage, mouse-level permutation. Negative cross-validated o² clipped at 0.\n"
           "θ is noisy when |v| is small or WH are few; axis noise biases θ upward equally in both halves / stages. Within-day (e, f) and across-day (i) positions come from different estimators (halves with few WH vs whole sessions): compare changes, not positions. Learners, stable units, SL reference.")
    ax.text(0, 1, txt, fontsize=4.6, va="top", wrap=True)
    for a_, l_ in zip([fig.axes[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)], "abcdefghijk"):
        bb = a_.get_position(); fig.text(bb.x0 - 0.03, bb.y1 + 0.025, l_, fontsize=8, fontweight="bold")
    m62.save(fig, OUT, "along_vs_orthogonal")
    plt.close(fig)


def main():
    L, D = across_days(), within_day()
    T = tests(L, D)
    T.to_csv(OUT / "along_vs_orthogonal_stats.csv", index=False)
    print(T.round(4).to_string())
    figure(L, D, T)
    print("ALL DONE", OUT / "along_vs_orthogonal.png")


if __name__ == "__main__":
    main()
