"""035 -- shared whisker generalisation g_W0 across cohorts (user 2026-10-29: "g_W0 should be the same across cohorts").

The cohort likelihoods add and g_W0 is their only shared parameter, so the joint maximum-likelihood value is the minimum of
nll_R+(g) + nll_R-(g), each cohort refitted with g_W0 fixed (027 --variants profiles-g -> fits/027j_fits.csv). The grid minimum is
refined by a parabola through it and its two neighbours. Prints the shared value (used by 027 --shared-g) and, if fits/027k_fits.csv
exists, the alpha_W profiles at the shared value (profSimpleS-*) next to the free-g profiles (profSimple-*, 027i).
Output: tables/035_shared_g.csv, figures/pub_simple/figS6_shared_g.{png,pdf,svg}
Run (haas): python 035_shared_g_profile.py [--print-only]
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np
import pandas as pd

import common as C

F = importlib.import_module("026_pub_figures")
plt = F.plt


def profile(fits="027j_fits.csv", prefix="profG-"):
    P = pd.concat([pd.read_csv(C.FITS / f) for f in fits.split(",")], ignore_index=True)
    P = P[P.variant.str.startswith(prefix)].copy()
    P["g"] = P.variant.str.split("-").str[-1].astype(float)
    T = P.pivot_table(index="g", columns="cohort_set", values="nll").sort_index()
    T["sum"] = T["R+"] + T["R-"]
    g = T.index.to_numpy(); y = T["sum"].to_numpy()
    i = int(np.argmin(y))
    gs = g[i]
    if 0 < i < len(g) - 1:
        a, b, c = np.polyfit(g[i - 1:i + 2], y[i - 1:i + 2], 2)
        if a > 0:
            gs = float(np.clip(-b / (2 * a), g[i - 1], g[i + 1]))
    return T, round(gs, 3)


def alpha_profiles():
    out = {}
    for f, pre in (("027i_fits.csv", "profSimple-"), ("027k_fits.csv", "profSimpleS-")):
        if not (C.FITS / f).exists():
            continue
        P = pd.read_csv(C.FITS / f)
        P = P[P.variant.str.startswith(pre)].copy()
        P["a"] = P.variant.str.split("-").str[-1].astype(float)
        out[pre] = P
    return out


def figure(T, gs, A):
    fig, axs = plt.subplots(1, 3, figsize=(F.W, 2.1), gridspec_kw=dict(wspace=0.45, left=0.08, right=0.985, top=0.83, bottom=0.22))
    ax = axs[0]
    for cs in F.COH:
        ax.plot(T.index, T[cs] - T[cs].min(), "o-", color=F.COH[cs], ms=2.5, lw=0.8, label=F.cs_lab(cs) if hasattr(F, "cs_lab") else cs)
    ax.plot(T.index, T["sum"] - T["sum"].min(), "o-", color="0.2", ms=2.5, lw=0.8, label="sum (shared)")
    ax.axvline(gs, color="0.5", lw=0.5, ls="--")
    ax.set_ylim(-5, max(60, 1.1 * min(400, (T["sum"] - T["sum"].min()).max()))); ax.set_xlabel("g_W0 (fixed)")
    ax.set_ylabel("Δ nll (nats)"); ax.legend(loc="upper left"); ax.set_title(f"shared optimum g_W0 = {gs:.3f}", fontsize=6)
    for ax, cs in zip(axs[1:], F.COH):
        for pre, ls, lab in (("profSimple-", ":", "g_W0 per cohort"), ("profSimpleS-", "-", "g_W0 shared")):
            if pre not in A:
                continue
            P = A[pre][A[pre].cohort_set == cs].sort_values("a")
            ax.plot(P.a, P.nll - P.nll.min(), ls=ls, marker="o", ms=2.2, lw=0.8, color=F.COH[cs], label=lab)
        ax.set_xscale("log"); ax.set_ylim(-5, 150); ax.set_xlabel("α_W (fixed)"); ax.set_ylabel("Δ nll (nats)")
        ax.set_title(f"{cs}: whisker learning-rate profile", fontsize=6); ax.legend(loc="upper left")
    F.letter_row(fig, list(axs), "abc")
    F.OUT = C.FIG / "pub_simple"; F.OUT.mkdir(exist_ok=True)
    F.save(fig, "figS6_shared_g")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("--fits", default="027j_fits.csv", help="comma-separated fits files holding the g_W0 profile")
    ap.add_argument("--prefix", default="profG-", help="profile variant prefix (core combos: profC-<tag>-)")
    ap.add_argument("--tag", default="", help="output suffix (tables/035<tag>_shared_g.csv); with a tag only the table is written")
    a = ap.parse_args()
    T, gs = profile(a.fits, a.prefix)
    if a.print_only:
        print(gs)
    elif a.tag:
        print(T.round(2).to_string())
        T.reset_index().assign(g_shared=gs).to_csv(C.TAB / f"035{a.tag}_shared_g.csv", index=False)
        print("shared g_W0 =", gs)
    else:
        print(T.round(2).to_string())
        A = alpha_profiles()
        rows = T.reset_index().assign(g_shared=gs)
        rows.to_csv(C.TAB / "035_shared_g.csv", index=False)
        for pre, P in A.items():
            print(pre, P.pivot_table(index="a", columns="cohort_set", values="nll").round(2).to_string())
        figure(T, gs, A)
        print("shared g_W0 =", gs)
