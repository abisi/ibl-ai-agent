"""033 -- publication summary figures for the simple model with whisker generalisation (027 variant simple-gen; user 2026-10-29:
"make the summary figures for simple-gen when done"). Reuses 029 (figure code) and 026 (style), with the simple-gen inputs:
cohort fits 027h, alpha_W profiles 027i (profSimple-*), comparison fits 027c-027h; mouse-level fit 025G (025 --lean), 028_025G metrics.
Output: figures/pub_simple/*.{png,pdf,svg}; tables/033_figure_numbers.csv, tables/033_mouse_table.csv
Run (haas): python 033_pub_figures_simple.py
"""
from __future__ import annotations

import importlib

import common as C

P = importlib.import_module("029_pub_figures_v8")
F = P.F
F.OUT = C.FIG / "pub_simple"
F.OUT.mkdir(exist_ok=True)
P.FINAL = "simple-gen"
P.TAG = "025G"
P.FIT_TAGS = ("027", "027b", "027c", "027d", "027e", "027f", "027h", "027i")
P.NUM_FILE, P.MOUSE_FILE = "033_figure_numbers.csv", "033_mouse_table.csv"
P.VW_TITLE = "Learned whisker value from its generalised start: R+ climbs toward +1, R− falls toward 0"
P.PROF = {"R+": ("profSimple-R+-", "alpha_neg"), "R-": ("profSimple-R--", "alpha_neg")}
P.PROF_AP = None
P.VLAB = [("simple-gen", "simple + whisker generalisation (11 par.)"),
          ("simple", "simple, no generalisation (10)"),
          ("simple-m1", "simple, unrewarded = −1 (10)"),
          ("lean-abs-stim", "previous final (17–18, HMM, lag, κ, θ)"),
          ("lean-abs-stim-nodis", "previous final without HMM"),
          ("lean-abs-stim-hom", "previous final + thirst-scaled reward")]
P.EQ_TITLE = "Go/NoGo Q-learning with thirst-dependent lick drive and whisker generalisation"
P.EQ = [r"$\mathbf{Learning}$ (licked stimulus trials):  $\delta = o - V_s$,   $V_s \leftarrow V_s + \alpha_s\,\delta$",
        r"$o = 1$ water, $0$ no water;   $\alpha_A$ auditory, $\alpha_W$ whisker",
        r"$\mathbf{Generalisation}$: at the first whisker trial $V_W = g_{W0}\,V_A$",
        r"no-stimulus value $V_N$: starts at $w_{N0}$, learned from false alarms ($\alpha_N$)",
        r"$\mathbf{Thirst}$:  $T = \max(0,\,1 - \rho_s\,S)$,  $S$ = water drunk in the session",
        r"$\mathbf{Choice}$:  $P(\mathrm{lick}) = 0.01 + 0.98\,\sigma(z)$,   $z = \beta\,T\,V_s - c + s_B\,T + u_s$",
        r"per session: $\log\rho_s \sim N(\mu_\rho, \sigma_\rho^2)$, $u_s \sim N(0, \sigma_u^2)$ (integrated out)",
        r"same 11 cohort parameters in R+ and R−",
        r"per mouse: $\alpha_A, \alpha_W, \beta, \mu_\rho$ (hierarchical EM, Huys et al. 2011)"]


def forced_rows(Fa):
    """show the whisker rate the cohort did NOT pick: fast (0.2) if its optimum is slow, slow (0.003) if fast"""
    rows, labs = {}, {}
    for cs in ("R+", "R-"):
        best = Fa[(Fa.variant == P.FINAL) & (Fa.cohort_set == cs)].p_alpha_neg.iloc[0]
        v = 0.2 if best < 0.03 else 0.003
        rows[cs] = f"profSimple-{cs}-alpha_W-{v:g}"
        labs[cs] = f"forced α_W = {v:g}"
    return rows, labs


if __name__ == "__main__":
    Fa = P.fits()
    P.FORCED, P.FORCED_LAB = forced_rows(Fa)
    P.main()
    import shutil                                    # single-trial composite from 031 (written next to the v8 figures)
    for ext in ("png", "pdf", "svg"):
        src = C.FIG / "pub_v8" / f"fig4c_single_trials_{P.TAG}.{ext}"
        if src.exists():
            shutil.copyfile(src, F.OUT / src.name)         # NAS: no chmod
