"""SuppFigure3_rastermap plot functions: the source figures' functions (embedding, continuum, areas / cohorts) re-used
as they are, plus the ISI-statistics panel and small compact wrappers."""
import importlib.util
import pathlib

import numpy as np

FIGS = pathlib.Path(__file__).resolve().parents[1]
FS = 5.0
RED = "#e8222e"


def _load(name):
    s = importlib.util.spec_from_file_location(f"src_{name}", FIGS / name / "plot.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


EMB, CONT, AREA = _load("SuppFigure3_embedding"), _load("SuppFigure3_continuum"), _load("SuppFigure3_areas_cohorts")
plot_schematic = EMB.plot_schematic
plot_cv_cluster = EMB.plot_cv_cluster
plot_meta_matrix = EMB.plot_meta_matrix
plot_meta_cut = EMB.plot_meta_cut
plot_margin = CONT.plot_margin
plot_area_sampling = AREA.plot_area_sampling
plot_mouse_yield = AREA.plot_mouse_yield
plot_occupancy = AREA.plot_occupancy
plot_specialization = AREA.plot_specialization


def _style(ax):
    ax.tick_params(labelsize=FS, length=1.5, pad=1, width=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)


def plot_decay(ax, df):
    """r vs distance along the order (rastermap vs random orders) + the neighbouring-cluster test as text"""
    CONT.plot_decay(ax, df)
    r = df.iloc[0]
    p = f"p < {1 / (1 + r.adj_n_perm):.3f}" if r.adj_p <= 1 / (1 + r.adj_n_perm) + 1e-12 else f"p = {r.adj_p:.3g}"
    ax.text(0.97, 0.62, f"neighbours: mean r {r.adj_obs:.2f}\nrandom orders {r.adj_null_mean:.2f}\n({p}, "
            f"{int(r.adj_n_perm)} perm.)", transform=ax.transAxes, fontsize=FS - 0.5, ha="right", va="top")


def plot_isi_stats(ax, df):
    """per neuron: burstiness vs ISI CV (deterministic curve B = (CV - 1) / (CV + 1)) and memory vs ISI CV"""
    ax.axis("off")
    r = df.iloc[0]
    for j, (col, lab) in enumerate((("burstiness", "Burstiness B"), ("memory", "Memory M"))):
        a = ax.inset_axes([0.0, 0.56 - j * 0.56, 1.0, 0.40])
        a.scatter(df.cv, df[col], s=0.6, c=df.color, lw=0, alpha=0.5, rasterized=True)
        xs = np.linspace(max(df.cv.min(), 0.05), np.percentile(df.cv, 99.5), 100)
        if col == "burstiness":
            a.plot(xs, (xs - 1) / (xs + 1), color="k", lw=0.6)
            a.text(0.97, 0.06, "B = (CV − 1) / (CV + 1)", transform=a.transAxes, fontsize=FS - 0.5, ha="right")
        else:
            a.axhline(0, color="0.6", lw=0.3)
            a.text(0.97, 0.94, f"Spearman ρ = {r.rho_cv_memory:.2f}", transform=a.transAxes, fontsize=FS - 0.5,
                   ha="right", va="top")
        a.set_xlim(0, np.percentile(df.cv, 99.5))
        a.set_ylim(*np.percentile(df[col], [0.5, 99.5]))
        a.set_ylabel(lab, fontsize=FS, labelpad=1)
        a.set_xlabel("ISI CV" if j == 1 else "", fontsize=FS, labelpad=1)
        _style(a)
    ax.text(0.0, 1.0, f"{int(r.n_neurons):,} neurons", transform=ax.transAxes, fontsize=FS - 0.5, va="bottom")
