"""Figure conventions (skills/ssl-figure-style; same rcParams as ssl-prelick-convergence 062)."""
import numpy as np

W_IN = 7.4
WH_C, AUD_C = "#f7b519", "#2c2cdb"
# one fixed colour per area (both levels), used in every figure
AREA_C = {
    "Somatosensory-whisker": "#f7b519", "Auditory areas": "#2c2cdb", "Motor areas": "#1b9e77", "Midbrain": "#d95f02",
    "Striatum": "#7570b3", "Thalamus": "#e7298a",
    "SSp-bfd": "#f7b519", "SSs": "#a6761d", "SCm": "#d95f02", "MO-wM1": "#1b9e77", "MO-wM2": "#66a61e",
    "DMS": "#7570b3", "DLS": "#b3a2e0", "MO-ALM": "#0b5d46",
}
SHORT = {"Somatosensory-whisker": "SS-whisker", "Auditory areas": "Auditory", "Motor areas": "Motor"}


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "lines.linewidth": 0.9,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
    return plt


def letter_row(fig, axes, letters, dx_in=0.32, dy_in=0.12):
    W, H = fig.get_size_inches()
    top = max(ax.get_position().y1 for ax in axes)
    for ax, l in zip(axes, letters):
        fig.text(ax.get_position().x0 - dx_in / W, top + dy_in / H, l, fontsize=9, weight="bold", ha="left", va="bottom")


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{name}.{ext}", dpi=300)


def short(a):
    return SHORT.get(a, a)


def corr_panel(ax, x, y, colors=None, sizes=None):
    """scatter + OLS line + 95% CI band (solid line only if p < 0.05); returns Pearson / Spearman stats"""
    from scipy import stats
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[m], np.asarray(y, float)[m]
    ax.scatter(x, y, s=(np.asarray(sizes)[m] if sizes is not None else 8), c=(np.asarray(colors, object)[m].tolist() if colors is not None else "0.3"),
               lw=0.3, edgecolors="white", zorder=3)
    out = dict(n=len(x), r=np.nan, p=np.nan, rho=np.nan, p_rho=np.nan)
    if len(x) < 4:
        return out
    r, p = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    X = np.c_[np.ones(len(x)), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = res @ res / (len(x) - 2)
    xs = np.linspace(x.min(), x.max(), 100)
    Xs = np.c_[np.ones(100), xs]
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xs, s2 * np.linalg.inv(X.T @ X), Xs))
    tq = stats.t.ppf(0.975, len(x) - 2)
    ax.fill_between(xs, Xs @ b - tq * se, Xs @ b + tq * se, color="0.6", alpha=0.25, lw=0, edgecolor="none")
    ax.plot(xs, Xs @ b, color="0.2", lw=0.8, ls="-" if p < 0.05 else "--")
    out.update(r=r, p=p, rho=rho, p_rho=prho)
    return out
