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
SHORT = {"Somatosensory-whisker": "SS-whisker", "Somatosensory-orofacial": "SS-orofacial", "Somatosensory-body": "SS-body",
         "Auditory areas": "Auditory", "Motor areas": "Motor", "Frontal areas": "Frontal", "Retrosplenial areas": "Retrosplenial",
         "Posterior parietal areas": "Post. parietal", "Lateral septal complex": "Lat. septum", "Visual areas": "Visual",
         "Insular areas": "Insular", "Olfactory areas": "Olfactory", "Amygdala and hypothalamus": "Amygdala + hypoth."}


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


def _exp(x, c, a, tau):
    return c + a * np.exp(-x / tau)


def exp_fit(x, y):
    """onset = c + a exp(-x / tau) by least squares; None if it does not converge"""
    from scipy.optimize import curve_fit
    try:
        p, _ = curve_fit(_exp, x, y, p0=(np.min(y), np.ptp(y) + 1e-6, max(np.ptp(x) / 3, 1e-3)),
                         bounds=([-np.inf, 0, 1e-4], [np.inf, np.inf, 10]), maxfev=20000)
        return p
    except Exception:
        return None


def exp_panel(ax, x, y, colors=None, sizes=None, n_boot=500, seed=0, scatter=True):
    """scatter + decreasing-exponential fit (solid only if Spearman p < 0.05) with a 95 % bootstrap band over points
    (user 2026-10-06: OLS fits onset vs early accuracy poorly). Returns Spearman, OLS and exponential fit stats
    (R^2, AIC with Gaussian errors; parameters c, a, tau)."""
    from scipy import stats
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[m], np.asarray(y, float)[m]
    if scatter:
        ax.scatter(x, y, s=(np.asarray(sizes)[m] if sizes is not None else 8),
                   c=(np.asarray(colors, object)[m].tolist() if colors is not None else "0.3"), lw=0.3, edgecolors="white", zorder=3)
    n = len(x)
    out = dict(n=n, rho=np.nan, p_rho=np.nan, r2_ols=np.nan, r2_exp=np.nan, aic_ols=np.nan, aic_exp=np.nan,
               exp_c=np.nan, exp_a=np.nan, exp_tau=np.nan)
    if n < 5:
        return out
    rho, prho = stats.spearmanr(x, y)
    tss = np.sum((y - y.mean()) ** 2)
    b = np.polyfit(x, y, 1)
    rss_o = np.sum((y - np.polyval(b, x)) ** 2)
    out.update(rho=rho, p_rho=prho, r2_ols=1 - rss_o / tss, aic_ols=n * np.log(rss_o / n) + 2 * 3)
    p = exp_fit(x, y)
    if p is None:
        return out
    rss_e = np.sum((y - _exp(x, *p)) ** 2)
    out.update(r2_exp=1 - rss_e / tss, aic_exp=n * np.log(rss_e / n) + 2 * 4, exp_c=p[0], exp_a=p[1], exp_tau=p[2])
    xs = np.linspace(x.min(), x.max(), 100)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        pb = exp_fit(x[k], y[k])
        if pb is not None:
            bs.append(_exp(xs, *pb))
    if len(bs) > 20:
        bs = np.array(bs)
        ax.fill_between(xs, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), color="0.6", alpha=0.25, lw=0,
                        edgecolor="none")
    ax.plot(xs, _exp(xs, *p), color="0.2", lw=0.8, ls="-" if prho < 0.05 else "--")
    pad = 0.08 * np.ptp(y)                            # y range from the data: the band may not stretch the axis
    ax.set_ylim(y.min() - pad, y.max() + pad)
    return out


def label_points(ax, xs, ys, texts, colors, fontsize=4.2):
    """labels next to points without overlaps (greedy: the first free candidate offset per point, in display space)"""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    placed = []
    cand = [(3, 1, "left"), (3, -6, "left"), (-3, 1, "right"), (-3, -6, "right"), (3, 7, "left"), (3, -12, "left"),
            (-3, 7, "right"), (-3, -12, "right"), (0, 9, "center"), (0, -14, "center")]
    for i in np.argsort(ys):
        x, y, t, c = xs[i], ys[i], texts[i], colors[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        for k, (dx, dy, ha) in enumerate(cand):
            a = ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=c, ha=ha,
                            va="bottom")
            bb = a.get_window_extent(r).expanded(1.05, 1.1)
            if not any(bb.overlaps(p) for p in placed) or k == len(cand) - 1:
                placed.append(bb)
                break
            a.remove()
