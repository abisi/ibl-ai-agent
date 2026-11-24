"""004 -- Supp. Fig. 3 d-f refit options (user 2026-11-23: "the PCA figure looks kinda squished, refit and visualize
differently?"). Same input as SuppFigure3 compute._flow (held-out even-trial PSTHs, stimulus-aligned conditions, edge
trim 1 bin), PCA on the time bins after three neuron normalisations: raw (current), soft-normalised (rate / (range +
5 Hz), Churchland et al. 2012) and z-scored. Plots PC1-PC2, PC1-PC3 and PC time courses per normalisation.

Run on haas: cd ~/code/unit_spikes_analysis; PYTHONPATH=~/code/NWB_reader:. .venv/bin/python \
  ~/code/ibl-ai-agent/projects/ssl-paper-figures/checks/004_pca_flow_variants.py
Output: <Figures>/_checks/004_pca_flow_variants.png
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures" / "SuppFigure3_rastermap_clustering"
spec = importlib.util.spec_from_file_location("s3c", FIG / "compute.py")
s3 = importlib.util.module_from_spec(spec)
sys.argv = sys.argv[:1]
spec.loader.exec_module(s3)

R = s3.base()["R"]
X = R["X"]
labels_c, colors_c = list(s3.pms.rmu.COND_LABELS), list(s3.pms.rmu.COND_COLORS)
off = np.r_[0, np.cumsum(R["n_bins_list"])]
cols, seg = [], []
for c, lab in enumerate(labels_c):
    if "passive" in lab.lower() or "lick" in lab.lower():
        continue
    lo, hi = s3.FLOW_EDGE_TRIM, int(R["n_bins_list"][c]) - s3.FLOW_EDGE_TRIM
    t = np.asarray(R["t_ctrs"][c])[lo:hi]
    seg.append((lab, colors_c[c], len(cols), len(cols) + len(t), t))
    cols.extend(range(off[c] + lo, off[c] + hi))
Xb = np.nan_to_num(X[:, cols].T)   # bins x neurons
rng_ = Xb.max(0) - Xb.min(0)
SN = Xb / (rng_ + 5.0)
# condition-independent part removed (user 2026-11-23: variants not much better): per time bin, subtract the mean over
# the conditions sharing that window (passive 58 bins, active 98 bins), so PCs capture trial-type differences
CI = SN.copy()
for n_b in sorted({b - a for _, _, a, b, _ in seg}):
    grp = [(a, b) for _, _, a, b, _ in seg if b - a == n_b]
    m = np.mean([SN[a:b] for a, b in grp], 0)
    for a, b in grp:
        CI[a:b] = SN[a:b] - m
NORMS = {"raw": Xb, "soft-norm (range + 5 Hz)": SN, "soft-norm, whitened PC scores": SN,
         "soft-norm, condition-mean removed": CI}
fig, axs = plt.subplots(len(NORMS), 4, figsize=(16, 4 * len(NORMS)))
for r, (name, Z) in enumerate(NORMS.items()):
    p = PCA(n_components=3).fit(Z)
    pc = p.transform(Z)
    if "whitened" in name:   # each PC scaled to unit variance (display only)
        pc = pc / pc.std(0)
    v = p.explained_variance_ratio_
    for lab, col, a, b, t in seg:
        axs[r, 0].plot(pc[a:b, 0], pc[a:b, 1], color=col, lw=0.8, label=lab)
        axs[r, 1].plot(pc[a:b, 0], pc[a:b, 2], color=col, lw=0.8)
        axs[r, 2].plot(t, pc[a:b, 0], color=col, lw=0.8)
        axs[r, 3].plot(t, pc[a:b, 1], color=col, lw=0.8)
    axs[r, 0].set(xlabel=f"PC1 {v[0]:.0%}", ylabel=f"PC2 {v[1]:.0%}", title=name)
    axs[r, 1].set(xlabel="PC1", ylabel=f"PC3 {v[2]:.0%}")
    axs[r, 2].set(xlabel="t (s)", ylabel="PC1")
    axs[r, 3].set(xlabel="t (s)", ylabel="PC2")
    axs[r, 0].legend(fontsize=6)
out = Path("/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures/_checks")
out.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out / "004_pca_flow_variants.png", dpi=110)
print(out / "004_pca_flow_variants.png", [(s[0], s[3] - s[2]) for s in seg])
