"""Template for projects/ssl-paper-figures/paperstyle.py: Nature rcParams, square-cell grid on an A4 page,
placeholder panels, exports and the source-data workbook."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

MM = 1 / 25.4
A4_MM = (210.0, 297.0)


def set_nature_rc():
    """Arial 5-7 pt, thin lines, editable text in SVG and PDF."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
        "axes.linewidth": 0.5, "lines.linewidth": 0.75,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


class PaperGrid:
    """Square unit cells on an A4 page; the figure block is `width_mm` wide at its final print size.

    A panel spanning nrows x ncols cells includes the gaps between them, so its axes edges line up with
    single-cell panels. Positions are those of the axes box (not the tight bounding box).
    """

    def __init__(self, width_mm=183, n_cols=5, col_gap_mm=8, row_gap_mm=10, pad_left_mm=8,
                 top_margin_mm=20, page_mm=A4_MM):
        self.col_gap, self.row_gap = col_gap_mm, row_gap_mm
        self.unit = (width_mm - pad_left_mm - (n_cols - 1) * col_gap_mm) / n_cols
        self.x0 = (page_mm[0] - width_mm) / 2 + pad_left_mm
        self.y0 = top_margin_mm + row_gap_mm  # room for the first row's letters and titles
        self.page = page_mm
        self.fig = plt.figure(figsize=(page_mm[0] * MM, page_mm[1] * MM))
        self.bottom_mm = self.y0

    def axes(self, row, col, nrows=1, ncols=1, letter=None):
        x = self.x0 + col * (self.unit + self.col_gap)
        y = self.y0 + row * (self.unit + self.row_gap)
        w = ncols * self.unit + (ncols - 1) * self.col_gap
        h = nrows * self.unit + (nrows - 1) * self.row_gap
        self.bottom_mm = max(self.bottom_mm, y + h)
        pw, ph = self.page
        ax = self.fig.add_axes([x / pw, 1 - (y + h) / ph, w / pw, h / ph])
        if letter:
            # same row -> same y; letter sits in the left padding, above the axes top
            self.fig.text((x - self.col_gap + 1) / pw, 1 - (y - 4) / ph, letter,
                          fontsize=8, fontweight="bold", va="bottom", ha="left")
        return ax

    def placeholder(self, ax, label):
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(True), side.set_linestyle("--"), side.set_edgecolor("0.6")
        ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes, color="0.5", wrap=True)

    def figure_height_mm(self):
        """Height of the figure block (check <= ~170 mm for Nature)."""
        return self.bottom_mm - (self.y0 - self.row_gap)


def save_figure(fig, out_dir, name, dpi=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi)


def write_source_data(data_dir, workbook_path):
    """One sheet per panel CSV (sheet name = CSV stem up to the first '_', e.g. Fig1b)."""
    csvs = sorted(Path(data_dir).glob("*.csv"))
    with pd.ExcelWriter(workbook_path) as writer:
        for csv in csvs:
            pd.read_csv(csv).to_excel(writer, sheet_name=csv.stem.split("_")[0][:31], index=False)
