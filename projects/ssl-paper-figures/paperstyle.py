"""Shared figure style for the Bisi & Hamon manuscript: Nature rcParams, square-cell grid on an A4 page, placeholder
panels, exports (PDF, SVG, AI, EPS, PNG, JPG) and the Nature source-data workbook.
Canonical template: skills/ssl-paper-figures/references/paperstyle_template.py."""

import datetime
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

MM = 1 / 25.4
A4_MM = (210.0, 297.0)
RASTER_DPI = 1200  # Nature line art; PNG and JPG

COLORS = {
    "R+": "#00B400", "R-": "#C800C8",
    "whisker": "#f7b519", "auditory": "#2c2cdb", "no_stim": "#262626",
}
TRIAL_TYPE_COLORS = {"whisker_trial": COLORS["whisker"], "auditory_trial": COLORS["auditory"],
                     "no_stim_trial": COLORS["no_stim"]}
TRIAL_TYPE_LABELS = {"whisker_trial": "Whisker", "auditory_trial": "Auditory", "no_stim_trial": "No stimulus"}


def light(color, amount=0.75):
    """Opaque mix of `color` with white (amount 1 = white). Use instead of alpha: EPS has no transparency."""
    r, g, b = mpl.colors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def set_nature_rc():
    """Arial 5-7 pt, thin lines, editable text in SVG, PDF/AI and EPS."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
        "axes.linewidth": 0.5, "lines.linewidth": 0.75,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


class PaperGrid:
    """Unit cells on an A4 page; the figure block is `width_mm` wide at its final print size. Cells are square unless
    `cell_height_mm` is given (to keep a many-row figure within the journal's page height).

    A panel spanning nrows x ncols cells includes the gaps between them, so its axes edges line up with
    single-cell panels. Positions are those of the axes box (not the tight bounding box).
    """

    def __init__(self, width_mm=183, n_cols=5, col_gap_mm=8, row_gap_mm=10, pad_left_mm=8,
                 top_margin_mm=20, page_mm=A4_MM, cell_height_mm=None):
        self.col_gap, self.row_gap, self.pad_left = col_gap_mm, row_gap_mm, pad_left_mm
        self.unit = (width_mm - pad_left_mm - (n_cols - 1) * col_gap_mm) / n_cols
        self.unit_h = cell_height_mm or self.unit
        self.x0 = (page_mm[0] - width_mm) / 2 + pad_left_mm
        self.y0 = top_margin_mm + row_gap_mm  # room for the first row's letters and titles
        self.page = page_mm
        self.fig = plt.figure(figsize=(page_mm[0] * MM, page_mm[1] * MM))
        self.bottom_mm = self.y0

    def axes(self, row, col, nrows=1, ncols=1, letter=None):
        # positions and spans may be fractional (figure planner, quarter cells): a box starts and ends inside cells,
        # never inside a gap, so 1.5 cells from col 0 = one cell, one gap and half a cell
        def start(v, unit, gap):
            return math.floor(v) * (unit + gap) + (v - math.floor(v)) * unit

        def end(v, unit, gap):
            return (math.ceil(v) - 1) * (unit + gap) + (v - math.ceil(v) + 1) * unit

        x = self.x0 + start(col, self.unit, self.col_gap)
        y = self.y0 + start(row, self.unit_h, self.row_gap)
        w = self.x0 + end(col + ncols, self.unit, self.col_gap) - x
        h = self.y0 + end(row + nrows, self.unit_h, self.row_gap) - y
        self.bottom_mm = max(self.bottom_mm, y + h)
        pw, ph = self.page
        ax = self.fig.add_axes([x / pw, 1 - (y + h) / ph, w / pw, h / ph])
        if letter:
            # same row -> same y; letter sits in the left padding, above the axes top
            self.fig.text((x - self.col_gap + 1) / pw, 1 - (y - 4) / ph, letter,
                          fontsize=8, fontweight="bold", va="bottom", ha="left")
        return ax

    def axes_mm(self, x_mm, y_mm, w_mm, h_mm, letter=None, margins_mm=(7.5, 6.0, 3.0, 0.5)):
        """Panel drawn in the figure planner, in mm: x from the left edge of the `width_mm` block, y from the top of
        the figure. The box is the whole panel: its letter sits at the top-left corner and the axes are inset by
        `margins_mm` (left, bottom, top, right) so tick labels and axis labels stay inside the box (2 mm gaps between
        boxes in the planner layout)."""
        left, bottom, top, right = margins_mm
        x = self.x0 - self.pad_left + x_mm
        y = self.y0 - self.row_gap + y_mm  # no letter band above the first row: letters are inside the boxes
        self.bottom_mm = max(self.bottom_mm, y + h_mm)
        pw, ph = self.page
        ax_w, ax_h = w_mm - left - right, h_mm - top - bottom
        ax = self.fig.add_axes([(x + left) / pw, 1 - (y + top + ax_h) / ph, ax_w / pw, ax_h / ph])
        if letter:
            self.fig.text((x - 0.5) / pw, 1 - y / ph, letter, fontsize=8, fontweight="bold", va="top", ha="left")
        return ax

    def placeholder(self, ax, label):
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(True), side.set_linestyle("--"), side.set_edgecolor("0.6")
        ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes, color="0.5", wrap=True)

    def figure_height_mm(self):
        """Height of the figure block (check <= ~170 mm for Nature)."""
        return self.bottom_mm - (self.y0 - self.row_gap)


def save_figure(fig, out_dir, name, raster_dpi=RASTER_DPI):
    """PDF, SVG, EPS (vector, editable text), AI (the PDF under .ai: Illustrator opens it as a PDF-compatible AI file
    with editable text), PNG and JPG at `raster_dpi`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def _save(ext, write):
        """A file open in a viewer on Windows cannot be overwritten on the NAS: write a timestamped copy instead."""
        path = out_dir / f"{name}.{ext}"
        try:
            write(path)
        except OSError as err:
            path = out_dir / f"{name}_{datetime.datetime.now():%H%M%S}.{ext}"
            print(f"warning: {name}.{ext} is locked ({err.strerror}); wrote {path.name}")
            write(path)
        written.append(path)
        return path

    for ext in ("pdf", "svg", "eps"):
        pdf = _save(ext, lambda p: fig.savefig(p))
        if ext == "pdf":
            pdf_path = pdf
    _save("ai", lambda p: shutil.copyfile(pdf_path, p))
    _save("png", lambda p: fig.savefig(p, dpi=raster_dpi))
    _save("jpg", lambda p: fig.savefig(p, dpi=raster_dpi, pil_kwargs={"quality": 95}))
    return written


def write_source_data(data_dir, workbook_path):
    """One sheet per panel CSV (sheet name = CSV stem up to the first '_', e.g. Fig1b)."""
    csvs = sorted(Path(data_dir).glob("*.csv"))
    with pd.ExcelWriter(workbook_path) as writer:
        for csv in csvs:
            pd.read_csv(csv).to_excel(writer, sheet_name=csv.stem.split("_")[0][:31], index=False)
