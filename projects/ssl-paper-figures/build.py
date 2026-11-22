"""Assemble one figure from its snapshot's CSVs, export all formats, write the source-data workbook, the generation note
and the caption draft. Panels without a CSV (or of kind 'drawing') are drawn as placeholders, so the layout renders
before any panel is filled.

Run on haas after the figure's compute.py (repo root):
    ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/build.py figures/Figure1_behaviour [--date YYYY-MM-DD]
"""

import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import EXTERNAL_REPOS, FIGURES_ROOT, REPO, environment, load_spec, repo_state, snapshot_dir, write_caption  # noqa: E402
from paperstyle import PaperGrid, save_figure, set_nature_rc, write_source_data  # noqa: E402

BEHAVIOUR_FILES = ["behavior_analysis_utils.py", "learning_utils.py"]


def figure_code(fig_dir):
    """The haas copy of ibl-ai-agent is synced, not a git checkout: record the md5 of the figure code."""
    files = sorted(HERE.glob("*.py")) + sorted(fig_dir.glob("*.py")) + sorted(fig_dir.glob("*.yaml"))
    return [str(p.relative_to(REPO)) for p in files]


def load_plot_module(fig_dir):
    if not (fig_dir / "plot.py").exists():
        return None
    spec = importlib.util.spec_from_file_location(f"plot_{fig_dir.name}", fig_dir / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PDFIUM_PATH = Path.home() / ".local" / "pylib_pdfium"  # pypdfium2 installed with uv --target (not in the venv)


def render_pdf(src, dst, dpi=1200):
    """First page of a PDF drawing (e.g. Figures/FromFigma) to PNG, when the PDF is newer than the PNG."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    sys.path.insert(0, str(PDFIUM_PATH))
    import pypdfium2 as pdfium
    pdfium.PdfDocument(str(src))[0].render(scale=dpi / 72).to_pil().save(dst)
    print(f"rendered {src.name} -> {dst.name} ({dpi} dpi)")


def place_image(ax, path, source=None):
    """Raster drawing in the panel box, aspect kept, anchored at the top left; re-rendered from `source` (PDF) when that
    is newer."""
    import matplotlib.image as mpimg
    if source is not None:
        render_pdf(source, path)
    ax.imshow(mpimg.imread(path), interpolation="lanczos")
    ax.set_anchor("NW")
    ax.axis("off")


def assemble(spec, out, plot):
    set_nature_rc()
    grid = PaperGrid(**spec["grid"])
    for panel in spec["panels"]:
        cell = {"nrows": 1, "ncols": 1, **panel["cell"]}
        if "x_mm" in cell:  # box drawn in the figure planner
            extra = {"margins_mm": tuple(cell["margins_mm"])} if "margins_mm" in cell else {}  # e.g. drawings: no axes
            ax = grid.axes_mm(cell["x_mm"], cell["y_mm"], cell["w_mm"], cell["h_mm"], letter=panel["id"], **extra)
        else:
            ax = grid.axes(cell["row"], cell["col"], cell["nrows"], cell["ncols"], letter=panel["id"])
        if "image" in panel:  # drawing exported elsewhere (e.g. Figma), path relative to FIGURES_ROOT
            place_image(ax, FIGURES_ROOT / panel["image"],
                        FIGURES_ROOT / panel["source"] if str(panel.get("source", "")).endswith(".pdf") else None)
            continue
        csv = out / "data" / panel["data"] if "data" in panel else None
        fn = f"plot_{panel.get('key', panel['id'])}"  # panels are drawn by their stable key; letters follow the layout
        if csv is None or not csv.exists() or plot is None or not hasattr(plot, fn):
            grid.placeholder(ax, f"{panel['title']}\n({panel.get('kind', 'not filled')})")
            continue
        getattr(plot, fn)(ax, pd.read_csv(csv))
        panel_heading(grid, panel)
    figure_legend(grid, spec)
    figure_key(grid, spec)
    return grid


def panel_heading(grid, panel):
    """spec panel 'heading': what the panel shows, one or two lines at the top of the planner box, right of the letter
    (user 2026-11-20: explain each panel on the figure itself). Leave room for it in the cell's top margin."""
    if not panel.get("heading") or "x_mm" not in panel["cell"]:
        return
    pw, ph = grid.page
    x = grid.x0 - grid.pad_left + panel["cell"]["x_mm"] + 3.2
    y = grid.y0 - grid.row_gap + panel["cell"]["y_mm"] + 0.15
    grid.fig.text(x / pw, 1 - y / ph, panel["heading"], fontsize=6, va="top", ha="left", linespacing=1.05)  # not bold (user 2026-11-20)


def figure_key(grid, spec):
    """spec 'figure_key': list of lines (abbreviations, area names) printed under the last row of panels, full width."""
    lines = spec.get("figure_key")
    if not lines:
        return
    import textwrap
    pw, ph = grid.page
    x = grid.x0 - grid.pad_left
    width_chars = int(spec["grid"]["width_mm"] / 1.05)  # 5 pt Arial ~ 1 mm per character
    text = "\n".join(textwrap.fill(line, width_chars) for line in lines)
    y = grid.bottom_mm + 4.0
    t = grid.fig.text(x / pw, 1 - y / ph, text, fontsize=5, va="top", ha="left", linespacing=1.25)
    grid.bottom_mm = y + 2.3 * (text.count("\n") + 1)
    return t


def figure_legend(grid, spec):
    """Colour key at the top right of the figure block (user 2026-11-16: one R+ / R- key for the whole figure instead
    of 'Bars: R+ left, R- right' under panels). spec: figure_legend: [{label, color}, ...]."""
    entries = spec.get("figure_legend")
    if not entries:
        return
    import matplotlib as mpl
    pw, ph = grid.page
    right = grid.x0 - grid.pad_left + spec["grid"]["width_mm"]
    top = grid.y0 - grid.row_gap - 6.0  # above the first row of panels and their titles
    handles = [mpl.patches.Patch(color=e["color"], label=e["label"]) for e in entries]
    grid.fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(right / pw, 1 - top / ph), ncol=len(handles),
                    frameon=False, fontsize=6, handlelength=1.2, handleheight=0.8, handletextpad=0.4, columnspacing=1.0,
                    borderaxespad=0, borderpad=0)


def generation_note(spec, out, grid, files, fig_dir):
    meta_file = out / "_build" / "panel_meta.json"
    meta = json.load(open(meta_file)) if meta_file.exists() else {}
    lines = [f"# {spec['figure']} -- generation note", "",
             f"- Built: {datetime.datetime.now().isoformat(timespec='minutes')}",
             f"- Figure block: {spec['grid']['width_mm']} x {grid.figure_height_mm():.0f} mm on A4, unit cell "
             f"{grid.unit:.1f} x {grid.unit_h:.1f} mm",
             f"- ibl-ai-agent: {json.dumps(repo_state(REPO, figure_code(fig_dir)))}",
             f"- behaviour_analysis: {json.dumps(repo_state(EXTERNAL_REPOS['behaviour_analysis'], BEHAVIOUR_FILES))}",
             f"- NWB_reader: {json.dumps(repo_state(EXTERNAL_REPOS['NWB_reader'], ['NWB_reader_functions.py']))}",
             f"- Environment: {json.dumps(environment())}",
             f"- Files: {', '.join(f.name for f in files)}", "", "## Shared inputs", ""]
    for name, s in spec["shared"].items():
        lines.append(f"- **{name}**: `{s['code']}`; calls {s.get('calls', '-')}; params {s.get('params', {})}")
    if "_shared" in meta:
        lines.append(f"- run: {meta['_shared']}")
    lines += ["", "## Panels", ""]
    for panel in spec["panels"]:
        lines.append(f"### {panel['id']}. {panel['title']}")
        for key in ("kind", "calls", "params", "stats", "data", "status"):
            if key in panel:
                lines.append(f"- {key}: {panel[key]}")
        if panel["id"] in meta:
            lines.append(f"- run: {json.dumps(meta[panel['id']])}")
        csv = out / "data" / panel.get("data", "-")
        if csv.exists():
            lines.append(f"- columns: {', '.join(pd.read_csv(csv, nrows=0).columns)}")
        lines.append("")
    (out / f"{spec['figure']}.md").write_text("\n".join(lines))


def build(fig_dir, date=None):
    fig_dir = Path(fig_dir).resolve()
    spec = load_spec(fig_dir)
    out = snapshot_dir(spec, date)
    grid = assemble(spec, out, load_plot_module(fig_dir))
    files = save_figure(grid.fig, out, spec["figure"])
    if any((out / "data").glob("*.csv")):
        write_source_data(out / "data", out / f"{spec['figure']}_source_data.xlsx")
    generation_note(spec, out, grid, files, fig_dir)
    meta_file = out / "_build" / "panel_meta.json"
    write_caption(spec, json.load(open(meta_file)) if meta_file.exists() else {}, out / f"{spec['figure']}_caption.md")
    print(out, f"height {grid.figure_height_mm():.0f} mm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("figure_dir", help="e.g. figures/Figure1_behaviour (relative to this folder or absolute)")
    parser.add_argument("--date")
    args = parser.parse_args()
    fig_dir = Path(args.figure_dir)
    build(fig_dir if fig_dir.is_absolute() or fig_dir.exists() else HERE / fig_dir, args.date)


if __name__ == "__main__":
    main()
