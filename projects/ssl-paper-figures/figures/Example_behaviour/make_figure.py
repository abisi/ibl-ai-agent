"""Example_behaviour -- assemble the figure from the snapshot's CSVs, export all formats, write the source-data workbook
and the generation note. Panels without a CSV (or of kind 'drawing') are drawn as placeholders.

Run on haas after compute.py (repo root):
    ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Example_behaviour/make_figure.py [--date YYYY-MM-DD]
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
sys.path.insert(0, str(FIG_DIR))
import plot  # noqa: E402
from common import EXTERNAL_REPOS, REPO, environment, load_spec, repo_state, snapshot_dir  # noqa: E402
from paperstyle import PaperGrid, save_figure, set_nature_rc, write_source_data  # noqa: E402

BEHAVIOUR_FILES = ["behavior_analysis_utils.py", "learning_curve_exact.py", "learning_utils.py"]
# the haas copy of ibl-ai-agent is synced, not a git checkout: record the md5 of the figure code
FIGURE_CODE = [str(p.relative_to(REPO)) for p in sorted(FIG_DIR.parents[1].glob("*.py")) + sorted(FIG_DIR.glob("*.py")) + sorted(FIG_DIR.glob("*.yaml"))]


def assemble(spec, out):
    set_nature_rc()
    grid = PaperGrid(**spec["grid"])
    for panel in spec["panels"]:
        cell = {"nrows": 1, "ncols": 1, **panel["cell"]}
        ax = grid.axes(cell["row"], cell["col"], cell["nrows"], cell["ncols"], letter=panel["id"])
        csv = out / "data" / panel["data"] if "data" in panel else None
        if csv is None or not csv.exists():
            grid.placeholder(ax, f"{panel['title']}\n({panel.get('kind', 'not filled')})")
            continue
        getattr(plot, f"plot_{panel['id']}")(ax, pd.read_csv(csv))
    return grid


def generation_note(spec, out, grid, files):
    meta_file = out / "_build" / "panel_meta.json"
    meta = json.load(open(meta_file)) if meta_file.exists() else {}
    lines = [f"# {spec['figure']} -- generation note", "",
             f"- Built: {datetime.datetime.now().isoformat(timespec='minutes')}",
             f"- Figure block: {spec['grid']['width_mm']} x {grid.figure_height_mm():.0f} mm on A4, unit cell "
             f"{grid.unit:.1f} mm",
             f"- ibl-ai-agent: {json.dumps(repo_state(REPO, FIGURE_CODE))}",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    args = parser.parse_args()
    spec = load_spec(FIG_DIR)
    out = snapshot_dir(spec, args.date)
    grid = assemble(spec, out)
    files = save_figure(grid.fig, out, spec["figure"])
    write_source_data(out / "data", out / f"{spec['figure']}_source_data.xlsx")
    generation_note(spec, out, grid, files)
    print(out, f"height {grid.figure_height_mm():.0f} mm")


if __name__ == "__main__":
    main()
