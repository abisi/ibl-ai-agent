"""Shared helpers for project report generators (skills/project-report). Import from a projects/<slug>/report/build_report.py:

    sys.path.insert(0, str(REPO / "skills" / "project-report"))
    from report_lib import Report
    R = Report(banned=("arrival",))              # words the user does not want in this report (case-insensitive)
    R.register([("fig-main", "Figure 1"), ("tbl-sizes", "Table S1"), ...])   # every anchor, in reading order, first
    text = f"... grew ({R.ref('fig-main')}) ... {R.num('acc_sw', 0.412)} ... p {R.pv('p_x', 0.0003)} ..."
    md += R.figure("fig-main", src_png, "Full caption ...") + R.table("tbl-sizes", df, "Full caption ...")
    R.write(out_dir, md)                          # checks, copies figures, writes report.md, numbers.json, build.sh

Conventions (user, 2026-10-06; see SKILL.md "Defaults for every report"):
- Anchors are empty Pandoc spans `[]{#id}` placed before the figure / table: they become `\\phantomsection\\label{id}` in
  LaTeX and `id` attributes in HTML; references are `[Figure 1](#id)` -> `\\hyperref[id]{Figure 1}` / `<a href="#id">`.
  (Pandoc figure / table attributes such as `{#fig:x}` need pandoc-crossref and are not used.) Use hyphens in ids.
- Captions carry their own bold labels ("**Figure 1.**"); LaTeX auto-labels are off in build.sh.
- Figures are copied into report/figures/ under the anchor name (<anchor>.png), so names are stable across rebuilds;
  stale copies are removed. Copies use shutil.copyfile (the NAS refuses the metadata copy of shutil.copy2).
- write() refuses to write a report with an anchor that is not defined, an anchor that is never cited, or a banned word.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BUILD_SH = Path(__file__).resolve().parent / "build.sh"


class Report:
    def __init__(self, banned=()):
        self.numbers: dict = {}
        self.figs: dict[str, Path] = {}
        self.anchors: dict[str, str] = {}
        self.banned = tuple(w.lower() for w in banned)

    # ------------------------------------------------------------------------------------------------ numbers
    def num(self, key, value, fmt="{:.3f}"):
        """record a quoted number in numbers.json and return it formatted (never type a number by hand)"""
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            self.numbers[key] = int(value)
            return str(int(value))
        v = float(value)
        self.numbers[key] = None if np.isnan(v) else v
        return "n/a" if np.isnan(v) else fmt.format(v)

    def pv(self, key, value):
        """p-value as '< 0.001' or '= 0.xxx' (recorded)"""
        v = float(value)
        self.numbers[key] = v
        return "< 0.001" if v < 0.001 else f"= {v:.3f}"

    # ------------------------------------------------------------------------------------------------ anchors
    def register(self, pairs):
        """[(anchor, label), ...] in reading order, before any text cites them"""
        for a, label in pairs:
            if a in self.anchors:
                raise ValueError(f"anchor registered twice: {a}")
            self.anchors[a] = label

    def label(self, anchor):
        return self.anchors[anchor]

    def ref(self, anchor):
        """clickable reference, e.g. [Figure 2](#fig-main)"""
        return f"[{self.anchors[anchor]}](#{anchor})"

    # ------------------------------------------------------------------------------------------------ elements
    def figure(self, anchor, src: Path, caption: str) -> str:
        dest = f"{anchor}{Path(src).suffix or '.png'}"
        self.figs[dest] = Path(src)
        return f"\n[]{{#{anchor}}}\n\n![**{self.anchors[anchor]}.** {caption}](figures/{dest}){{width=100%}}\n"

    def table(self, anchor, df: pd.DataFrame, caption: str, footnotesize=True) -> str:
        """pipe table (first column left-aligned, others right) with a bold numbered caption"""
        cols = list(df.columns)
        head = "| " + " | ".join(map(str, cols)) + " |\n|" + "|".join(":--" if i == 0 else "--:" for i in range(len(cols))) + "|\n"
        body = "".join("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |\n"
                       for r in df.itertuples(index=False))
        t = f"{head}{body}\n: **{self.anchors[anchor]}.** {caption}\n"
        if footnotesize:
            t = f"```{{=latex}}\n\\begingroup\\footnotesize\n```\n\n{t}\n```{{=latex}}\n\\endgroup\n```\n"
        return f"\n[]{{#{anchor}}}\n\n{t}"

    # ------------------------------------------------------------------------------------------------ checks + output
    def problems(self, md: str):
        bad = [f"{a}: not defined" for a in self.anchors if f"{{#{a}}}" not in md]
        bad += [f"{a}: never cited" for a in self.anchors if not re.search(rf"\(#{re.escape(a)}\)", md)]
        bad += [f"cites undefined anchor #{a}" for a in set(re.findall(r"\]\(#([\w-]+)\)", md)) if a not in self.anchors]
        bad += [f"contains banned word '{w}'" for w in self.banned if w in md.lower()]
        return bad

    def write(self, out: Path, md: str):
        """check, copy figures (remove stale ones), write report.md, numbers.json and build.sh into out/"""
        bad = self.problems(md)
        if bad:
            raise SystemExit("report not written:\n  " + "\n  ".join(bad))
        out = Path(out)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        for old in (out / "figures").iterdir():
            if old.is_file() and old.name not in self.figs:
                old.unlink()
        for dest, src in self.figs.items():
            shutil.copyfile(src, out / "figures" / dest)
        (out / "report.md").write_text(md, encoding="utf-8")
        (out / "numbers.json").write_text(json.dumps(self.numbers, indent=1), encoding="utf-8")
        shutil.copyfile(BUILD_SH, out / "build.sh")
        n_fig = sum(a.startswith("fig") for a in self.anchors)
        print(f"report.md ({len(md.split())} words), {n_fig} figures, {len(self.anchors) - n_fig} tables, "
              f"{len(self.numbers)} numbers -> {out}")
