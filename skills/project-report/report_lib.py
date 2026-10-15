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

Added 2026-10-15 (user: figure versioning, conference figures as supplementary, claims registry, shared methods, frozen
statistics, executive summary, area spelling):
- SPELLING: wording that is not banned but must be written differently (e.g. "S1 whisker" -> "Ss-whisker") is rewritten
  automatically in write(), and each rewrite is printed. Project-specific additions: Report(spelling={...}).
- figure(..., script=..., kind="main" | "supplementary" | "conference", panels_from=[main anchors]): figures.json records the
  source path, sha1 and modification time of every figure (figure versioning). A "conference" figure (talk / COSYNE
  composite) may only enter as supplementary material, and must name the main figures whose panels it re-assembles.
- claim(cid, text, figure=..., stat=..., status=...): one row of the claims registry (registry.csv + an appendix table via
  registry_table()); status in {holds, fragile, n.s., superseded}.
- methods(*keys): canonical text of skills/project-report/shared_methods.md sections (same definitions in every report).
- snapshot(files, out): copies the statistics files the report reads into out/_snapshots/<date>/ with a manifest, and
  returns the frozen paths (reports read frozen numbers while the review is running).
- write(..., summary=md): also writes summary.md, the one-page executive summary (key claims + one figure).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BUILD_SH = Path(__file__).resolve().parent / "build.sh"
SHARED_METHODS = Path(__file__).resolve().parent / "shared_methods.md"
# area-group spelling (user, 2026-10-13 / 10-15): somatosensory groups are "Ss-<part>", never "S1 <part>" / "SS-<part>"
SPELLING = {r"\bS1[ -]?(whisker|orofacial|body)\b": r"Ss-\1", r"\bSS-(whisker|orofacial|body)\b": r"Ss-\1",
            r"\bwS1\b(?= (?:area )?group)": "Ss-whisker"}
STATUS = ("holds", "fragile", "n.s.", "superseded")


class Report:
    def __init__(self, banned=(), spelling=None):
        self.numbers: dict = {}
        self.figs: dict[str, Path] = {}
        self.fig_meta: dict[str, dict] = {}
        self.claims: list[dict] = []
        self.anchors: dict[str, str] = {}
        self.banned = tuple(w.lower() for w in banned)
        self.spelling = {**SPELLING, **(spelling or {})}

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
    def figure(self, anchor, src: Path, caption: str, script: str = "", kind: str = "main", panels_from=()) -> str:
        """kind: main / supplementary / conference (conference = talk or abstract composite: supplementary only, and
        panels_from must list the main-figure anchors whose panels it re-assembles)"""
        if kind not in ("main", "supplementary", "conference"):
            raise ValueError(f"{anchor}: unknown figure kind {kind}")
        if kind == "conference" and not panels_from:
            raise ValueError(f"{anchor}: a conference figure must name the main figures its panels come from (panels_from)")
        dest = f"{anchor}{Path(src).suffix or '.png'}"
        src = Path(src)
        self.figs[dest] = src
        st = src.stat() if src.exists() else None
        self.fig_meta[anchor] = dict(label=self.anchors[anchor], kind=kind, source=str(src), script=script,
                                     panels_from=list(panels_from), sha1=hashlib.sha1(src.read_bytes()).hexdigest()[:12] if st else None,
                                     modified=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="minutes") if st else None)
        return f"\n[]{{#{anchor}}}\n\n![**{self.anchors[anchor]}.** {caption}](figures/{dest}){{width=100%}}\n"

    # ------------------------------------------------------------------------------------------------ registry / shared text
    def claim(self, cid, text, figure="", stat="", n="", status="holds", note=""):
        """record one claim of the report (registry.csv); returns the text so it can be placed in the prose"""
        if status not in STATUS:
            raise ValueError(f"claim {cid}: status must be one of {STATUS}")
        self.claims.append(dict(id=cid, claim=text, figure=figure, stat=stat, n=n, status=status, note=note))
        return text

    def registry_table(self, anchor, caption="Claims registry: every claim of the report, where it is shown, the statistic "
                       "and its status (holds / fragile / n.s. / superseded)."):
        df = pd.DataFrame(self.claims, columns=["id", "claim", "figure", "stat", "n", "status", "note"])
        return self.table(anchor, df, caption)

    @staticmethod
    def methods(*keys) -> str:
        """sections of shared_methods.md (## <key> headers), verbatim, so every report uses the same definitions"""
        txt = SHARED_METHODS.read_text(encoding="utf-8")
        parts = dict(re.findall(r"^## (\S+)\n(.*?)(?=^## |\Z)", txt, re.S | re.M))
        missing = [k for k in keys if k not in parts]
        if missing:
            raise KeyError(f"shared_methods.md has no section {missing}")
        return "\n\n".join(parts[k].strip() for k in keys) + "\n\n"

    @staticmethod
    def snapshot(files, out: Path, tag=None):
        """freeze the statistics files read by the report: copies into out/_snapshots/<tag>/ + manifest; returns new paths"""
        tag = tag or datetime.date.today().isoformat()
        d = Path(out) / "_snapshots" / tag
        d.mkdir(parents=True, exist_ok=True)
        new, man = [], []
        for f in map(Path, files):
            dst = d / f.name
            if not dst.exists():
                shutil.copyfile(f, dst)
            new.append(dst); man.append(dict(file=f.name, source=str(f), sha1=hashlib.sha1(dst.read_bytes()).hexdigest()[:12]))
        (d / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        return new

    def respell(self, md: str) -> str:
        n = 0
        for pat, rep in self.spelling.items():
            md, k = re.subn(pat, rep, md)
            n += k
        if n:
            print(f"respelled {n} area name(s) (SPELLING)")
        return md

    def eq(self, anchor, latex: str, indent: str = "") -> str:
        """numbered display equation (register the anchor with a label "Eq. (n)"): centred on its own line, number on the
        right (\\tag in LaTeX); cite it with ref(anchor). indent: two spaces when the equation sits inside a list item"""
        n = self.anchors[anchor].split("(")[-1].rstrip(")")
        return f"\n{indent}[]{{#{anchor}}}\n\n{indent}$$ {latex} \\tag{{{n}}} $$\n\n"

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
        bad += [f"unnumbered display equation: {m[:60]}" for m in re.findall(r"\$\$(.+?)\$\$", md, re.S) if "\\tag{" not in m]
        bad += [f"contains banned word '{w}'" for w in self.banned if w in md.lower()]
        return bad

    def write(self, out: Path, md: str, summary: str | None = None):
        """check, copy figures (remove stale ones), write report.md, numbers.json, figures.json, registry.csv, summary.md
        and build.sh into out/"""
        md = self.respell(md)
        summary = self.respell(summary) if summary else summary
        bad = self.problems(md)
        for a, m in self.fig_meta.items():
            bad += [f"{a}: conference figure cites unknown main figure {p}" for p in m["panels_from"]
                    if p not in self.fig_meta or self.fig_meta[p]["kind"] != "main"]
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
        # HTML source: Pandoc's MathML drops the equation tag, so the number is written next to the equation instead
        (out / "report_html.md").write_text(re.sub(r"\\tag\{([^}]+)\}", lambda m: r"\qquad\qquad (" + m.group(1) + ")", md), encoding="utf-8")
        (out / "numbers.json").write_text(json.dumps(self.numbers, indent=1), encoding="utf-8")
        (out / "figures.json").write_text(json.dumps(self.fig_meta, indent=1), encoding="utf-8")
        if self.claims:
            pd.DataFrame(self.claims).to_csv(out / "registry.csv", index=False)
        if summary:
            (out / "summary.md").write_text(summary, encoding="utf-8")
        shutil.copyfile(BUILD_SH, out / "build.sh")
        k = lambda p: sum(a.startswith(p) for a in self.anchors)
        print(f"report.md ({len(md.split())} words), {k('fig')} figures, {k('tbl')} tables, {k('eq')} equations, "
              f"{len(self.numbers)} numbers -> {out}")
