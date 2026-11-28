"""LaTeX build of a snapshot's text files (user 2026-11-28: the caption and methods .md and .pdf are generated from LaTeX,
plus one PDF with the figure, its caption and its methods, cross-referenced with hyperref + cleveref).

Sources (markdown, fields already filled): the spec captions (per panel, `common.fill_caption`) and the filled methods
template. They are converted to LaTeX fragments (Pandoc), and three documents are compiled with XeLaTeX (latexmk):
  <F>_full.pdf     the figure (cropped to its block, one invisible anchor per panel letter), caption, methods
  <F>_caption.pdf  caption only; <F>_methods.pdf methods only (labels of the other parts via xr-hyper: links open
                   <F>_full.pdf)
The .md files are written from the same LaTeX, with every reference resolved from the full document's .aux.

Cross-references, written in the templates (raw LaTeX passes through Pandoc):
  - panels: "panel B", "panels B, D-J", "Panel F:" are linked automatically (\\cref{pnl:b}); \\cref{pnl:b} also works.
  - methods paragraphs: a line starting with **B** (or **D, E**) is panel B's paragraph (label meth:b); the caption's
    panel letters link to it.
  - figure: \\cref{fig:main}; methods section: \\hyperref[sec:methods]{Methods} ("details in Methods" is linked).
  - equations: \\begin{equation} ... \\label{eq:name} \\end{equation}, cited \\cref{eq:name};
    tables: Pandoc pipe table with caption "Table: text \\label{tab:name}", cited \\cref{tab:name}.
Toolchain: Pandoc (or Quarto's bundled one) and TinyTeX (`quarto install tinytex`); not on haas -> build locally.
Standalone (rebuild the text files of an existing snapshot):
    python latex_docs.py figures/<Figure> <snapshot_dir>
"""

import json
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APPDATA = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
SEP = "ZZSPLITPARAGRAPHZZ"
SUPER = {"⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
P_EXP = re.compile(r"(\d+(?:\.\d+)?) × 10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
RANGE = r"[A-Z](?:\s*[-–]\s*[A-Z])?"
PANEL_REF = re.compile(rf"\b([Pp])anels?\s+({RANGE}(?:\s*(?:,\s*and|,|and)\s*{RANGE})*)(?![A-Za-z0-9])")
METHOD_HEAD = re.compile(r"^\*\*([A-Z](?:\s*(?:[-–]|,|and)\s*[A-Z])*)\*\*", re.M)
CREF = re.compile(r"\\([cC])ref(range)?\{([^}]*)\}(?:\{([^}]*)\})?")
NAMES = {"panel": ("panel", "panels"), "mpanel": ("Methods, panel", "Methods, panels"), "figure": ("Fig.", "Figs."),
         "equation": ("Eq.", "Eqs."), "table": ("Table", "Tables"), "section": ("section", "sections")}


def _which(name, *fallbacks):
    found = shutil.which(name)
    if found:
        return found
    return next((str(p) for p in fallbacks if Path(p).exists()), None)


def pandoc():
    return _which("pandoc", Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Quarto/bin/tools/pandoc.exe")


def latexmk():
    return _which("latexmk", APPDATA / "TinyTeX/bin/windows/latexmk.exe", Path.home() / ".TinyTeX/bin/x86_64-linux/latexmk")


def available():
    return bool(pandoc() and latexmk())


# ---------------------------------------------------------------- markdown -> LaTeX fragments

def expand(group):
    """'B, D-J' / 'D and E' -> ['b', 'd', 'e', ..., 'j']."""
    out = []
    for part in re.split(r"\s*(?:,\s*and|,|and)\s*", group.strip()):
        m = re.fullmatch(r"([A-Z])\s*[-–]\s*([A-Z])", part)
        out += [chr(c).lower() for c in range(ord(m.group(1)), ord(m.group(2)) + 1)] if m else [part.lower()]
    return out


def prep(md, letters):
    """Markdown -> markdown with raw LaTeX: P exponents as math, panel references as \\cref, methods panel heads anchored."""
    def p_exp(m):  # text-mode exponent (Arial), not math: matches the rest of the caption
        exp = "".join(SUPER[c] for c in m.group(2)).replace("-", "−")
        return f"{m.group(1)} × 10\\textsuperscript{{{exp}}}"

    md = P_EXP.sub(p_exp, md)

    def head(m):
        keys = expand(m.group(1))
        if not all(k in letters for k in keys):
            return m.group(0)
        return "".join(f"\\methpanel{{{letters[k]}}}{{{k}}}" for k in keys) + f"\\textbf{{{m.group(1)}}}"

    md = METHOD_HEAD.sub(head, md)

    def ref(m):
        keys = expand(m.group(2))
        if not all(k in letters for k in keys):
            return m.group(0)
        return f"\\{'C' if m.group(1) == 'P' else 'c'}ref{{{','.join('pnl:' + k for k in keys)}}}"

    md = PANEL_REF.sub(ref, md)
    # caption shorthand "As f," / "as K;": a single panel letter followed by punctuation
    md = re.sub(r"\b([Aa]s) ([A-Za-z])(?=[,;:.)])",
                lambda m: f"{m.group(1)} \\cref{{pnl:{m.group(2).lower()}}}" if m.group(2).lower() in letters
                else m.group(0), md)
    return re.sub(r"\b(details in|see|in the) Methods\b", lambda m: f"{m.group(1)} \\hyperref[sec:methods]{{Methods}}", md)


def to_latex(chunks):
    """Markdown chunks -> LaTeX fragments in one Pandoc call."""
    src = f"\n\n{SEP}\n\n".join(chunks)
    out = subprocess.run([pandoc(), "-f", "markdown+raw_tex-auto_identifiers-smart", "-t", "latex", "--wrap=preserve"],
                         input=src.encode("utf-8"), capture_output=True, check=True).stdout.decode("utf-8")
    parts = [p.strip() for p in out.split(SEP)]
    assert len(parts) == len(chunks), (len(parts), len(chunks))
    return parts


# ---------------------------------------------------------------- documents

PREAMBLE = r"""\documentclass[9pt]{extarticle}
\usepackage[a4paper,margin=15mm]{geometry}
\usepackage{amsmath}
\usepackage{fontspec}
\usepackage{unicode-math}
\setmainfont{Arial}
\setmathfont{Cambria Math}
\usepackage{xcolor,graphicx,booktabs,longtable,array,calc}
\usepackage[font=small]{caption}
\setcounter{secnumdepth}{0}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\usepackage{xr-hyper}
%EXTERNAL%
\usepackage[colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!60!black,filecolor=blue!60!black]{hyperref}
\usepackage[nameinlink,noabbrev]{cleveref}
\newcounter{panel}\renewcommand{\thepanel}{\Alph{panel}}
\newcounter{mpanel}\renewcommand{\thempanel}{\Alph{mpanel}}
\crefname{panel}{panel}{panels}\Crefname{panel}{Panel}{Panels}
\crefname{mpanel}{Methods, panel}{Methods, panels}\Crefname{mpanel}{Methods, panel}{Methods, panels}
\crefname{figure}{Fig.}{Figs.}\Crefname{figure}{Fig.}{Figs.}
\crefname{equation}{Eq.}{Eqs.}\Crefname{equation}{Eq.}{Eqs.}
\crefname{table}{Table}{Tables}\Crefname{table}{Table}{Tables}
\newcommand{\crefrangeconjunction}{--}
\newcommand{\panelanchor}[2]{\setcounter{panel}{#1}\addtocounter{panel}{-1}\refstepcounter{panel}\label{pnl:#2}}
\newcommand{\methpanel}[2]{\setcounter{mpanel}{#1}\addtocounter{mpanel}{-1}\refstepcounter{mpanel}\label{meth:#2}}
\renewcommand{\thefigure}{%NUMBER%}
\begin{document}
"""


def figure_block(spec, layout, pdf_name):
    """The figure PDF cropped to its block, scaled into the text area, with one anchor per panel letter."""
    pw, ph = layout["page_mm"]
    x0, y0, w, h = layout["block_mm"]
    scale = min(180.0 / w, 255.0 / h, 1.0)
    trim = f"{x0}mm {ph - y0 - h}mm {pw - x0 - w}mm {y0}mm"
    puts = [f"\\put({x - x0:.1f},{h - (y - y0):.1f}){{\\panelanchor{{{layout['index'][k]}}}{{{k}}}}}"
            for k, (x, y) in layout["letters"].items()]
    return "\n".join([
        "\\begin{figure}[p]\\centering",
        f"\\setlength{{\\unitlength}}{{{scale:.4f}mm}}",
        f"\\begin{{picture}}({w:.1f},{h:.1f})",
        f"\\put(0,0){{\\includegraphics[trim={trim},clip,width={w * scale:.2f}mm]{{{pdf_name}}}}}",
        *puts, "\\end{picture}",
        f"\\refstepcounter{{figure}}\\label{{fig:main}}",
        f"\\caption*{{\\textbf{{Fig. {spec['number']}}} (caption on the next page; panels are linked from the text).}}",
        "\\end{figure}"])


def caption_body(spec, title, panels, footer, has_methods):
    parts = [f"\\section{{Figure legend}}\\label{{sec:caption}}", f"\\textbf{{Fig. {spec['number']} | {title}}}"]
    for key, text in panels:
        letter = f"\\textbf{{{key.upper()}}}"
        if key in has_methods:
            letter = f"\\hyperref[meth:{key}]{{{letter}}}"
        parts.append(f"{letter}, {text}")
    if footer:
        parts.append(footer)
    return parts[0] + "\n\n" + " ".join(parts[1:]) + "\n"


def document(spec, body, external=None):
    pre = PREAMBLE.replace("%NUMBER%", str(spec["number"]))
    pre = pre.replace("%EXTERNAL%", f"\\externaldocument{{{external}}}[{external}.pdf]" if external else "")
    return pre + body + "\n\\end{document}\n"


def compile_tex(tex_path):
    proc = subprocess.run([latexmk(), "-xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                          cwd=tex_path.parent, capture_output=True)
    log = tex_path.with_suffix(".log")
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else proc.stdout.decode(errors="replace")
    missing = sorted(set(re.findall(r"Missing character: There is no (.) ", text)))
    undefined = sorted(set(re.findall(r"Reference `([^']+)' on page", text)))
    if proc.returncode != 0 or not tex_path.with_suffix(".pdf").exists():
        tail = "\n".join(text.splitlines()[-30:])
        raise RuntimeError(f"LaTeX failed for {tex_path.name}:\n{tail}")
    if missing:
        print(f"{tex_path.name}: glyphs missing in the font: {' '.join(missing)}")
    if undefined:
        print(f"{tex_path.name}: undefined references: {', '.join(undefined)}")
    return tex_path.with_suffix(".pdf")


# ---------------------------------------------------------------- LaTeX -> markdown

def cref_table(aux):
    """label -> (type, value) from cleveref's @cref entries."""
    table = {}
    for m in re.finditer(r"\\newlabel\{(.+?)@cref\}\{\{\[([^\]]*)\]\[[^\]]*\]\[[^\]]*\]([^}]*)\}", aux):
        table[m.group(1)] = (m.group(2), m.group(3))
    return table


def plain_ref(labels, table, capital, is_range=False):
    found = [table[k] for k in labels if k in table]
    if not found:
        return "??"
    kind = found[0][0]
    one, many = NAMES.get(kind, (kind, kind + "s"))
    vals = [v for _, v in found]
    if is_range:
        text = f"{vals[0]}–{vals[-1]}"
    elif kind in ("panel", "mpanel") and len(vals) > 2:
        runs, run = [], [vals[0]]
        for v in vals[1:]:
            if len(v) == 1 and len(run[-1]) == 1 and ord(v) == ord(run[-1]) + 1:
                run.append(v)
            else:
                runs.append(run), (run := [v])
        runs.append(run)
        items = [f"{r[0]}–{r[-1]}" if len(r) > 2 else ", ".join(r) for r in runs]
        text = ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1]
    else:
        text = ", ".join(vals[:-1]) + (" and " if len(vals) > 1 else "") + vals[-1]
    name = many if (len(vals) > 1 or is_range) else one
    return f"{name[0].upper() + name[1:] if capital else name} {text}"


def latex_to_md(fragment, table):
    def ref(m):
        if m.group(2):
            return plain_ref([m.group(3), m.group(4)], table, m.group(1) == "C", True)
        return plain_ref([k.strip() for k in m.group(3).split(",")], table, m.group(1) == "C")

    text = CREF.sub(ref, fragment)
    text = re.sub(r"\\(methpanel|panelanchor)\{[^}]*\}\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\hyperref\[[^\]]*\]\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", text)
    return subprocess.run([pandoc(), "-f", "latex", "-t", "markdown-smart-raw_tex-raw_attribute", "--wrap=none"],
                          input=text.encode("utf-8"), capture_output=True, check=True).stdout.decode("utf-8")


# ---------------------------------------------------------------- entry point

def build_docs(spec, out, meta, layout, methods_md=None, figure_pdf=None):
    """Write <F>_full.pdf, <F>_caption.pdf/.md and <F>_methods.pdf/.md into the snapshot `out`; the .tex sources go to
    out/_build/latex. methods_md: the filled methods markdown (None: no methods template)."""
    from common import fill_caption
    name, out = spec["figure"], Path(out)
    letters = {p["id"]: string.ascii_lowercase.index(p["id"]) + 1 for p in spec["panels"]}
    layout = dict(layout, index=letters, letters={k: v for k, v in layout["letters"].items() if k in letters})
    caption = spec.get("caption", {})
    shared = meta.get("_shared", {})
    chunks = [fill_caption(caption.get("title", name), shared)]
    for p in spec["panels"]:
        chunks.append(fill_caption(p.get("caption") or f"[caption missing: {p['title']}]",
                                   {**shared, **meta.get(p["id"], {})}))
    chunks.append(fill_caption(caption.get("footer", ""), shared) or "-")
    methods_chunk = None
    if methods_md:
        methods_chunk = re.sub(r"\A#\s[^\n]*\n", "", methods_md.lstrip())
        chunks.append(methods_chunk)
    tex = to_latex([prep(c, letters) for c in chunks])
    title, panel_tex, footer = tex[0], tex[1:1 + len(spec["panels"])], tex[1 + len(spec["panels"])]
    footer = "" if footer.strip() == "-" else footer
    methods_tex = tex[-1] if methods_md else None
    has_methods = set(re.findall(r"\\methpanel\{\d+\}\{(\w)\}", methods_tex or ""))
    cap_body = caption_body(spec, title, [(p["id"], t) for p, t in zip(spec["panels"], panel_tex)], footer, has_methods)
    meth_body = f"\\section{{Methods}}\\label{{sec:methods}}\n\n{methods_tex}\n" if methods_tex else ""

    src_dir = out / "_build" / "latex"
    src_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copyfile(figure_pdf or out / f"{name}.pdf", tmp / "figure.pdf")
        docs = {f"{name}_full": document(spec, figure_block(spec, layout, "figure.pdf") + "\n\\clearpage\n" + cap_body
                                         + ("\n\\clearpage\n" + meth_body if meth_body else ""))}
        # caption / methods alone: their own labels local, the others from the full document (links open it)
        cap_alone = re.sub(r"\\hyperref\[meth:(\w)\]", r"\\hyperref[meth:\1]", cap_body)
        docs[f"{name}_caption"] = document(spec, cap_alone, external=f"{name}_full")
        if meth_body:
            docs[f"{name}_methods"] = document(spec, meth_body, external=f"{name}_full")
        pdfs = {}
        for stem, text in docs.items():
            (tmp / f"{stem}.tex").write_text(text, encoding="utf-8")
            (src_dir / f"{stem}.tex").write_text(text.replace("{figure.pdf}", f"{{../../{name}.pdf}}"), encoding="utf-8")
            pdfs[stem] = compile_tex(tmp / f"{stem}.tex")
        table = cref_table((tmp / f"{name}_full.aux").read_text(encoding="utf-8", errors="replace"))
        written = []
        for stem, pdf in pdfs.items():
            dst = out / f"{stem}.pdf"
            try:
                shutil.copyfile(pdf, dst)
            except OSError:  # open in a viewer on Windows
                dst = out / f"{stem}_new.pdf"
                shutil.copyfile(pdf, dst)
                print(f"warning: {stem}.pdf is locked; wrote {dst.name}")
            written.append(dst)
    (out / f"{name}_caption.md").write_text(latex_to_md(cap_body, table), encoding="utf-8")
    written.append(out / f"{name}_caption.md")
    if meth_body:
        (out / f"{name}_methods.md").write_text(latex_to_md(meth_body, table), encoding="utf-8")
        written.append(out / f"{name}_methods.md")
    return written


def layout_from_spec(spec):
    """Panel letter positions and figure block from a placeholder-only assembly (no data drawn)."""
    import matplotlib.pyplot as plt
    from build import assemble
    grid = assemble(spec, Path(tempfile.gettempdir()), None)
    plt.close(grid.fig)
    return figure_layout(grid, spec)


def figure_layout(grid, spec):
    # full page width (labels may run past the width_mm block); from just above the figure-level legend to the last row
    y = max(0.0, grid.y0 - grid.row_gap - 12.0)
    return {"page_mm": list(grid.page), "block_mm": [0.0, y, grid.page[0], min(grid.bottom_mm + 2.0, grid.page[1]) - y],
            "letters": {k: list(v) for k, v in grid.letters.items()}}


def main():
    sys.path.insert(0, str(HERE))
    from common import load_spec
    fig_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    fig_dir = fig_dir if fig_dir.is_absolute() or fig_dir.exists() else HERE / fig_dir
    spec = load_spec(fig_dir.resolve())
    meta_file = out / "_build" / "panel_meta.json"
    meta = json.load(open(meta_file)) if meta_file.exists() else {}
    lay_file = out / "_build" / "figure_layout.json"
    layout = json.load(open(lay_file)) if lay_file.exists() else layout_from_spec(spec)
    src = out / "_build" / f"{spec['figure']}_methods_src.md"
    if not src.exists() and (fig_dir / "methods.md").exists():
        from common import write_methods
        write_methods(fig_dir, meta, src, None)
    for f in build_docs(spec, out, meta, layout, src.read_text(encoding="utf-8") if src.exists() else None):
        print("wrote", f)


if __name__ == "__main__":
    main()
