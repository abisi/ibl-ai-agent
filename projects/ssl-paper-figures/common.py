"""Paths, external-repo imports and provenance shared by all figures."""

import datetime
import hashlib
import importlib.metadata
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

CODE_ROOT = Path.home() / "code"  # haas checkouts of the analysis repos
EXTERNAL_REPOS = {"behaviour_analysis": CODE_ROOT / "behaviour_analysis", "NWB_reader": CODE_ROOT / "NWB_reader"}
FIGURES_ROOT = axel_bisi_root() / "manuscripts" / "Bisi_Hamon" / "Figures"
CACHE_DIR = FIGURES_ROOT / "_cache"
REFERENCE_SHEET = axel_bisi_root() / "dataset_info" / "joint_mouse_reference_weight.xlsx"
NWB_DIR = axel_bisi_root() / "NWB_ks4"


def import_external_repos():
    """Put the analysis repos on sys.path. learning_utils imports statannotations at module level for its plotting
    helpers only; it is not installed in the haas env, so a stub stands in (no figure code uses Annotator)."""
    for path in EXTERNAL_REPOS.values():
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if "statannotations" not in sys.modules:
        import types
        stub = types.ModuleType("statannotations")
        stub.Annotator = types.ModuleType("statannotations.Annotator")
        stub.Annotator.Annotator = None
        sys.modules["statannotations"], sys.modules["statannotations.Annotator"] = stub, stub.Annotator


def load_spec(figure_dir):
    return yaml.safe_load(open(Path(figure_dir) / "spec.yaml"))


def snapshot_dir(spec, date=None):
    date = date or datetime.date.today().isoformat()
    sub = spec.get("output_subdir", "")
    return FIGURES_ROOT / sub / f"{spec['figure']}_{date}"


def repo_state(path, files=()):
    """Commit and dirty flag of a git repo; for a repo without git, the md5 of the files used."""
    path = Path(path)
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
        return {"path": str(path), "commit": commit, "dirty": dirty}
    except subprocess.CalledProcessError:
        md5 = {f: hashlib.md5((path / f).read_bytes()).hexdigest() for f in files if (path / f).exists()}
        return {"path": str(path), "commit": None, "note": "not under git", "md5": md5}


def environment():
    pkgs = ["numpy", "pandas", "scipy", "matplotlib", "pynwb"]
    return {"python": platform.python_version(), "machine": platform.node(),
            **{p: importlib.metadata.version(p) for p in pkgs}}


def two_group_tests(a, b):
    """R+ vs R- unpaired comparison: Mann-Whitney U and Welch t, both reported (user rule)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    welch = stats.ttest_ind(a, b, equal_var=False)
    return {"n_R+": len(a), "n_R-": len(b), "mannwhitney_U": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
            "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue)}


# ---------------------------------------------------------------- captions

SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
CAPTION_FIELD = re.compile(r"\{([^{}:]+)(?::([^{}]+))?\}")


def format_p(p):
    """Exact P value, two significant digits (Nature asks for exact values)."""
    if p >= 0.001:
        return f"*P* = {p:#.2g}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"*P* = {mantissa} × 10{str(int(exponent)).translate(SUPERSCRIPT)}"


def fill_caption(template, meta):
    """Replace {dotted.path[:fmt]} fields with values from a panel's run meta. fmt is a format spec ('d', '.2f') or 'p'
    for a P value. Missing fields stay visible as [?path] so they are caught at review."""
    def field(match):
        value = meta
        for key in match.group(1).strip().split("."):
            if not isinstance(value, dict) or key not in value:
                return f"[?{match.group(1)}]"
            value = value[key]
        fmt = match.group(2)
        if fmt == "p":
            return format_p(value)
        return format(value, fmt) if fmt else str(value)
    return CAPTION_FIELD.sub(field, " ".join(template.split()))


def write_caption(spec, meta, path):
    """Figure legend in journal style: bold title, one sentence block per panel, then the shared methods footer."""
    caption = spec.get("caption", {})
    shared = meta.get("_shared", {})
    parts = [f"**Fig. {spec['number']} | {fill_caption(caption.get('title', spec['figure']), shared)}**"]
    for panel in spec["panels"]:
        text = panel.get("caption") or f"[caption missing: {panel['title']}]"
        parts.append(f"**{panel['id']}**, {fill_caption(text, {**shared, **meta.get(panel['id'], {})})}")
    if caption.get("footer"):
        parts.append(fill_caption(caption["footer"], shared))
    text = f"# {spec['figure']} -- caption (draft, generated)\n\n" + " ".join(parts) + "\n"
    path.write_text(text, encoding="utf-8")
