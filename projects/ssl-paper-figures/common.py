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

# canonical git checkouts on the NAS (GitHub abisi/behaviour_analysis, abisi/NWB_reader); ~/code copies on haas are old
EXTERNAL_REPOS = {"behaviour_analysis": axel_bisi_root() / "behaviour_analysis", "NWB_reader": axel_bisi_root() / "NWB_reader"}
FIGURES_ROOT = axel_bisi_root() / "manuscripts" / "Bisi_Hamon" / "Figures"
CACHE_DIR = FIGURES_ROOT / "_cache"
# canonical sheet on share_internal (M:\share_internal on the PC, /mnt/share_internal on haas); the copy in
# analysis/Axel_Bisi/dataset_info is stale (March 2026: AB144 / AB155 / MH038 still excluded, no MH033), fixed 2026-11-06
_SHARE_INTERNAL = next((p for p in (Path(r"M:\share_internal"), Path("/mnt/share_internal")) if p.exists()), None)
REFERENCE_SHEET = _SHARE_INTERNAL / "Axel_Bisi_Share" / "dataset_info" / "joint_mouse_reference_weight.xlsx"
NWB_DIR = axel_bisi_root() / "NWB_ks4"
SESSION_NWB = re.compile(r"^[A-Z]+\d+_\d{8}_\d{6}\.nwb$")  # skips *_backup_<date>.nwb left by reconversions


def session_nwbs(mouse_id=None):
    """Session NWB files in NWB_DIR (one per session; backups excluded), optionally of one mouse."""
    files = (f for f in NWB_DIR.glob(f"{mouse_id or '*'}_*.nwb") if SESSION_NWB.match(f.name))
    return sorted(files)


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
    """Commit of a git repo and the files whose content differs from it (line-ending-only changes of the NAS checkouts
    ignored); for a repo without git, the md5 of the files used. NAS checkouts need safe.directory on haas."""
    path = Path(path)
    git = ["git", "-c", "safe.directory=*", "-C", str(path)]
    try:
        commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        changed = subprocess.run(git + ["diff", "--ignore-cr-at-eol", "--name-only", "HEAD"], capture_output=True,
                                 text=True).stdout.split()
        changed = [f for f in changed if not f.startswith(".idea")]
        return {"path": str(path), "commit": commit, "dirty": bool(changed), "changed_files": changed}
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
