"""Paths, external-repo imports and provenance shared by all figures."""

import datetime
import hashlib
import importlib.metadata
import platform
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
