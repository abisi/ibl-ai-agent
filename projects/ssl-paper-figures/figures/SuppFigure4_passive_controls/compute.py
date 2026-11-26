"""One compute_<key>() per panel (shared_passive_controls), each writing data/<csv> of the dated snapshot, with the run meta used
by the caption. Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/<this folder>/compute.py
[--only key ...]; then build.py figures/<this folder>.
"""
import argparse
import json
import sys
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
import shared_passive_controls as S  # noqa: E402
from common import load_spec, snapshot_dir  # noqa: E402

SPEC = load_spec(FIG_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    out = snapshot_dir(SPEC)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "_build").mkdir(exist_ok=True)
    mf = out / "_build" / "panel_meta.json"
    allmeta = json.load(open(mf)) if mf.exists() else {}
    for p in SPEC["panels"]:
        key = p.get("key", p["id"])
        fn = getattr(S, f"compute_{key}", None)
        if fn is None or "data" not in p or (a.only and key not in a.only):
            continue
        df = fn()
        df.to_csv(out / "data" / p["data"], index=False)
        allmeta[p["id"]] = S.META.get(key, {})
        print(p["id"], key, len(df), flush=True)
    allmeta["_shared"] = dict(tables=str(S.TAB), category_column=S.CATEGORY_COLUMN, n_perm=S.N_PERM)
    json.dump(allmeta, open(mf, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
