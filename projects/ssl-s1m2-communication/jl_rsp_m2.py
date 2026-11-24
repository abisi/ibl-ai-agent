"""RSP vs M2 time-shifted CCA on Jules Lebert's context dataset (user 2026-11-20: "For JL; now do RSP vs M2").

Same recipe, units, trials, conditions and artefact gap as the JL S1-M2 run (s1m2_lib with S1M2_DATASET=jl, 002 / 003 code
reused unchanged); only the source area changes: RSP (area_acronym_custom == "RSP", all RSP sub-areas pooled) takes the
S1 slot, so delay > 0 = RSP leads M2. M2 sets as S1-M2: wMO (MO-wM1 + MO-wM2 pooled, main), MO-wM1, MO-wM2.
>= 10 stable good + mua units per area (min_units of the JL S1-M2 run, kept for comparability).
Outputs: combined_results_ks4/ssl-s1m2-communication/jules_lebert/rsp_m2/ (sessions_v2.csv, maps_v3/, maps_v4_hires/,
run_meta_*.csv); pair names RSP__wMO / RSP__MO-wM1 / RSP__MO-wM2. Files in jules_lebert/ itself are not touched.

Run on haas (conda brain_wide_analysis, as jules_lebert/logs/run_all.sh):
  OMP_NUM_THREADS=1 nice -n 10 python jl_rsp_m2.py sessions
  OMP_NUM_THREADS=1 nice -n 10 python jl_rsp_m2.py maps --profile v3 --n-jobs 40 [--skip-done]
  OMP_NUM_THREADS=1 nice -n 10 python jl_rsp_m2.py maps --profile hires --n-jobs 40 [--skip-done]
  python jl_rsp_m2.py post     # stats_v3.csv, stats_hires_total.csv, replication/*.pdf (004 / 005 / 007, jl_pages
                               # relabelled S1 -> RSP; the paper's S1-M2 values stay as grey reference)
"""
import argparse
import importlib
import os
import time

os.environ["S1M2_DATASET"] = "jl"
import pandas as pd
from joblib import Parallel, delayed

import s1m2_lib as L

SOURCE = "RSP"
OUT_RSP = L.OUT / "rsp_m2" if L.OUT.name == "jules_lebert" else L.OUT


def apply_source():
    """Point s1m2_lib at RSP + the rsp_m2 output folder (called in the main process and in every loky worker)."""
    L.CONFIG["s1_area"] = SOURCE
    L.CONFIG["source_note"] = "RSP in the S1 slot: delay > 0 = RSP leads M2; n_s1 / s1_units = RSP units"
    L.OUT = OUT_RSP


def job(sid, cond, rows, profile):
    os.environ["S1M2_DATASET"] = "jl"
    apply_source()
    return importlib.import_module("003_ccv_maps").safe(sid, cond, rows, profile)


PAIRS = [f"{SOURCE}__wMO", f"{SOURCE}__MO-wM1", f"{SOURCE}__MO-wM2"]
# jl_pages labels the source "S1"; relabel to RSP (the paper's S1-M2 values stay as the grey reference, marked as such)
LABELS = [("SSp-bfd__wMO", f"{SOURCE}__wMO"), ("S1 (SSp-bfd) -> M2", f"{SOURCE} -> M2"), ("S1 leads", f"{SOURCE} leads"),
          ("S1-leading", f"{SOURCE}-leading"), ("S1 leading", f"{SOURCE} leading"), ("S1 -> M2", f"{SOURCE} -> M2"),
          ("S1 and M2", f"{SOURCE} and M2"), ('f"paper\\n{plab}', 'f"paper S1-M2\\n{plab}'),
          ('label=f"paper {pc}"', 'label=f"paper S1-M2 {pc}"')]


def relabelled_jl_pages():
    """jl_pages with RSP labels, registered as sys.modules['jl_pages'] so 005 / 007 use it."""
    import sys
    import types
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jl_pages.py"), encoding="utf-8").read()
    # page_v3 panel A: the paper maps are S1-M2, keep their S1 axis label (first occurrence = page_v3; hires has no
    # paper panels)
    yl = 'ax.set_ylabel("Delay (ms; > 0: S1 leads)")'
    assert yl in src
    src = src.replace(yl, f'ax.set_ylabel(PAPER_YLABEL if ttl.startswith("Paper") else '
                          f'"Delay (ms; > 0: {SOURCE} leads)")', 1)
    src += '\nPAPER_YLABEL = "Delay (ms; > 0: " + "S" + "1 leads)"\n'   # split so the relabelling below keeps it
    src = src.replace('f"Paper, expert\\n{pc} trials', 'f"Paper S1-M2, expert\\n{pc} trials')
    for a, b in LABELS:
        assert a in src, a
        src = src.replace(a, b)
    mod = types.ModuleType("jl_pages")
    mod.__file__ = "jl_pages.py (relabelled for RSP by jl_rsp_m2.py)"
    exec(compile(src, "jl_pages_rsp", "exec"), mod.__dict__)
    sys.modules["jl_pages"] = mod
    return mod


def post():
    """stats (004 jl branch) for v3, replication pages (005, total, all cells + option a) and hires pages (007)."""
    import sys
    relabelled_jl_pages()
    m4 = importlib.import_module("004_stats_figures")
    m4.PAIRS = PAIRS
    L.set_profile("v3")
    sys.argv = ["004", "--profile", "v3", "--stats-only"]
    m4.main()
    m5 = importlib.import_module("005_replication_figure")
    m5.PAIR = PAIRS[0]
    for extra in ([], ["--clean"]):
        sys.argv = ["005", "--kind", "total", *extra]
        m5.main()
    L.set_profile("hires")
    m7 = importlib.import_module("007_hires_figure")
    m7.PAIR = PAIRS[0]
    sys.argv = ["007", "--kind", "total"]
    m7.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sessions", "maps", "post"])
    ap.add_argument("--profile", default="v3", choices=list(L.PROFILES))
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--conditions", nargs="*", default=list(L.CONFIG["conditions"]))
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    apply_source()
    L.OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "sessions":
        importlib.import_module("002_sessions").main_jl()
        return
    if a.step == "post":
        import sys
        sys.argv = sys.argv[:1]
        return post()
    m3 = importlib.import_module("003_ccv_maps")
    L.set_profile(a.profile)
    s = pd.read_csv(L.OUT / "sessions_v2.csv")
    jobs = [(sid, cond) for sid in s.session_id.unique() for cond in a.conditions
            if not (a.skip_done and all((m3.maps_dir() / f"{sid}__{p}__{cond}.npz").exists()
                                        for p in s[s.session_id == sid].pair))]
    print(f"{s.session_id.nunique()} sessions, {len(jobs)} jobs, profile {a.profile}, out {m3.maps_dir()}", flush=True)
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(job)(sid, cond, s[s.session_id == sid], a.profile)
                                                    for sid, cond in jobs)
    pd.DataFrame([m for r in res for m in r]).to_csv(
        L.OUT / f"run_meta_v{L.CONFIG['version']}_{time.strftime('%Y%m%d_%H%M')}.csv", index=False)


if __name__ == "__main__":
    main()
