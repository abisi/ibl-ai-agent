import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

TEST_SESSIONS = [
    ("AB080_20230622_152205", "AB080", "R+", "good"),
    ("MH021_20250309_122552", "MH021", "R+", "good"),
]


def run_combo(decode_target, alignment):
    sys.argv = ["024_master_sweep.py", decode_target, alignment]
    spec = importlib.util.spec_from_file_location("master024", Path(__file__).parent / "024_master_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    for sid, subj, rg, lc in TEST_SESSIONS:
        t0 = time.time()
        rows = mod.process_one_session((sid, subj, rg, lc, scripts_dir, decode_target, alignment))
        dt = time.time() - t0
        n_ok = sum(1 for r in rows if r.get("skipped_reason") is None)
        n_skip = len(rows) - n_ok
        print(f"[{decode_target}/{alignment}] {sid}: {n_ok} ok rows, {n_skip} skipped -- {dt:.1f}s")
        ok_rows = [r for r in rows if r.get("skipped_reason") is None]
        if ok_rows:
            r = ok_rows[0]
            print(f"    sample row: area_col={r['area_col']} area_value={r['area_value']} condition_type={r['condition_type']} "
                  f"condition_value={r['condition_value']} n_units={r['n_units']} n_trials={r['n_trials']} "
                  f"curve_len={len(r['real_curve'])} has_crossgen={r.get('crossgen_curve_to_other') is not None}")
        # Verify condition_type coverage
        cond_types = {r['condition_type'] for r in ok_rows}
        print(f"    condition_types present: {cond_types}")
        crossgen_present = any(r.get('crossgen_curve_to_other') is not None for r in ok_rows)
        print(f"    any crossgen curve present: {crossgen_present}")


if __name__ == "__main__":
    print("=== hitmiss / stim ===")
    run_combo("hitmiss", "stim")
    print("\n=== modality / stim ===")
    run_combo("modality", "stim")
    print("\n=== modality / lick ===")
    run_combo("modality", "lick")
