"""004 -- Per-mouse fits with the learning parameters (b, a, v, tr) fixed to a 003 joint mean fit without group-specific transfer
parameters (same learning rule for every mouse, whatever its group); initial weights fitted per mouse.
Output: fits/004_<model>_individual_common_dynamics.csv
Run (haas): python 004_joint_fit_individual.py --model shared [--base fits/003_shared_mean.json] [--n-jobs 32]
"""
import argparse
import multiprocessing as mp

import common as C
import rl_joint as J

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(J.MODELS), default="shared")
    ap.add_argument("--targets", default=str(C.TARGETS / "final_individual_withoutfree"))
    ap.add_argument("--base", default=None)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--n-jobs", type=int, default=32)
    a = ap.parse_args()
    a.base = a.base or str(C.FITS / f"003_{a.model}_mean.json")
    J.run_individual(a)
