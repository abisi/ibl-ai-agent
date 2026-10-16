"""002 -- Parameter recovery of the joint fit (no data): simulate targets from known common parameters (one pretraining for R+ and R-),
refit from a default starting point, print true vs fitted b, a, v, tr and check that the simulated pretraining is identical across
groups. Run (haas): python 002_parameter_recovery.py --model shared [--maxiter 300]
"""
import argparse
import multiprocessing as mp

import common as C
import rl_joint as J

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(J.MODELS), default="shared")
    ap.add_argument("--maxiter", type=int, default=300)
    J.run_selftest(ap.parse_args())
