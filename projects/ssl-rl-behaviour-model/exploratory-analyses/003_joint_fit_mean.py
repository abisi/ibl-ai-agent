"""003 -- Joint fit of the group-mean targets (R+ and R- together) with one pretraining parameter set (b, a, v, initial weights) shared
by both groups; transfer parameters common by default, or group-specific for the parameters named in --free-transfer (nested model).
Nelder-Mead in log10 space from three starts (the previous group-specific values and their average). See rl_joint for details.
Output: fits/003_<model>_mean[_free-<params>].{json,png}
Run (haas): python 003_joint_fit_mean.py --model shared [--free-transfer tr] [--maxiter 3000]
"""
import argparse
import multiprocessing as mp

import common as C
import rl_joint as J

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(J.MODELS), default="shared")
    ap.add_argument("--targets", default=str(C.TARGETS / "final_mean_withoutfree"))
    ap.add_argument("--free-transfer", nargs="*", default=[], choices=J.DYN_KEYS)
    ap.add_argument("--maxiter", type=int, default=3000)
    J.run_mean(ap.parse_args())
