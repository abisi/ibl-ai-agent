"""Metric-definition figure for the two behavioral trial factors used to test
whether neural TCA components "track learning": smoothed P(lick) and rolling
d-prime, both computed directly from trials.parquet (see behavior_lib.py) and
both defined at every trial in the tensor's own trial axis, so they can be
correlated against a TCA trial-mode loading vector index-for-index later.
"""
import numpy as np
import matplotlib.pyplot as plt

from tca_lib import load_session_tables, align_whisker_trials_first_hit
from behavior_lib import smoothed_plick, rolling_dprime

SESSION_ID = "AB116_20240724_102941"
PLICK_WINDOW = 10  # trials; matches the WINDOW used in ssl-task-performance/000
DPRIME_WINDOW = 20  # trials; wider than P(lick) window since d-prime needs
                     # separate hit- and false-alarm-rate estimates, each
                     # noisier than a single rate estimate

sessions, units, trials = load_session_tables()
trials_session = trials[trials["session_id"] == SESSION_ID]
aligned = align_whisker_trials_first_hit(trials_session)

plick = smoothed_plick(aligned, window=PLICK_WINDOW)
dprime = rolling_dprime(trials_session, aligned, window_trials=DPRIME_WINDOW)

fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

axes[0].plot(aligned["whisker_trial_id"], aligned["lick_flag"], "o", ms=3, color="0.6")
axes[0].set_ylabel("lick_flag\n(raw, per trial)")
axes[0].set_title(f"{SESSION_ID}: behavioral trial-factor definitions")

axes[1].plot(aligned["whisker_trial_id"], plick, color="k", lw=1.5)
axes[1].set_ylabel(f"P(lick)\n(rolling mean, w={PLICK_WINDOW})")

axes[2].plot(aligned["whisker_trial_id"], dprime, color="tab:blue", lw=1.5)
axes[2].set_ylabel(f"d-prime\n(rolling, w={DPRIME_WINDOW})")
axes[2].set_xlabel("whisker trial id (0 = first active hit)")

for ax in axes:
    ax.axvline(0, color="crimson", lw=1, ls="--")

fig.tight_layout()
fig.savefig("002_behavior_variables.png", dpi=150, bbox_inches="tight")
print("saved 002_behavior_variables.png")
print(f"plick: {(~np.isnan(plick)).sum()}/{len(plick)} defined")
print(f"dprime: {(~np.isnan(dprime)).sum()}/{len(dprime)} defined")
