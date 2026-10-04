"""Validation test (user request 2026-09-19): "instead of decoding state
label, regress the learning curves instead: whisker learning curves, false
alarm and difference of the two (call it performance), but quantize it in
5 classes (20% splits), regress the value and check whether it lands in
the correct classes (so, a classification)." Added ALONGSIDE the existing
2-class curve-based `perfstate` target, not replacing it (user decision).

Mirrors `043_perfstate_curve_labels_test.py`'s role for the original
curve-based perfstate definition: a small-scale, whole_brain-only,
learning-stage-only plausibility check of a brand-new pipeline mechanism
before it's a candidate for a full production sweep -- NOT a result to
report as final.

Three regression targets per whisker trial (`ssl_timeresolved_decoding.
prep_perfquant_curve_targets`): `whisker_curve` (that trial's whisker
learning-curve p_mean), `falsealarm_curve` (the no-stim/catch-trial
curve's p_mean, linearly interpolated onto this trial's start_time), and
`performance_curve` (their difference). Quantile-class edges (5 classes,
20% splits) are fit ONCE on the FULL pooled population of all 88
curve-available learning-stage sessions' trial values (user decision
2026-09-19: "pooled across all sessions/trials") -- a cheap, non-neural
first pass -- then reused unchanged for every session's decode in the
second (neural) pass, which runs on a smaller TEST_MICE subset (same list
`043` used) to keep this validation's compute cost comparable to that
script's own plausibility check.

Sensory window only (5-50ms post-stim, current `SENSORY_WINDOW` -- see
`026`-`041`'s 2026-09-18 update), not a full time-resolved curve: this
checks whether the mechanism works at all, not when in the trial the
signal is strongest.

**Extended 2026-09-20** (user: real quantile-match accuracy tracked its
own shift-null closely in the first run -- "Ok try 1) and 2)", responding
to two proposed diagnostics): (1) the shift-null now also reports R^2, not
just quantile-match accuracy, since a substantial real R^2 alone doesn't
say anything without a null R^2 to compare it to (this run's own
`ssl_timeresolved_decoding.linear_shift_null_regression_quantile_pooled`
was extended to return both). (2) a RESIDUALIZED variant of each target
(`ssl_timeresolved_decoding.residualize_curve`: each target minus its own
session-local centered rolling mean, window=15 trials) is now decoded
alongside the raw target -- see that function's docstring for why: the
raw curve value and task-irrelevant slow drift in firing rates share a
common "where in the session are we" component that a regressor can
exploit on EITHER correctly-paired or shift-shuffled data, which is
exactly the real~null pattern the first run showed. Residualizing removes
that shared trend from the target; if the population still predicts the
trial-to-trial deviation that's left, that's evidence of real coupling
beyond shared drift. Both raw and residual quantile edges are fit in the
same global (all 88 sessions), pooled first pass -- residual values pooled
AFTER per-session residualizing (the rolling mean must stay within one
session, never cross a session boundary).
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)  # matches 026-041's 2026-09-18 update
DEAD_ZONE = (-0.001, 0.004)      # production dead zone (DEAD_ZONE_START_S/STOP_S)
N_REPEATS_TEST = 3               # vs. production 5 -- plausibility check only
N_WORKERS = 10
MIN_UNITS_PER_AREA = 5           # matches ssl_timeresolved_decoding's own constant
MIN_TRIALS_FOR_REGRESSION = 25   # enough for a meaningful 5-fold CV (5/fold min)

TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
RESID_WINDOW = 15  # trials, centered rolling mean -- see residualize_curve
# Every base target decoded twice: raw, and residualized against its own
# session-local trend (see module docstring, 2026-09-20 extension).
VARIANTS = TARGETS + [f"{t}_resid" for t in TARGETS]

# Same 20-mouse subset 043 used (curve-decodable sessions from 045's sweep) --
# keeps this validation's compute cost comparable to that earlier check.
TEST_MICE = [
    "AB092", "AB120", "AB121", "AB126", "AB127", "AB128", "AB129", "AB130",
    "AB133", "AB138", "AB143", "AB144", "AB147", "AB154", "AB158", "AB159",
    "MH011", "MH014", "MH022", "MH030",
]


def compute_global_quantile_edges() -> dict[str, np.ndarray]:
    """First pass (cheap, no neural data): pool every curve-available
    learning-stage session's per-trial target values (raw AND
    residualized -- residualizing is per-session, done BEFORE pooling, so
    the rolling mean never crosses a session boundary) and fit the 5-class
    quantile edges once, globally -- see module docstring."""
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        hitmiss_session_list, prep_perfquant_curve_targets, quantile_edges_from_pooled_values, residualize_curve,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    hm = hitmiss_session_list(sessions_tbl)
    learning = hm[hm["day_stage"] == "learning"]

    pooled = {v: [] for v in VARIANTS}
    n_ok = 0
    for _, r in learning.iterrows():
        out = prep_perfquant_curve_targets(dataset_root, r["session_id"], sessions_tbl, trials_tbl)
        if out is None:
            continue
        n_ok += 1
        for t in TARGETS:
            values = out[t].to_numpy()
            pooled[t].append(values)
            pooled[f"{t}_resid"].append(residualize_curve(values, window=RESID_WINDOW))
    print(f"global quantile-edge pass: {n_ok} learning-stage sessions with curve data")
    edges = {}
    for v in VARIANTS:
        values = np.concatenate(pooled[v])
        edges[v] = quantile_edges_from_pooled_values(values)
        print(f"  {v}: n={len(values)}, edges={np.round(edges[v], 4).tolist()}")
    return edges


def process_one_session(args: tuple) -> dict:
    mouse, session_id, scripts_dir, edges = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_bwm_trial_prep import load_reward_group
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units,
        decode_regression_quantile_pooled, linear_shift_null_regression_quantile_pooled,
        load_session_unit_spikes, prep_perfquant_curve_targets, residualize_curve, select_fixed_alpha_pooled,
        sliding_bin_population_matrices,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    out = dict(mouse=mouse, session_id=session_id, log=[], cells={}, reward_group=None)
    reward_group = load_reward_group(mouse)
    if reward_group is None:
        out["log"].append("no usable reward_group")
        return out
    out["reward_group"] = reward_group

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None:
        out["log"].append("no curve data for this session")
        return out
    for t in TARGETS:
        targets_df[f"{t}_resid"] = residualize_curve(targets_df[t].to_numpy(), window=RESID_WINDOW)

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    # `data_sufficiency_ok` is a binary-classification (hit/miss-style) check
    # -- meaningless here (a continuous regression target has no "classes"
    # before the post-hoc quantile binning), and passing it a dummy
    # all-True `y` as a stand-in incorrectly reads as zero negative-class
    # trials and skips every single session (found running this script
    # 2026-09-19). Sufficiency for a continuous target is just: enough
    # units, and enough trials for a meaningful 5-fold CV.
    if len(unit_ids) < MIN_UNITS_PER_AREA:
        out["log"].append(f"SKIPPED -- only {len(unit_ids)} units (< {MIN_UNITS_PER_AREA})")
        return out
    if len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        out["log"].append(f"SKIPPED -- only {len(targets_df)} trials (< {MIN_TRIALS_FOR_REGRESSION})")
        return out

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)
    matrices = sliding_bin_population_matrices(
        unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE,
    )
    X = matrices[0]

    for variant in VARIANTS:
        y = targets_df[variant].to_numpy()
        alpha = select_fixed_alpha_pooled(X, y, rng)
        real = decode_regression_quantile_pooled(X, y, alpha, edges[variant], rng, n_repeats=N_REPEATS_TEST)
        null = linear_shift_null_regression_quantile_pooled(
            X, y, alpha, edges[variant], rng, n_shuf=20, n_repeats=1,
        )
        out["cells"][variant] = dict(
            n_trials=len(y), alpha=alpha, real_quantile_acc=real["quantile_acc"], real_r2=real["r2"],
            null_acc_mean=float(np.nanmean(null["quantile_acc"])), null_acc_std=float(np.nanstd(null["quantile_acc"])),
            null_r2_mean=float(np.nanmean(null["r2"])), null_r2_std=float(np.nanstd(null["r2"])),
        )
        out["log"].append(
            f"[{variant}] n={len(y)} real_acc={real['quantile_acc']:.3f} vs "
            f"null_acc={np.nanmean(null['quantile_acc']):.3f}+-{np.nanstd(null['quantile_acc']):.3f} (chance=0.2) | "
            f"real_r2={real['r2']:.3f} vs null_r2={np.nanmean(null['r2']):.3f}+-{np.nanstd(null['r2']):.3f}"
        )
    return out


def main():
    edges = compute_global_quantile_edges()

    from ibl_ai_agent.data_locations import resolve_dataset_dir
    sessions_tbl = pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "sessions.parquet")

    tasks = []
    for mouse in TEST_MICE:
        sess = sessions_tbl[(sessions_tbl["subject_id"] == mouse) & (sessions_tbl["session_description"] == "whisker_0")
                             & (sessions_tbl["has_ephys"])]
        if len(sess) == 0:
            print(f"{mouse}: no learning-stage (whisker_0) has_ephys session, skipping")
            continue
        tasks.append((mouse, sess["session_id"].iloc[0], SCRIPTS_DIR, edges))

    records = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            print(f"\n=== {res['mouse']} / {res['session_id']} ({res['reward_group']}) ===")
            for line in res["log"]:
                print(f"  {line}")
            for variant, cell in res["cells"].items():
                records.append(dict(mouse=res["mouse"], session_id=res["session_id"],
                                     reward_group=res["reward_group"], variant=variant, **cell))

    df = pd.DataFrame(records)
    out_csv = OUT_DIR / "049_perfquant_curve_regression_test.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv.name}")

    if len(df) == 0:
        print("\nno sessions were decodable -- nothing to summarize")
        return

    print("\n=== summary across sessions ===")
    for variant in VARIANTS:
        sub = df[df["variant"] == variant]
        if len(sub) == 0:
            continue
        above_acc = int((sub["real_quantile_acc"] > sub["null_acc_mean"] + sub["null_acc_std"]).sum())
        above_r2 = int((sub["real_r2"] > sub["null_r2_mean"] + sub["null_r2_std"]).sum())
        print(f"{variant}: n_sessions={len(sub)} | "
              f"acc: real={sub['real_quantile_acc'].mean():.3f} vs null={sub['null_acc_mean'].mean():.3f} "
              f"(above null+1sd: {above_acc}/{len(sub)}) | "
              f"r2: real={sub['real_r2'].mean():.3f} vs null={sub['null_r2_mean'].mean():.3f} "
              f"(above null+1sd: {above_r2}/{len(sub)})")


if __name__ == "__main__":
    main()
