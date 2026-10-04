"""Generic, checkpointed/resumable master sweep for the full rebuild
(user request 2026-09-11): one script handles any (decode_target,
alignment, day_stage) combination, all 3 condition types (whole session,
halves, perf-states), all 3 area schemes (`area_group`, `area_acronym_custom`,
`whole_brain`), both cohorts -- writing one reusable long-format parquet
that every figure/stats script downstream reads from (no recomputation per
figure).

Usage: `python 024_master_sweep.py <decode_target> <alignment> [day_stage]`
  decode_target: 'hitmiss' (lick_flag, whisker trials only), 'modality'
    (whisker vs auditory trial_type), or 'perfstate' (high vs low
    performance state itself as the target -- added 2026-09-12; since
    perf-state IS the target here, there is no separate 'perfstate'
    condition_type for this target, and only 'whole' (full-session) is
    computed -- no 'half' split either, per user decision 2026-09-12
    ("only do full session runs" for perf decoding))
  alignment: 'stim' (start_time-aligned; only valid alignment for hitmiss
    and perfstate, since miss trials have no lick_time to anchor a
    lick-aligned window to, and perf-state is not inherently tied to a lick
    event -- user-confirmed 2026-09-11/12) or 'lick' (first-lick-aligned,
    modality only, restricted to lick_flag==1 trials). Since 2026-09-27 the lick
    anchor is the CORRECTED `first_lick_time` (start_time + lick_time -
    response_window_start_time, skills/ssl-lick-alignment); the stored `lick_time` is
    late by the artifact window. Runs before 2026-09-27 used the stored lick_time
    (backed up as *.bak-prelickfix-20260927); SSL_CORRECT_LICK_TIME=0 reproduces them.
  day_stage: 'learning' (default) or 'expert'

Design locked with the user 2026-09-11 (see project `question.md`):
  - causal bins: `(t-50ms, t]`, labeled at their end `t` (never uses spikes
    after the labeled time), 5ms stride -- `causal_bin_edges` (was
    center-labeled `sliding_bin_edges` at 10ms stride in the prior sweeps
    002/008/015/021, all superseded by this rebuild).
  - dead zone widened to -10ms/+5ms (was the canonical minimum -1ms/+4ms)
    -- explicit user choice, allowed per `ssl_artifact_dead_zone.md`
    ("wider is always an acceptable conservative choice").
  - third area scheme `whole_brain`: every QC-passing unit in a session,
    one pseudo-area (`add_whole_brain_column`).
  - cross-condition generalization for `half` and `perfstate` (not `whole`,
    which has only one condition value): fit on all of condition A's
    trials, test on all of condition B's, both directions, using each
    direction's own within-condition C (no extra grid search).
  - `unit_area_labels.parquet` rebuilt from the user's updated allen_utils
    custom area groups (Somatosensory split into whisker/orofacial/body,
    etc. -- `scripts/build_ssl_area_labels.py`, run 2026-09-11) before this
    sweep, so `area_group` here already reflects the new scheme.
  - cluster-mass permutation significance approach kept as-is (per user
    confirmation), not changed to a "more continuous" per-bin corrected-p
    version.
  - second null control added 2026-09-13, `hitmiss`/`perfstate` only (not
    `modality`, which has no comparable slow-drift confound to guard
    against): `linear_shift_null_curves` -- a linear (truncating, no
    wraparound) shift null alongside the existing full-permutation
    label-shuffle null, stored separately as `shift_null_curves`. Where a
    label shuffle destroys all label structure, a linear shift preserves the
    label sequence's own autocorrelation/block structure (e.g. perf_state's
    5-trial blocks, slow session-wide hit-rate drift) while breaking its
    true trial-by-trial correspondence to the neural data -- catches the
    specific failure mode where neural activity and the label each drift
    slowly and independently, which a plain permutation null wouldn't
    detect. (Started as a circular `np.roll` shift, corrected 2026-09-13 to
    a non-wrapping linear shift instead, since a circular wrap manufactures
    an end-to-start pairing a real session's drift never has.)

Cost note: combines what used to be 2 separate sweeps (halves-only,
perfstate-only) into one, at half the stride (2x bins) plus 2 cheap
cross-gen curves per paired condition type -- expect several hours per
(decode_target, alignment) run. Run sequentially, not concurrently with
other sweeps (lesson from 015/021's mutual slowdown this session). The
2026-09-13 shift-null addition roughly doubles null-curve cost for
hitmiss/perfstate runs specifically (modality unaffected).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

STIM_WINDOW = (-0.2, 0.6)
LICK_WINDOW = (-0.6, 0.2)  # full window (user correction 2026-09-12); zoomed figures/pre-lick quantification use smaller sub-windows at plot/stats time, not here
BIN_WIDTH = 0.05
STRIDE = 0.005
WIDE_DEAD_ZONE = (-0.010, 0.005)
N_REPEATS = 5
# Worker-pool size: default 110 (user decision 2026-09-14, sized for the
# haas056.rcp.epfl.ch remote host's 112 cores now that it's set up as a
# compute target) -- override with SSL_DECODE_N_WORKERS when running on a
# smaller machine (e.g. this local machine has 32 cores; 110 workers here
# would badly oversubscribe it).
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "110"))
# Fill-in mode (2026-09-28, MIN_TRIALS_PER_CLASS lowered 5 -> 3): SSL_FILLIN_TARGETS = parquet with columns
# session_id, area_col, area_value, condition_type; only those condition groups are recomputed and their rows REPLACE
# the existing ones in the tag's parquet (every other row stays byte-identical). Targets come from 024b.
FILL_PATH = os.environ.get("SSL_FILLIN_TARGETS")
FILL = (set(pd.read_parquet(FILL_PATH)[["session_id", "area_col", "area_value", "condition_type"]].itertuples(index=False, name=None))
        if FILL_PATH else None)


def _want(session_id, area_col, area_value, condition_type):
    return FILL is None or (session_id, area_col, area_value, condition_type) in FILL


N_SHUF = 25  # null-distribution shuffle count (user decision 2026-09-12: compromise between the cheap
             # single-shuffle surrogate and a full 100-shuffle null, given the ~20x-per-shuffle-batch cost)
NULL_N_REPEATS = 2  # each null draw needs less CV precision than the real curve -- keeps N_SHUF tractable

DECODE_TARGET = sys.argv[1]  # 'hitmiss' or 'modality'
ALIGNMENT = sys.argv[2]      # 'stim' or 'lick'
DAY_STAGE = sys.argv[3] if len(sys.argv) > 3 else "learning"
# Area-scheme subset (user decision 2026-09-11, staged rollout given cost):
# start with 'whole_brain' only (cheapest -- 1 pseudo-area/session vs. ~18
# for area_group+area_acronym_custom combined), decide on the rest after
# seeing those results. sys.argv[4], comma-separated, e.g. "whole_brain" or
# "area_group,area_acronym_custom,whole_brain" (default: all three).
AREA_SCHEMES = sys.argv[4].split(",") if len(sys.argv) > 4 else ["area_group", "area_acronym_custom", "whole_brain"]

assert DECODE_TARGET in ("hitmiss", "modality", "perfstate")
assert ALIGNMENT in ("stim", "lick")
if DECODE_TARGET in ("hitmiss", "perfstate"):
    assert ALIGNMENT == "stim", f"{DECODE_TARGET} decoding is stim-aligned only"

OUT_DIR = Path(__file__).resolve().parent
_suffix = "" if DAY_STAGE == "learning" else f"_{DAY_STAGE}"
_scheme_tag = "" if AREA_SCHEMES == ["area_group", "area_acronym_custom", "whole_brain"] else f"_{'-'.join(AREA_SCHEMES)}"
TAG = f"{DECODE_TARGET}_{ALIGNMENT}{_suffix}{_scheme_tag}"
PARTIAL_PATH = OUT_DIR / f"024_master_results_{TAG}.parquet"
BIN_EDGES_PATH = OUT_DIR / f"024_bin_edges_{TAG}.json"

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _worker_init():
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"


def process_one_session(args: tuple) -> list[dict]:
    session_id, subject_id, reward_group, learning_category, scripts_dir, decode_target, alignment = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH,
        LICK_TIME_DEFINITION,
        add_whole_brain_column,
        area_units,
        areas_with_enough_units,
        causal_bin_edges,
        linear_shift_null_curves_pooled,
        data_sufficiency_ok,
        decode_curve_pooled,
        decode_curve_crossgen,
        label_shuffle_null_curves_pooled,
        lick_aligned_bin_population_matrices,
        load_session_unit_spikes,
        prep_hitmiss_trials,
        prep_lick_aligned_trials,
        prep_modality_trials,
        prep_perfstate_trials_curve,
        select_fixed_c_pooled,
        sliding_bin_population_matrices,
        wide_window_matrix_from_bins,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    window = STIM_WINDOW if alignment == "stim" else LICK_WINDOW
    bin_edges = causal_bin_edges(window, bin_width=BIN_WIDTH, stride=STRIDE)
    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))

    base_row = dict(
        session_id=session_id, subject_id=subject_id, reward_group=reward_group,
        learning_category=learning_category, decode_target=decode_target, alignment=alignment,
        # provenance (skills/ssl-lick-alignment quality gate, 2026-09-27): which lick time lick-aligned
        # windows are centred on -- corrected first_lick_time unless SSL_CORRECT_LICK_TIME=0
        lick_time_definition=LICK_TIME_DEFINITION if alignment == "lick" else None,
    )

    # --- trial prep: 'whole' set (also carries the 'half' column) + 'perfstate' set ---
    # perf-state definition: `prep_perfstate_trials_curve` (curve-based,
    # user decision 2026-09-16: adopt as pipeline default) -- learning-stage
    # only, returns None if no learning-curve file exists for this mouse
    # (e.g. most MH-prefix mice, and all expert-stage sessions), in which
    # case that session is skipped for perfstate the same way any other
    # "no usable trials" case is, NOT silently downgraded to the old
    # block-median `prep_perfstate_trials_generic` (kept in this module,
    # untouched, for backward comparison -- see 043/044/045 in
    # exploratory-analyses/).
    if decode_target == "hitmiss":
        trials_whole = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
        trials_perfstate = prep_perfstate_trials_curve(
            dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["whisker_trial"],
        )
    elif decode_target == "perfstate":
        # perf-state IS the target here, so there is no separate 'perfstate' condition
        # to also split by. Full-session decoding only (user decision 2026-09-12:
        # "only do full session runs" -- no 'half' split for this target either,
        # since a within-session half-split of the perf-state label itself adds
        # little and roughly doubles the cost for no clear question it answers).
        # whisker-trial-only (REVERTED 2026-09-16: briefly widened to all 3
        # trial types, then reverted -- mixing whisker/auditory/no_stim
        # trials into the SAME time-resolved curve adds post-stimulus
        # within-class heterogeneity unrelated to perf-state, since evoked
        # responses differ hugely by trial type; more trials there isn't
        # more power, it's more noise, and it muddies any post-stimulus
        # interpretation. The "more trials helps" case only actually holds
        # for a pre-stimulus-only aggregate -- see the separate, dedicated
        # `perfstate_baseline` analysis planned for that instead of baking
        # it into this time-resolved target.)
        trials_whole = prep_perfstate_trials_curve(
            dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["whisker_trial"],
        )
        trials_perfstate = None
    else:
        if alignment == "stim":
            trials_whole = prep_modality_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
        else:
            trials_whole = prep_lick_aligned_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
        trials_perfstate = prep_perfstate_trials_curve(
            dataset_root, session_id, sessions_tbl, trials_tbl,
            decode_trial_types=["whisker_trial", "auditory_trial"], licked_only=(alignment == "lick"),
        )

    if trials_whole is None or len(trials_whole) == 0:
        return [dict(base_row, area_col=None, area_value=None, condition_type=None, condition_value=None,
                      skipped_reason="no usable trials for this decode_target/alignment")]

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)

    def matrices_and_y(sub_trials: pd.DataFrame) -> tuple[list[np.ndarray] | None, np.ndarray, np.ndarray]:
        """Returns (matrices, y, is_whisker_mask) for a given unit_ids-free
        trial subset -- unit_ids filled in by the caller per area."""
        if decode_target == "hitmiss":
            y = sub_trials["lick_flag"].to_numpy().astype(bool)
            is_whisker = np.ones(len(sub_trials), dtype=bool)
        elif decode_target == "perfstate":
            y = (sub_trials["perf_state"] == "high").to_numpy()
            # NOT hardcoded True anymore (2026-09-16): decode_trial_types now
            # includes auditory_trial/no_stim_trial alongside whisker_trial,
            # so is_whisker must reflect each row's actual trial type -- the
            # whisker-stimulus artifact dead zone only applies to real
            # whisker trials; applying it to auditory/no_stim rows would
            # incorrectly excise a valid spike window there.
            is_whisker = (sub_trials["trial_type"] == "whisker_trial").to_numpy()
        else:
            y = (sub_trials["trial_type"] == "whisker_trial").to_numpy()
            is_whisker = y.copy()
        return y, is_whisker

    def build_matrices(sub_trials: pd.DataFrame, unit_ids: np.ndarray, is_whisker: np.ndarray) -> list[np.ndarray]:
        if alignment == "stim":
            start_time = sub_trials["start_time"].to_numpy()
            return sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges, dead_zone=WIDE_DEAD_ZONE)
        else:
            event_time = sub_trials["first_lick_time"].to_numpy()  # corrected first lick (2026-09-27), NOT stored lick_time
            start_time = sub_trials["start_time"].to_numpy()
            return lick_aligned_bin_population_matrices(unit_spikes, unit_ids, event_time, start_time, is_whisker, bin_edges, dead_zone=WIDE_DEAD_ZONE)

    # One shuffle type per (decode_target, alignment) (user decision
    # 2026-09-15, replacing the earlier "compute both, hitmiss/perfstate
    # only" policy; extended 2026-09-17 for modality_lick -- see below):
    # shift-null for hitmiss/perfstate (their meaningful confound is slow
    # session-level drift/autocorrelation, which label-shuffle doesn't
    # probe), label-shuffle for modality_stim (no comparable drift confound
    # there -- modality_stim uses the FULL, experimenter-randomized
    # whisker+auditory trial set, `prep_modality_trials`'s "any lick_flag",
    # so trial-type identity really is independent of time).
    #
    # modality_lick is DIFFERENT (user finding, 2026-09-17): it's restricted
    # to `lick_flag==1` (`prep_lick_aligned_trials`, needs a real `lick_time`
    # to align to) -- and that hit-filter is NOT time-uniform the same way
    # for both classes. Auditory trials are rewarded on lick in both cohorts
    # at a high, session-stable rate, so the auditory-hit subset stays close
    # to the original randomized delivery; whisker hit rate is behaviorally
    # variable across the session (learning/engagement, and for R- it isn't
    # even the rewarded response), so the whisker-hit subset is NOT uniform
    # in time. That gives the two decoded classes different temporal
    # densities within the session purely as a hit-filter artifact -- a
    # label-shuffle null would destroy exactly the label-autocorrelation
    # structure needed to catch a decoder exploiting that artifact (same
    # failure mode hitmiss/perfstate already moved off label-shuffle for),
    # so modality_lick gets the shift-null instead, same as hitmiss/perfstate.
    compute_shift_null = decode_target in ("hitmiss", "perfstate") or (decode_target == "modality" and alignment == "lick")
    compute_label_shuffle = not compute_shift_null

    def decode_condition(sub_trials: pd.DataFrame, unit_ids: np.ndarray) -> dict | None:
        y, is_whisker = matrices_and_y(sub_trials)
        ok, reason = data_sufficiency_ok(len(unit_ids), y)
        if not ok:
            return dict(ok=False, reason=reason, n_trials=len(y))
        matrices = build_matrices(sub_trials, unit_ids, is_whisker)
        X_wide = wide_window_matrix_from_bins(matrices)
        # Pooled-scoring + resample-on-missing-class estimator is the
        # default as of 2026-09-15 (user decision: "the pooled-cv estimator
        # should be the default method") -- was `select_fixed_c`/
        # `decode_curve`/`label_shuffle_null_curves`/`linear_shift_null_curves`
        # (still in ssl_timeresolved_decoding.py, untouched, for the
        # already-run pooled-CV pilot comparison and any results computed
        # under the old default before this switch).
        C = select_fixed_c_pooled(X_wide, y, rng)
        real = decode_curve_pooled(matrices, y, C, rng, n_repeats=N_REPEATS)
        null_curves = (
            label_shuffle_null_curves_pooled(matrices, y, C, rng, n_shuf=N_SHUF, n_repeats=NULL_N_REPEATS)
            if compute_label_shuffle else None
        )
        shift_null_curves = (
            linear_shift_null_curves_pooled(matrices, y, C, rng, n_shuf=N_SHUF, n_repeats=NULL_N_REPEATS)
            if compute_shift_null else None
        )
        return dict(ok=True, n_trials=len(y), C=C, real_curve=real, null_curves=null_curves,
                    shift_null_curves=shift_null_curves, peak_acc=float(np.nanmax(real)), matrices=matrices, y=y)

    def null_fields(res: dict) -> dict:
        """`null_curves` and `shift_null_curves` are mutually exclusive
        (exactly one is non-None per `compute_label_shuffle`/
        `compute_shift_null` above) -- bug fixed 2026-09-17: this used to
        unconditionally do `res["null_curves"][0].tolist()` for
        `surrogate_curve`/`.tolist()` for the stored column, which crashed
        with `TypeError("'NoneType' object is not subscriptable")` the
        moment `null_curves` was None (as it now is for modality_lick,
        which switched to shift-null-only) -- `shift_null_curves` already
        had the `is not None` guard `null_curves` was missing."""
        null_curves_list = res["null_curves"].tolist() if res["null_curves"] is not None else None
        shift_null_curves_list = res["shift_null_curves"].tolist() if res["shift_null_curves"] is not None else None
        surrogate = res["null_curves"][0].tolist() if res["null_curves"] is not None else res["shift_null_curves"][0].tolist()
        return dict(surrogate_curve=surrogate, null_curves=null_curves_list, shift_null_curves=shift_null_curves_list)

    rows = []
    for area_col in AREA_SCHEMES:
        areas = areas_with_enough_units(session_id, area_col, area_labels)
        for area_value in areas:
            unit_ids = area_units(session_id, area_col, area_value, area_labels)
            area_row = dict(base_row, area_col=area_col, area_value=area_value, n_units=len(unit_ids))

            # --- whole session (no split, no cross-gen) ---
            res = decode_condition(trials_whole, unit_ids) if _want(session_id, area_col, area_value, "whole") else None
            if res is None:
                pass
            elif not res["ok"]:
                rows.append(dict(area_row, condition_type="whole", condition_value="whole",
                                  n_trials=res["n_trials"], skipped_reason=res["reason"]))
            else:
                rows.append(dict(area_row, condition_type="whole", condition_value="whole", n_trials=res["n_trials"],
                                  C=res["C"], real_curve=res["real_curve"].tolist(), **null_fields(res),
                                  peak_acc=res["peak_acc"], crossgen_curve_to_other=None, skipped_reason=None))

            # --- halves (with cross-gen) -- not for perfstate: full-session runs only ---
            if decode_target != "perfstate" and _want(session_id, area_col, area_value, "half"):
                half_res = {}
                for half_val in ("first", "second"):
                    sub = trials_whole[trials_whole["half"] == half_val]
                    half_res[half_val] = decode_condition(sub, unit_ids)
                crossgen = {"first": None, "second": None}
                if half_res["first"]["ok"] and half_res["second"]["ok"]:
                    crossgen["first"] = decode_curve_crossgen(
                        half_res["first"]["matrices"], half_res["first"]["y"],
                        half_res["second"]["matrices"], half_res["second"]["y"], half_res["first"]["C"],
                    ).tolist()
                    crossgen["second"] = decode_curve_crossgen(
                        half_res["second"]["matrices"], half_res["second"]["y"],
                        half_res["first"]["matrices"], half_res["first"]["y"], half_res["second"]["C"],
                    ).tolist()
                for half_val in ("first", "second"):
                    res = half_res[half_val]
                    if not res["ok"]:
                        rows.append(dict(area_row, condition_type="half", condition_value=half_val,
                                          n_trials=res["n_trials"], skipped_reason=res["reason"]))
                    else:
                        rows.append(dict(area_row, condition_type="half", condition_value=half_val, n_trials=res["n_trials"],
                                          C=res["C"], real_curve=res["real_curve"].tolist(), **null_fields(res),
                                          peak_acc=res["peak_acc"], crossgen_curve_to_other=crossgen[half_val], skipped_reason=None))

            # --- perf-states (with cross-gen) ---
            if trials_perfstate is not None and len(trials_perfstate) > 0 and _want(session_id, area_col, area_value, "perfstate"):
                state_res = {}
                for state_val in ("high", "low"):
                    sub = trials_perfstate[trials_perfstate["perf_state"] == state_val]
                    state_res[state_val] = decode_condition(sub, unit_ids)
                crossgen_state = {"high": None, "low": None}
                if state_res["high"]["ok"] and state_res["low"]["ok"]:
                    crossgen_state["high"] = decode_curve_crossgen(
                        state_res["high"]["matrices"], state_res["high"]["y"],
                        state_res["low"]["matrices"], state_res["low"]["y"], state_res["high"]["C"],
                    ).tolist()
                    crossgen_state["low"] = decode_curve_crossgen(
                        state_res["low"]["matrices"], state_res["low"]["y"],
                        state_res["high"]["matrices"], state_res["high"]["y"], state_res["low"]["C"],
                    ).tolist()
                for state_val in ("high", "low"):
                    res = state_res[state_val]
                    if not res["ok"]:
                        rows.append(dict(area_row, condition_type="perfstate", condition_value=state_val,
                                          n_trials=res["n_trials"], skipped_reason=res["reason"]))
                    else:
                        rows.append(dict(area_row, condition_type="perfstate", condition_value=state_val, n_trials=res["n_trials"],
                                          C=res["C"], real_curve=res["real_curve"].tolist(), **null_fields(res),
                                          peak_acc=res["peak_acc"], crossgen_curve_to_other=crossgen_state[state_val], skipped_reason=None))
    return rows


def load_done_sessions() -> set[str]:
    if not PARTIAL_PATH.exists():
        return set()
    return set(pd.read_parquet(PARTIAL_PATH, columns=["session_id"])["session_id"].unique())


def append_rows(rows: list[dict]):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if PARTIAL_PATH.exists():
        existing = pd.read_parquet(PARTIAL_PATH)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_parquet(PARTIAL_PATH, index=False)


def replace_rows(rows: list[dict]):
    """Fill-in mode: drop the existing rows of the recomputed (session, area, condition_type) groups, append new."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    key = ["session_id", "area_col", "area_value", "condition_type"]
    existing = pd.read_parquet(PARTIAL_PATH)
    keys = set(new_df[key].itertuples(index=False, name=None))
    keep = ~pd.Series(list(existing[key].itertuples(index=False, name=None))).isin(keys).to_numpy()
    pd.concat([existing[keep], new_df], ignore_index=True).to_parquet(PARTIAL_PATH, index=False)


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import causal_bin_edges, hitmiss_session_list

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")

    hitmiss_sessions = hitmiss_session_list(sessions_tbl)
    todo_sessions = hitmiss_sessions[hitmiss_sessions.day_stage == DAY_STAGE].reset_index(drop=True)
    print(f"[{TAG}] {DAY_STAGE}-stage sessions to process: {len(todo_sessions)}", flush=True)

    window = STIM_WINDOW if ALIGNMENT == "stim" else LICK_WINDOW
    bin_edges = causal_bin_edges(window, bin_width=BIN_WIDTH, stride=STRIDE)
    if FILL is None:
        BIN_EDGES_PATH.write_text(json.dumps(bin_edges))
    print(f"[{TAG}] bin grid: {len(bin_edges)} causal bins, window {window}, dead_zone {WIDE_DEAD_ZONE}", flush=True)

    done = load_done_sessions()
    print(f"[{TAG}] already-done sessions (resume): {len(done)}", flush=True)

    todo = todo_sessions[~todo_sessions.session_id.isin(done)]
    if FILL is not None:
        todo = todo_sessions[todo_sessions.session_id.isin({k[0] for k in FILL})]
        print(f"[{TAG}] FILL-IN mode: {len(FILL)} condition groups in {len(todo)} sessions", flush=True)
    print(f"[{TAG}] launching {len(todo)} sessions across {N_WORKERS} worker processes", flush=True)

    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    tasks = [
        (r.session_id, r.subject_id, r.reward_group, r.learning_category, scripts_dir, DECODE_TARGET, ALIGNMENT)
        for r in todo.itertuples()
    ]

    t_start = time.time()
    n_processed = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as pool:
        futures = {pool.submit(process_one_session, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            session_id = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{TAG}] ERROR {session_id}: {e!r}", flush=True)
                continue
            (replace_rows if FILL is not None else append_rows)(rows)
            n_processed += 1
            elapsed = time.time() - t_start
            remaining = len(tasks) - n_processed
            eta_min = (elapsed / n_processed) * remaining / 60.0 if n_processed else float("nan")
            n_computed_rows = sum(1 for r in rows if r.get("skipped_reason") is None)
            reward_group = next((t[2] for t in tasks if t[0] == session_id), "?")
            print(
                f"[{TAG}] [{n_processed}/{len(tasks)}] {session_id} ({reward_group}): "
                f"{n_computed_rows} computed rows, {len(rows) - n_computed_rows} skipped -- "
                f"elapsed {elapsed/60:.1f}min, ETA remaining {eta_min:.1f}min",
                flush=True,
            )

    print(f"[{TAG}] DONE", flush=True)


if __name__ == "__main__":
    main()
