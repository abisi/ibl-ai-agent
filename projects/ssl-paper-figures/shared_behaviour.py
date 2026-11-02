"""Shared input: trial table of all included mice, auditory days -2/-1 and whisker days 0/+1/+2, built with
behaviour_analysis from the NWB_ks4 files and cached in Figures/_cache/.

Selection: mice with exclude == 0 and reward_group R+ or R- in joint_mouse_reference_weight.xlsx (cohort per mouse from
the sheet). Trials: perf != 6, context label != 'passive', auditory warm-up block on whisker day 0 dropped except its
last trial (skills/ssl-trial-exclusion rules 1-3; rule 2 uses the context label only, not the per-trial ITI rule).
Reaction time is recomputed after reindex_whisker_days; whisker trials on auditory days are dropped.
Early-lick trials are kept with their `early_lick` flag; analyses drop them where behaviour_analysis does.
"""

import json
from multiprocessing import Pool

import pandas as pd

from common import CACHE_DIR, NWB_DIR, REFERENCE_SHEET, import_external_repos

PARAMS = {"nwb_dir": "NWB_ks4", "days": [-2, -1, 0, 1, 2], "behaviors": ["auditory", "whisker"], "version": 3}  # v3: canonical NAS behaviour_analysis
CACHE_FILE = CACHE_DIR / f"behaviour_table_v{PARAMS['version']}.parquet"
UNTRIMMED_DAY0_FILE = CACHE_DIR / f"behaviour_table_v{PARAMS['version']}_day0_untrimmed.parquet"
KEEP_COLUMNS = ["mouse_id", "session_id", "behavior", "day", "trial_id", "start_time", "trial_type", "perf",
                "lick_flag", "lick_time", "response_window_start_time", "computed_reaction_time", "early_lick",
                "context", "outcome_w", "outcome_a", "outcome_n", "opto_stim"]


def _one_file(path):
    import_external_repos()
    import behavior_analysis_utils as bhv_utils
    try:
        return bhv_utils.build_standard_behavior_table([str(path)])
    except Exception as err:  # unreadable or non-standard NWB: reported, skipped
        print("skip", path.name, type(err).__name__, err)
        return None


def mouse_reference():
    ref = pd.read_excel(REFERENCE_SHEET)
    ref = ref[(ref.exclude == 0) & ref.reward_group.isin(["R+", "R-"])]
    return ref[["mouse_id", "reward_group", "learning_category"]].rename(columns={"reward_group": "cohort"})


def drop_warmup_block(session):
    """Whisker day 0: drop trials before the first whisker trial except the last one."""
    session = session.sort_values("start_time")
    is_whisker = (session.trial_type == "whisker_trial").to_numpy()
    if session.day.iloc[0] != 0 or not is_whisker.any():
        return session
    first = int(is_whisker.argmax())
    return session.iloc[max(first - 1, 0):]


def build(n_workers=16):
    import_external_repos()
    import behavior_analysis_utils as bhv_utils
    ref = mouse_reference()
    files = sorted(f for f in NWB_DIR.glob("*.nwb") if f.name.split("_")[0] in set(ref.mouse_id))
    with Pool(n_workers) as pool:
        tables = [t for t in pool.map(_one_file, files) if t is not None]
    table = pd.concat(tables, ignore_index=True)
    table = bhv_utils.reindex_whisker_days(table)
    # reindex_whisker_days offsets every column containing "time" when it merges whisker_on_1/_2 sessions, including
    # computed_reaction_time (AB077-AB107 day +2): recompute it after the merge
    table["computed_reaction_time"] = table.lick_time - table.response_window_start_time
    table = table[table.behavior.isin(PARAMS["behaviors"]) & table.day.isin(PARAMS["days"])]
    # whisker trials only on whisker days (MH010 has 5 whisker trials on auditory day -2)
    table = table[~((table.behavior == "auditory") & (table.trial_type == "whisker_trial"))]
    table = table[(table.perf != 6) & (table.context.astype(str) != "passive")]
    untrimmed_day0 = _finish(table[(table.behavior == "whisker") & (table.day == 0)], ref)  # stored-curve trial set
    table = pd.concat([drop_warmup_block(s) for _, s in table.groupby("session_id")], ignore_index=True)
    return _finish(table, ref), untrimmed_day0, len(files)


def _finish(table, ref):
    table = table[KEEP_COLUMNS].merge(ref, on="mouse_id", how="inner")
    table["reward_group"] = table.cohort.map({"R+": 1, "R-": 0})  # coding used by behaviour_analysis
    return table.reset_index(drop=True)


def load(rebuild=False):
    if CACHE_FILE.exists() and not rebuild:
        return pd.read_parquet(CACHE_FILE)
    table, untrimmed_day0, n_files = build()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for df, path in ((untrimmed_day0, UNTRIMMED_DAY0_FILE), (table, CACHE_FILE)):
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.rename(path)
    json.dump({**PARAMS, "n_nwb_files": n_files, "n_mice": int(table.mouse_id.nunique()),
               "n_sessions": int(table.session_id.nunique())},
              open(CACHE_FILE.with_suffix(".json"), "w"), indent=1)
    return table


def load_untrimmed_day0():
    """Whisker day 0 without the warm-up cut (the trial set of the stored ks4 learning curves)."""
    if not UNTRIMMED_DAY0_FILE.exists():
        load(rebuild=True)
    return pd.read_parquet(UNTRIMMED_DAY0_FILE)
