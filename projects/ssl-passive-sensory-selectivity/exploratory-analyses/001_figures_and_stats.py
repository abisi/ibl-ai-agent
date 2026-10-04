"""Time-course figures, cohort x modality interaction, selectivity index, and
mixed-effects statistics on the passive before/after-task response summary --
MOUSE-LEVEL VERSION (2026-08-06): every figure and the GLMM are computed by
first averaging within each mouse (collapsing across that mouse's units),
and only then comparing/plotting/modeling across mice. This replaces the
earlier unit-level version, which pooled neurons as if they were independent
replicates (pseudoreplication) -- see ../question.md for the full rationale.

Must be run with unit_spikes_analysis's own venv python (for statsmodels
already pinned there). Input: ../artifacts/<SUMMARY_FILE>, produced by
000_extract_passive_responses.py with the 5-30ms window. Run separately per
day_stage (learning/expert).
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
SUMMARY_FILE = "passive_unit_summary.parquet"
WINDOW = "w5_30"
MODALITIES = ["whisker", "auditory"]

df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, SUMMARY_FILE))
with open(os.path.join(ARTIFACTS_DIR, "time_course_bin_centers_s.json")) as f:
    bin_centers = np.array(json.load(f))

print(f"Loaded {len(df)} units. day_stage: {df['day_stage'].value_counts().to_dict()}")
print(f"reward_group: {df['reward_group'].value_counts(dropna=False).to_dict()}")
print(f"area_group: {df['area_group'].value_counts(dropna=False).to_dict()}")


def mouse_mean_scalar(sub_df, value_col, group_cols):
    """Mean of a scalar column within each group (e.g. mouse x modality x area)."""
    return sub_df.groupby(group_cols, as_index=False)[value_col].mean()


def mouse_mean_arrays(sub_df, array_col, group_cols):
    """Element-wise mean of a list/array column within each group. Returns a
    DataFrame with group_cols plus a `mean_array` column (numpy array)."""
    rows = []
    for keys, g in sub_df.groupby(group_cols):
        arrs = [np.asarray(x, dtype=float) for x in g[array_col] if x is not None]
        arrs = [a for a in arrs if not np.all(np.isnan(a))]
        if not arrs:
            continue
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(group_cols, keys), mean_array=np.nanmean(np.vstack(arrs), axis=0)))
    return pd.DataFrame(rows)


for day_stage in sorted(df["day_stage"].dropna().unique()):
    stage_df = df[df["day_stage"] == day_stage].copy()
    if stage_df.empty:
        continue
    n_mice = stage_df["mouse_id"].nunique()
    print(f"\n=== day_stage={day_stage}: {len(stage_df)} units, {n_mice} mice ===")

    # ---------------- Time course figure (mouse-level, then grand mean across mice) ----------------
    area_groups = stage_df["area_group"].value_counts().head(4).index.tolist()
    fig, axes = plt.subplots(len(area_groups), len(MODALITIES), figsize=(10, 2.6 * len(area_groups)), sharex=True, sharey=True)
    if len(area_groups) == 1:
        axes = axes.reshape(1, -1)
    for ai, area in enumerate(area_groups):
        for mi, modality in enumerate(MODALITIES):
            ax = axes[ai, mi]
            sub = stage_df[stage_df["area_group"] == area]
            for cohort, color in zip(sorted(sub["reward_group"].dropna().unique()), ["tab:blue", "tab:red"]):
                csub = sub[sub["reward_group"] == cohort]
                pre_by_mouse = mouse_mean_arrays(csub, f"tc_pre_{modality}", ["mouse_id"])
                post_by_mouse = mouse_mean_arrays(csub, f"tc_post_{modality}", ["mouse_id"])
                merged = pre_by_mouse.merge(post_by_mouse, on="mouse_id", suffixes=("_pre", "_post"))
                if merged.empty:
                    continue
                delta = np.vstack((merged["mean_array_post"] - merged["mean_array_pre"]).to_numpy())
                mean_delta = np.nanmean(delta, axis=0)
                sem_delta = np.nanstd(delta, axis=0) / np.sqrt(delta.shape[0])
                ax.plot(bin_centers, mean_delta, color=color, lw=1.5, label=f"cohort {cohort} (n={delta.shape[0]} mice)")
                ax.fill_between(bin_centers, mean_delta - sem_delta, mean_delta + sem_delta, color=color, alpha=0.2)
            ax.axvline(0, color="k", lw=0.6, ls="--")
            ax.axhline(0, color="k", lw=0.5, ls=":")
            if ai == 0:
                ax.set_title(modality, fontsize=9)
            if mi == 0:
                ax.set_ylabel(f"{area}\nDelta rate (post-pre task, Hz)", fontsize=7)
            if ai == 0 and mi == 0:
                ax.legend(fontsize=6)
    axes[-1, 0].set_xlabel("time from stim onset (s)")
    axes[-1, 1].set_xlabel("time from stim onset (s)")
    fig.suptitle(f"Time course of passive response change, mouse-level mean +/- SEM across mice, day_stage={day_stage}", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), f"002_time_course_mouselevel_{day_stage}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved 002_time_course_mouselevel_{day_stage}.png")

    # ---------------- Mouse x modality diff table (pooled across areas), reused below ----------------
    unit_long = pd.concat([
        stage_df[["unit_key", "mouse_id", "reward_group", "n_whisker_active", "n_auditory_active", "area_group", f"diff_{modality}_{WINDOW}"]]
        .rename(columns={f"diff_{modality}_{WINDOW}": "diff"})
        .assign(modality=modality)
        for modality in MODALITIES
    ], ignore_index=True).dropna(subset=["diff"])

    mouse_modality_diff = mouse_mean_scalar(unit_long, "diff", ["mouse_id", "reward_group", "modality", "n_whisker_active", "n_auditory_active"])

    # ---------------- Cohort x modality interaction plot (mouse-level) ----------------
    cell_means = mouse_modality_diff.groupby(["reward_group", "modality"])["diff"].agg(["mean", "sem", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    for cohort, color in zip(sorted(cell_means["reward_group"].dropna().unique()), ["tab:blue", "tab:red"]):
        csub = cell_means[cell_means["reward_group"] == cohort].set_index("modality").reindex(MODALITIES)
        n_this_cohort = mouse_modality_diff.loc[mouse_modality_diff["reward_group"] == cohort, "mouse_id"].nunique()
        ax.errorbar(MODALITIES, csub["mean"], yerr=csub["sem"], marker="o", color=color, label=f"cohort {cohort} (n={n_this_cohort} mice)", capsize=3)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel(f"mean diff (post-pre task), window={WINDOW}\n(mean across mice +/- SEM across mice)")
    ax.set_title(f"Cohort x modality interaction (mouse-level), day_stage={day_stage}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), f"002_interaction_mouselevel_{day_stage}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved 002_interaction_mouselevel_{day_stage}.png")

    # ---------------- Selectivity index by area group (mouse-level) ----------------
    mouse_area_diff = mouse_mean_scalar(unit_long, "diff", ["mouse_id", "reward_group", "modality", "area_group"])
    wide = mouse_area_diff.pivot_table(index=["mouse_id", "reward_group", "area_group"], columns="modality", values="diff").reset_index()
    wide = wide.dropna(subset=MODALITIES)
    denom = wide["whisker"] + wide["auditory"]
    wide["selectivity_index"] = np.where(denom.abs() > 1e-9, (wide["whisker"] - wide["auditory"]) / denom, np.nan)
    idx_df = wide.dropna(subset=["selectivity_index"])
    if len(idx_df):
        area_order = idx_df["area_group"].value_counts().index.tolist()
        fig, ax = plt.subplots(figsize=(8, 4))
        data = [idx_df.loc[idx_df["area_group"] == a, "selectivity_index"].clip(-3, 3) for a in area_order]
        counts = [len(d) for d in data]
        ax.boxplot(data, tick_labels=[f"{a}\n(n={c} mice)" for a, c in zip(area_order, counts)], showfliers=False)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_ylabel(f"selectivity index ({WINDOW}, mouse-level)\n(whisker_diff - auditory_diff) / sum, clipped to [-3,3]")
        ax.set_title(f"Selectivity index by area group (mouse-level), day_stage={day_stage}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(__file__), f"002_selectivity_mouselevel_{day_stage}.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved 002_selectivity_mouselevel_{day_stage}.png")

    # ---------------- Mixed-effects model (mouse-level: one row per mouse x modality) ----------------
    model_df = mouse_modality_diff.dropna(subset=["diff", "reward_group", "n_whisker_active", "n_auditory_active"]).copy()
    model_df["reward_group"] = model_df["reward_group"].astype(str)
    n_mice_model = model_df["mouse_id"].nunique()
    print(f"  GLMM input: {len(model_df)} rows (mouse x modality), {n_mice_model} mice")
    if n_mice_model < 4:
        print(f"  SKIP GLMM {day_stage}: only {n_mice_model} mice available")
        continue
    try:
        md = smf.mixedlm(
            "diff ~ modality * reward_group + n_whisker_active + n_auditory_active",
            data=model_df,
            groups=model_df["mouse_id"],
        )
        result = md.fit()
        summary_path = os.path.join(ARTIFACTS_DIR, f"glmm_mouselevel_{day_stage}_{WINDOW}.txt")
        with open(summary_path, "w") as f:
            f.write(str(result.summary()))
        print(f"  saved GLMM summary: {summary_path}")
        print(result.summary())
    except Exception as exc:
        print(f"  GLMM FAILED for {day_stage}: {exc}")

print("\nDONE")
