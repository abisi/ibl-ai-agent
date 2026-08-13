## Purpose
Authoritative description of the SSL behavioral paradigm and terminology,
transcribed directly from Axel Bisi's own paper/thesis draft text (provided
verbatim 2026-08-13). This is the primary source for task design — where it
adds detail beyond or refines `ssl_task_semantics.md`'s dataset-column-level
description, this file is canonical; `ssl_task_semantics.md` should be read
alongside it for how these concepts map onto actual parquet columns.

## Paradigm: single-session whisker associative learning
After implantation and IOS imaging, mice are water restricted to induce
motivation and weighed before each behavioral session. Whiskers are trimmed
and a metallic particle is positioned on the **left C2 whisker**. White noise
plays throughout all sessions (masks environmental noise, signals task
context); levels are lowered only during each mouse's first two auditory
detection sessions. Water rewards are 5 microliters throughout. One session
per day. Three sequential stages, each corresponding to a `sessions.session_description`
prefix (see `ssl_task_semantics.md`'s Day/training-stage section):

| Stage | `session_description` prefix | Structure |
|---|---|---|
| 1. Free-licking | `free_licking_*` | No stimulus, no task — habituation to spout/setup, weight stabilization |
| 2. Auditory pretraining | `auditory_*` | Auditory Go/NoGo detection only |
| 3. Whisker learning | `whisker_*` | Auditory + whisker + no-stim trials interleaved |

## Timing parameters (constant across auditory pretraining and whisker learning)
| Parameter | Duration |
|---|---|
| Inter-trial interval | Uniform(6, 10) s |
| Quiet window (required no-lick before trial start) | Uniform(2, 5) s |
| Response window | 1 s |
| **Artifact window** | **100 ms** |
| Auditory stimulus | 10 ms, 10 kHz, bilateral, 74 dB |
| Whisker stimulus | 3 ms, 30 mT (calibrated per mouse at session start), right C2 whisker |

No behavioral shaping except the lowered white-noise levels noted above. No
punishment is enabled at any stage.

**Confirmed — whisker stimulation side (2026-08-13):** the **right** C2
whisker is the one stimulated; neural recordings are from the **left**
hemisphere, contralateral to the stimulated whisker (standard for whisker
sensory mapping). The surgical-prep sentence above, which places the metal
particle on the "left" C2 whisker, appears to be an error in the source
draft — treat "right C2 whisker" as the confirmed stimulus side for any
side-specific (contralateral/ipsilateral) interpretation.

**Note — artifact window is an acquisition-time parameter, not an analysis
window:** the 100 ms artifact window above is a behavioral-acquisition/
stimulus-delivery hardware parameter, not a spike-analysis exclusion
recommendation. It has no direct bearing on choosing a post-hoc firing-rate
dead-zone (e.g. the unrelated 10 ms dead-zone used in
`ssl-passive-sensory-selectivity`, see `ssl_analysis_patterns.md`'s
passive-selectivity section) — the two serve different purposes and are not
meant to be reconciled.

## Stage 2: Auditory pretraining
Mice lick to report detection of the auditory tone for a reward. 50% of
trials are auditory-stimulus trials, 50% are "no stimulus" trials (uncued,
randomly ordered). Response window is 1 s from stimulus onset. Auditory
trials end at the lick (or response-window timeout); no-stimulus trials
always end 5 s after trial start regardless of licking.

**Expert criterion** (when mice move on to whisker learning): auditory hit
rate ≥ 80% **and** false-alarm rate < 40%. Some mice continue an extra
auditory-pretraining day past reaching criterion purely for scheduling
reasons (cohort throughput for same-day whisker learning + ephys), not a
performance reason — do not treat an extra auditory day as evidence of worse
learning for those mice.

## Stage 3: Whisker learning
Same task structure, with a third trial type added: whisker stimulation (3 ms,
30 mT). Auditory and whisker stimuli never co-occur on the same trial.

**Trial-type proportions** — two observed eras, check which applies to the
session(s) in scope before assuming one:
- Most mice: 50% whisker / 10% auditory / 40% no-stimulus.
- Some earlier mice: 35% whisker / 15% auditory / 50% no-stimulus.

**Reward groups**, assigned randomly at the start of whisker learning:
- **R+ ("whisker-rewarded")**: reward follows a lick after whisker
  stimulation — mice must *learn to lick* for whisker. Auditory trials remain
  a standard Go task in both groups.
- **R- ("whisker non-rewarded")**: no reward follows a lick after whisker
  stimulation — mice must *learn to withhold* licking for whisker; whisker
  trials are effectively a Go/No-Go task for this group specifically (the
  task is **not** Go/No-Go for auditory trials, and **not** Go/No-Go for R+
  whisker trials — only R- whisker trials are the No-Go side).
Mice learn this new whisker contingency while continuing to perform the
already-learned auditory detection task in the background, on interleaved trials.

Whisker learning runs for **3 days** nominally, after which mice are
euthanized/perfused. Cross-check against `sessions.session_description` when
a specific dataset shows `day` values beyond +2 (observed in some opto
variants, e.g. `whisker_on_N_opto`/`whisker_off_N_opto`) — those extend past
the nominal 3-day window and are a distinct protocol variant, not evidence
the base paradigm ran longer.

### Trial-type / outcome terminology
| Trial type | Lick | No lick |
|---|---|---|
| Auditory | Hit | Miss |
| Whisker | Hit | Miss |
| No stimulus | False alarm | Correct rejection |

Hit/miss/false-alarm/correct-rejection labels themselves are defined the same
way for both reward groups (based only on lick vs. no-lick) — what differs by
group is which of these labels counts as the **rewarded/correct** outcome:

| Trial outcome | R+ mice | R- mice |
|---|---|---|
| Auditory hit | Correct (rewarded) | Correct (rewarded) |
| Auditory miss | Incorrect | Incorrect |
| Whisker hit | **Correct (rewarded)** | **Incorrect** |
| Whisker miss | **Incorrect** | **Correct (rewarded, no water)** |
| False alarm | Incorrect | Incorrect |
| Correct rejection | Correct (no water) | Correct (no water) |

This is the authoritative source for the cohort-dependent correction already
described in `ssl_task_semantics.md`'s "Cohort-dependent outcome" section —
that section's whisker R+/R- flip matches this table exactly. Note that
"correct" and "rewarded" are not synonyms here: correct rejection and R-
whisker miss are both "correct" (the right choice) but never deliver water —
only auditory hit and (R+) whisker hit ever trigger a reward.

## Passive stimulation
Recorded on **every day neural activity is recorded** (not just `day==0` —
corrects an earlier draft of this text), for subjects **AB116 and above
only**; subjects with numeric index below 116 (e.g. AB080) never have passive
blocks, on any day (confirmed 2026-08-13). 35 auditory and 35 whisker
stimuli, randomly interleaved, one every 3 s, delivered at the **beginning
and end** of the behavioral session (i.e. `passive_pre` and `passive_post`)
to capture sensory-evoked responses before/after that day's learning. The
lick spout is retracted throughout — no rewards are possible during passive
blocks, so `lick_flag`/outcome columns are not meaningful for passive trials.

This is the confirmed explanation for
`../../ssl-load/references/ssl_dataset_schema.md`'s observation that only
subjects from AB116 onward have `passive_pre`/`passive_post` epochs recorded:
it is a deterministic per-subject rule (numeric index >= 116), not a
per-session or day-dependent one. Every ephys day for an AB116+ subject
should have passive epochs; their absence for a specific AB116+ session is a
data-quality flag worth checking, not expected design behavior.
