# ssl-learning-axis-exploration

## Question
Beyond the within-session drift of whisker hits on the reward-lick coding direction (ssl-within-day-two-axes), which properties of
pre-lick and peri-stimulus population activity distinguish the learning session (day 0) from expert sessions, and R+ from R- mice?
Exploratory probes of the reward-lick CD and new coding-direction definitions designed to target learning.

## Analyses
- 001 separability: AUC of whisker hits vs auditory hits and vs spontaneous licks on the reward-lick CD; spread of whisker hits.
- 002 lick-aligned trajectories on the reward-lick CD (-400 to +100 ms): lambda(t), divergence onset.
- 003 trial history (previous reward, time since reward, previous whisker-trial lick) and post-lick licking (licks in 1 s, bout duration)
  vs the whisker-hit score.
- 004 new axes: reward expectation (auditory hits vs correct rejections, 50-150 ms, fast sensory response removed), outcome (rewarded vs
  unrewarded lick, +25 to +125 ms), alignment of the hit - miss axis with the identity and expectation axes, the reward-lick CD with the
  motor and sensory axes projected out, and a generic lick-preparation axis.

## Data
Inputs: ssl-within-day-two-axes tables/008_events.parquet (per-event CD scores, both baseline variants), ssl-prelick-convergence 105
per-event rate matrices, NWB trials and piezo lick times. Learners rule for day 0 (definition under review), all expert sessions; session
= unit. Moved here from ssl-within-day-two-axes (scripts 020-023) on 2026-10-16.

## Results
combined_results_ks4/ssl-learning-axis-exploration/{tables,figures}. No report (user 2026-10-16: not added to the within-day report).
