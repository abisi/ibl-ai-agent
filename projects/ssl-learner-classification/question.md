# ssl-learner-classification

## Question (user, 2026-11-13)
Who is a learner on whisker day 0? Using the exact learning curves, the learning-trial definitions
(ssl-learning-trial-identification) and the behaviour clustering (behaviour_analysis `behavior_clustering.py`), delineate
learners and non-learners without using the reference-sheet `learning_category`; decide what learners and non-learners
are with several metrics; compare with the sheet categories and with the clustering; compare performance between
learners and non-learners. Output: a report and a publication-quality supplementary figure ("what learners are").

## Constraint (user)
The R+ mice of the sheet's 'bad' category stay non-learners (MH020, MH013, AB142, MH068, AB151, AB117, AB107), except the
flexible ones (AB141, MH007, AB104, MH026, MH009), which take the agnostic label (as in ssl-within-day-two-axes 020).

## Approach
- Per mouse: features of the whisker day-0 session from the exact curves (grid HMM, sigma 1, 80 % band).
- Nine binary criteria per cohort (curve rules, rate tests, change points, mixture model, DTW clustering); consensus =
  fraction met; learner if >= 0.5; user constraint applied after.
- Comparisons: sheet categories (Cohen's kappa), the earlier DTW clusters and DTW on the exact curves (ARI / NMI).
- Validation: day +1 whisker P(lick) and d' (not used by any criterion), AUC per labelling.
- Robustness: leave one criterion out, thresholds 0.4 / 0.6; best single-feature rule (leave-one-out).
- Performance of learners vs non-learners per cohort (mice; Mann-Whitney and Welch).

## Scripts
- `exploratory-analyses/001_learner_classification.py` -- features, criteria, labels, comparisons, validation (haas).
- `exploratory-analyses/002_supp_figure.py` -- supplementary figure (Nature format).
- `report/build_report.py`, `report/render.sh` -- report.
