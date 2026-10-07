# Question

**Original:** test that whisker hits become more like auditory hits, more so in R+ mice, not necessarily in all areas
but in some; with pre-lick ROCs (WH vs AH, WH vs ref, AH vs ref) and population analyses.

**Explicated:** in the 100 ms before the corrected first lick, does activity on whisker hits (lick after a whisker
stimulus) move toward activity on auditory hits (rewarded lick) and away from an unrewarded-lick reference, from the
first whisker-training day (learning, day 0) to later days (expert), in R+ (whisker licks rewarded) but not R− mice?

**Definitions**
- WH / AH: lick after a whisker / auditory stimulus (cohort-independent definition).
- Reference: spontaneous licks (SL; headline) or false alarms (FA; licks on no-stimulus catch trials).
- Reward-lick neuron: significant AH vs ref pre-lick ROC; converging neuron: reward-lick neuron also significant for
  WH vs ref with the same sign.
- Δd = d(WH, ref) − d(WH, AH), cross-validated squared Euclidean distances per unit (> 0: WH closer to AH).
- λ: cross-validated projection of WH on the ref → AH axis (0 = ref, 1 = AH); λ_LDA: same on a shrinkage-LDA axis.
- Decoder numerator: P(AH | WH) − P(AH | ref) of an AH-vs-ref decoder, minus its linear-shift chance level.
- Unit of analysis: session; interaction tested by permuting cohort labels across mice.

**Exploration / confirmation:** all analyses so far are exploratory (all mice and learners-only populations, two
references); no held-out confirmation set has been defined.

**Next question:** does remapping occur within the learning day (session halves / pre- vs post-learning trial), and does
it differ between cohorts (R− already on day 0 vs R+ over days)?


# Part II: within days (formerly ssl-within-day-remapping/question.md)


**Original (user, 2026-10-03):** test whether a similar remapping is observed within session, in particular across
session halves on the learning day, at the population-geometry and decoding level; hypothesis: R− mice remap already on
day 0, whereas R+ mice need more days; expert sessions should not show the within-session effect.

**Explicated:** within day-0 sessions, does WH pre-lick activity move away from AH and toward SL in R− mice, and toward
AH in R+ mice; is the within-day change absent in expert sessions; does the end of day 0 reach the expert level?

**Definitions:** as in ssl-prelick-convergence (WH, AH, SL; Δd = d(WH, SL) − d(WH, AH); λ; decoder numerator).
Halves: split at the midpoint between the two middle auditory hits. Trial score: projection on the session's
cross-validated SL → AH axis normalised to SL = 0, AH = 1. Relative slope: slope(WH) − slope(SL).

**Exploration / confirmation:** exploratory; SL reference only.
