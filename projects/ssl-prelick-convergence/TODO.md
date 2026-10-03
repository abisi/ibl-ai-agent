# TODO

Steps performed in `ssl-rastermap-psth-variants` (2026-09-29 → 2026-10-03) before the split; outputs in
`combined_results_ks4/_roc_prelick{,_sl}/` (see the README there).
- [X] 051–056 pre-lick ROC (WH/AH/ref), variants (all, short/long RT, RT-matched), min FR 0.1 Hz, stage × area
      summaries, PSTHs.
- [X] 057 λ and cross-validated distances; 058/059 convergence summary and single-neuron transfer; 060 λ_LDA.
- [X] 061 learners-only rule (bad mice lose only their day-0 session).
- [X] 063 decoder sweep; 068/069 single-session decoders with linear-shift null (both references).
- [X] 064 pseudo-population hierarchical bootstrap (mice → sessions → neurons → trials), shift null; 2026-10-03 rerun
      with transfer_bin and num_prob readouts (SL first; FA running).
- [X] 065–067 converging-unit locations, area attrition, CCF density maps; 070 recap; 071 SL timing control.
- [X] 062 publication figures (distance-based Fig 3, Fig 3S λ/λ_LDA, Fig 1h, Fig 2f/g, Fig 2S, Fig 5 pseudo-pop rows,
      COSYNE v2/v3), complete captions, article PDFs per variant (build_article.py + render.sh).
- [X] Split into this project; LOCKED.md (locked set, results, caveats, TODOs).

Planned
- [ ] Within-session remapping (072): early vs late halves / thirds of active trials, or pre- vs post-learning trial,
      on day 0 and expert sessions; Δd, λ, decoder numerator; trial-count matched halves; odd/even null; per-half shift
      null; RT per half (RT-matched with FA); test cohort × stage × half (mouse-level permutation / LMM).
- [ ] Increase pseudo-population repetitions for final runs.
- [ ] RT-matched single-session decoders and pseudo-populations (FA reference).
- [ ] Final COSYNE figure choice (v3) and abstract text.
