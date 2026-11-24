#!/bin/bash
# Full run on Kuma (BM_SET=full): data/full_inputs (014), data/full_sessions.csv, data/full_train_sessions.txt.
# kpms: PCA + fit on the 16 training mice (chunks of 60 s every 360 s), then applied to all sessions in 6 slices per view;
# AR-HMM: fit on the training mice + Viterbi on all sessions (one job per view); lightning-action: CSV prep, 5 folds
# grouped by mouse (out-of-mouse predictions for every session) and a final model on all data (side and top).
# Partitions l40s / mig24gb only (run.sbatch), never h100.
cd /work/lsens/bisi/behaviour_motifs
export BM_SET=full
sb(){ sbatch --parsable --export=ALL,BM_SET=full "$@" 2>/dev/null | tail -1; }
NP=6
for v in side top; do
  if [ $v = side ]; then KA=1e9; KF=1e8; else KA=1e11; KF=1e10; fi
  p=$(sb -J f_pca_$v scripts/run.sbatch venv_kpms 003_kpms.py $v pca norm=rest)
  f=$(sb -d afterok:$p -J f_fit_$v scripts/run.sbatch venv_kpms 003_kpms.py $v full norm=rest kappa_ar=$KA kappa_full=$KF ns=25 seed=0 iters=150 train=ho0)
  TAG=$(python3 -c "print(f'ns25_ka{float(\"$KA\"):.0e}_kf{float(\"$KF\"):.0e}_s0_ho0')")
  for i in $(seq 0 $((NP - 1))); do
    sb -d afterok:$f -J f_apply_${v}_$i scripts/run.sbatch venv_kpms 003_kpms.py $v apply norm=rest ns=25 train=ho0 tag=$TAG part=$i/$NP >/dev/null
  done
  sb -J f_arhmm_$v scripts/run.sbatch venv_kpms 005_arhmm_features.py $v full kappa=1e10 ns=15 seed=0 iters=150 train=ho0 >/dev/null
  q=$(sb -J f_la_prep_$v scripts/run.sbatch venv_la 004_lightning_action.py $v 20 prep)
  if [ $v = side ]; then FOLDS="0 1 2 3 4 final"; else FOLDS="final"; fi  # top: final model only (job count; analyses use side)
  for i in $FOLDS; do sb -d afterok:$q -J f_la_${v}_$i scripts/run.sbatch venv_la 004_lightning_action.py $v 20 $i >/dev/null; done
done
squeue -u bisi -h -o "%j %T" | grep "^f_" | sort | uniq -c
