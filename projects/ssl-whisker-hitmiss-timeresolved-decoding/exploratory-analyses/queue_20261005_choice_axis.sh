#!/usr/bin/env bash
# 2026-10-05: choice-axis readout full run (146, stable + good units, whole brain + area groups) and reruns of 135 (lick-axis
# alignment) and 140 (coding direction / noise) on the corrected unit sets (skills/ssl-valid-data "Unit sets").
cd ~/code/ibl-ai-agent
P=projects/ssl-whisker-hitmiss-timeresolved-decoding; E=$P/exploratory-analyses; L=$E/queue_20261005_choice_axis.log
echo "[ca] start $(date)" >> $L
SSL_DECODE_N_WORKERS=30 .venv/bin/python $E/146_choice_axis_readout_epochs.py >> $L 2>&1; echo "[ca] 146 rc $? $(date)" >> $L
.venv/bin/python $E/147_choice_axis_readout_figure.py >> $L 2>&1; echo "[ca] 147 rc $? $(date)" >> $L
for U in good stable; do
  SSL_135_UNITSET=$U SSL_DECODE_N_WORKERS=30 .venv/bin/python $E/135_whisker_lick_alignment_epochs.py >> $L 2>&1; echo "[ca] 135 $U rc $? $(date)" >> $L
  SSL_135_UNITSET=$U .venv/bin/python $E/135_whisker_lick_alignment_epochs.py plot >> $L 2>&1
  (cd $E && SSL_135_UNITSET=$U ../../../.venv/bin/python 135b_alignment_figure.py >> ../../../$L 2>&1); echo "[ca] 135b $U rc $? $(date)" >> $L
done
mv $E/140_coding_direction_noise.parquet $E/140_coding_direction_noise_stable_v1.parquet
SSL_DECODE_N_WORKERS=30 .venv/bin/python $E/140_coding_direction_noise.py >> $L 2>&1; echo "[ca] 140 rc $? $(date)" >> $L
for s in all learners; do .venv/bin/python $E/145_coding_direction_figure.py $s >> $L 2>&1; done; echo "[ca] 145 rc $? $(date)" >> $L
bash $P/sync_results.sh >> $L 2>&1
echo "[ca] done $(date)" >> $L
