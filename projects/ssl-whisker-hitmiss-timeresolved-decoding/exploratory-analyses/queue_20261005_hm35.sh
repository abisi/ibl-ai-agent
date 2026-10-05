#!/usr/bin/env bash
# 2026-10-05 overnight: hit/miss 5-35 ms window for the learning-trial split (118) and every-whisker-trial placebo (122),
# then 126 per-session excess; results synced to the NAS results home.
cd ~/code/ibl-ai-agent
P=projects/ssl-whisker-hitmiss-timeresolved-decoding
E=$P/exploratory-analyses
L=$E/queue_20261005_hm35.log
echo "[hm35] start $(date)" >> $L
export SSL_HM_WINDOWS=5-35 SSL_DECODINGS=hitmiss SSL_DECODE_N_WORKERS=${NW:-40}
SSL_LT_OUT_TAG=_A1v2_hm35 .venv/bin/python $E/118_lt_definitions_split_decoding.py >> $L 2>&1; echo "[hm35] 118 rc $? $(date)" >> $L
SSL_PLACEBO_STEP=1 SSL_PLACEBO_TAG=_step1_hm35 .venv/bin/python $E/122_lt_definitions_placebo.py >> $L 2>&1; echo "[hm35] 122 rc $? $(date)" >> $L
(cd $E && SSL_PLACEBO_TAG=_step1_hm35 ../../../.venv/bin/python 126_lt_variants_modality.py >> ../../../$L 2>&1); echo "[hm35] 126 rc $? $(date)" >> $L
bash $P/sync_results.sh >> $L 2>&1
echo "[hm35] done $(date)" >> $L
