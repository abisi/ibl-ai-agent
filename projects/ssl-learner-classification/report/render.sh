#!/usr/bin/env bash
# Render the report built by build_report.py (run that on haas first). Locally (Git Bash, Quarto + TinyTeX):
#   bash projects/ssl-learner-classification/report/render.sh
# latexmk may exit non-zero even when the PDF is written: build.sh checks that report.pdf exists.
set -uo pipefail
SLUG=ssl-learner-classification
for root in "/m/analysis/Axel_Bisi" "/mnt/lsens-analysis/Axel_Bisi"; do
  if [ -d "$root/combined_results_ks4/$SLUG/report" ]; then R="$root/combined_results_ks4/$SLUG/report"; break; fi
done
TT="$(cygpath -u "${APPDATA:-}" 2>/dev/null)/TinyTeX/bin/windows"
[ -d "$TT" ] && export PATH="$TT:$PATH"
bash "$R/build.sh"
