#!/usr/bin/env bash
# Render the report built by build_report.py (run that on haas first). Locally (Git Bash, Quarto + TinyTeX):
#   bash projects/ssl-prelick-convergence/report/render.sh
set -euo pipefail
SLUG=ssl-prelick-convergence
for root in "/m/analysis/Axel_Bisi" "/mnt/lsens-analysis/Axel_Bisi"; do
  if [ -d "$root/combined_results_ks4/$SLUG/report" ]; then R="$root/combined_results_ks4/$SLUG/report"; break; fi
done
TT="$(cygpath -u "${APPDATA:-}" 2>/dev/null)/TinyTeX/bin/windows"
[ -d "$TT" ] && export PATH="$TT:$PATH"
bash "$R/build.sh"
