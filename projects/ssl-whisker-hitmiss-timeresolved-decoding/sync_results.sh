#!/usr/bin/env bash
# Results live in combined_results_ks4/<slug>/{tables,cache,figures} (skills/project-organization, moved 2026-10-04); the
# scripts still write next to themselves, through symlinks. A NEW output file, or one written atomically (tmp + rename,
# which replaces the symlink), lands as a regular file in exploratory-analyses/: this moves every such file to the results
# folder (replacing the older copy there) and links it back. Run on haas after each job:
#   bash projects/ssl-whisker-hitmiss-timeresolved-decoding/sync_results.sh
set -euo pipefail
SLUG=ssl-whisker-hitmiss-timeresolved-decoding
EA="$(cd "$(dirname "$0")" && pwd)/exploratory-analyses"
N=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/$SLUG
cat_of() { case "$1" in
  *.csv|*.parquet|*.json|*.md) echo tables;; *.txt|*.log) echo cache/logs;; *.png|*.pdf|*.svg) echo figures/_ea_root;; *) echo cache;; esac; }
n=0
while read -r f; do
  t="$N/$(cat_of "$f")/$f"; mkdir -p "$(dirname "$t")"
  mv -f "$EA/$f" "$t" 2>/dev/null || { cp "$EA/$f" "$t" && rm "$EA/$f"; }
  ln -s "$t" "$EA/$f"; n=$((n + 1))
done < <(find "$EA" -maxdepth 1 -type f ! -name '*.py' ! -name '*.sh' ! -name '_run_*' -printf '%f\n')
echo "synced $n file(s) to $N"
