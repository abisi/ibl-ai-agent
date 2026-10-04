#!/usr/bin/env bash
# build.sh -- rebuild a project report from its own folder (copied into combined_results_ks4/<slug>/report/).
#   report.md + figures/  ->  report.tex (Pandoc)  ->  report.pdf (latexmk, XeLaTeX)  +  report.html (self-contained)
# Tools: Pandoc (standalone or bundled with Quarto) and latexmk / XeLaTeX (TinyTeX installed with `quarto install tinytex`).
set -u
cd "$(dirname "$0")"
if command -v pandoc >/dev/null 2>&1; then PANDOC=(pandoc)
elif command -v quarto >/dev/null 2>&1; then PANDOC=(quarto pandoc)
else PANDOC=("/c/Program Files/Quarto/bin/quarto.exe" pandoc); fi
LATEXMK=$(command -v latexmk || echo "${APPDATA:-$HOME/AppData/Roaming}/TinyTeX/bin/windows/latexmk.exe")

"${PANDOC[@]}" report.md -s -o report.tex --number-sections --toc --toc-depth=2 \
  -V documentclass=article -V fontsize=10pt -V geometry:margin=2cm -V mainfont=Arial -V mathfont="Cambria Math" \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  -V header-includes='\usepackage{caption}\captionsetup{labelformat=empty,font=small}\usepackage{float}\floatplacement{figure}{H}'
rm -f report.pdf
"$LATEXMK" -xelatex -interaction=nonstopmode -halt-on-error report.tex > build.log 2>&1
"$LATEXMK" -c report.tex > /dev/null 2>&1
if [ -f report.pdf ]; then echo "report.pdf $(du -h report.pdf | cut -f1)"; else echo "PDF build failed: see build.log"; tail -20 build.log; exit 1; fi
"${PANDOC[@]}" report.md -s -o report.html --embed-resources --mathml --number-sections --toc --toc-depth=2 \
  --metadata pagetitle="$(sed -n 's/^title: *//p' report.md | head -1 | tr -d '"')" \
  -c "data:text/css,body{max-width:60em;margin:auto;font-family:Arial,sans-serif;line-height:1.45}img{max-width:100%}figcaption{font-size:0.9em}"
echo "report.html $(du -h report.html | cut -f1)"
