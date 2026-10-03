# render.sh <ref> <pop>: fetch figures + article qmd (written by build_article.py on haas), render locally with Quarto (typst), push the PDF back
A="${RENDER_DIR:-$PWD/_render}"      # local working directory for the figures and the rendered PDF
ref=$1; pop=$2; d=/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/_roc_prelick; [ $ref = sl ] && d=${d}_sl
L="$A/${ref}_${pop}"; mkdir -p "$L"
for f in Fig1_task_data Fig2_single_neurons Fig2S_converging_neurons Fig3_population_distance Fig3S_lambda_lambdaLDA Fig4_areas Fig5_decoders; do
  scp -q haas:$d/publication/$pop/$f.png "$L/"; done
scp -q haas:$d/publication/$pop/prelick_convergence_${ref}_${pop}.qmd "$L/"
(cd "$L" && quarto render prelick_convergence_${ref}_${pop}.qmd --to typst 2>&1 | tail -1)
scp -q "$L/prelick_convergence_${ref}_${pop}.pdf" haas:$d/publication/$pop/
ls -la "$L"/*.pdf
