<#
One-time wrapper for the 2026-08-13 23:00 scheduled task ONLY. Rebuilds the
KS4 datasets via run_ssl_ks4_build.ps1, then — only if that build succeeds —
launches a fully unattended headless Claude Code run that continues this
repo's most recent conversation and executes the full SSL whisker/auditory
PSTH + linear-mixed-model + suppression-selectivity-modulation-index
analysis through to the final report, per the user's explicit "fully
unattended chain" choice made on 2026-08-13.

Do NOT point the 2026-08-16 recurring rebuild task at this script — the
user only asked for the analysis to run after tonight's rebuild, not the
one three days later.
#>

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$buildScript = Join-Path $repoRoot 'scripts\run_ssl_ks4_build.ps1'
$claudeExe = 'C:\Users\bisi\.local\bin\claude.exe'
$logDir = Join-Path $repoRoot 'reports\ssl_analysis_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$wrapperLog = Join-Path $logDir "wrapper_$stamp.log"
$analysisLog = Join-Path $logDir "ssl_analysis_run_$stamp.jsonl"

Start-Transcript -Path $wrapperLog -Append | Out-Null

Write-Host "=== Build+analyze wrapper starting: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $buildScript
$buildExitCode = $LASTEXITCODE
Write-Host "Build script exit code: $buildExitCode"

if ($buildExitCode -ne 0) {
    Write-Host "Build FAILED -- skipping analysis run. See reports\ssl_build_logs\ for the build log."
    Stop-Transcript | Out-Null
    exit 1
}

Write-Host "Build succeeded -- launching unattended analysis run. Log: $analysisLog"

$prompt = @'
The scheduled KS4 rebuild for tonight (2026-08-13 23:00) has finished successfully. Proceed now with the full SSL whisker/auditory PSTH + linear-mixed-model + suppression-selectivity-modulation-index analysis exactly as scoped and resolved earlier in this conversation, tracked as tasks #1-#12 (task #1, verifying the rebuild, is now satisfied by this message -- confirm it against reports/ssl_build_logs/ before proceeding). Work through every task fully autonomously, in order, through to the final formal report, without waiting for further confirmation: this is an unattended overnight run and no one will respond if you ask a question, so make and document reasonable judgment calls instead of stopping. Write the final report to a local markdown/HTML file under reports/ in this repo, in addition to attempting to publish it as an Artifact -- do not treat Artifact publishing as the only deliverable, in case it is unavailable in this headless context. Keep a running written note of any deviations, unexpected data issues, or judgment calls you had to make, at the top of the report, so it can be reviewed critically in the morning.
'@

& $claudeExe -p -c --dangerously-skip-permissions --effort high --max-budget-usd 75 --output-format stream-json --verbose --include-partial-messages $prompt *> $analysisLog
$analysisExitCode = $LASTEXITCODE
Write-Host "Analysis run exit code: $analysisExitCode"

Write-Host "=== Build+analyze wrapper finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Stop-Transcript | Out-Null
exit $analysisExitCode
