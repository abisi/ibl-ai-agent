<#
Rebuilds the ssl_ephys and ssl_behavior (KS4) datasets from the raw NWB
source on M:\analysis\Axel_Bisi\NWB_ks4. Invoked unattended by Windows Task
Scheduler; see the two one-time tasks created for the 2026-08-13 and
2026-08-16 11pm rebuilds.

Each dataset build refuses to write if its output dir already exists, so an
existing output dir is renamed aside before the build runs and only deleted
once the new build has succeeded — a failed build leaves the previous good
dataset in place instead of a half-built or missing one.
#>

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$exe = Join-Path $repoRoot '.venv\Scripts\ibl-ai-agent.exe'
$logDir = Join-Path $repoRoot 'reports\ssl_build_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir "ssl_ks4_build_$stamp.log"

Start-Transcript -Path $logFile -Append | Out-Null

$script:allSucceeded = $true

function Invoke-DatasetBuild {
    param(
        [string]$Command,
        [string]$DatasetName
    )

    $targetDir = Join-Path $repoRoot "reports\datasets\$DatasetName\1.0.0"
    $backupDir = $null
    if (Test-Path $targetDir) {
        $backupDir = "$targetDir.bak-$stamp"
        Write-Host "Backing up existing $DatasetName output: $targetDir -> $backupDir"
        Move-Item -Path $targetDir -Destination $backupDir
    }

    Write-Host "Running: $exe $Command"
    & $exe $Command
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0 -and (Test-Path $targetDir)) {
        Write-Host "$DatasetName build succeeded."
        if ($backupDir) {
            Remove-Item -Recurse -Force $backupDir
            Write-Host "Removed backup $backupDir"
        }
    } else {
        Write-Host "$DatasetName build FAILED (exit code $exitCode)."
        $script:allSucceeded = $false
        if ($backupDir) {
            if (Test-Path $targetDir) { Remove-Item -Recurse -Force $targetDir }
            Move-Item -Path $backupDir -Destination $targetDir
            Write-Host "Restored previous $DatasetName output from backup."
        }
    }
}

Write-Host "=== SSL KS4 build run starting: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

if (-not (Test-Path 'M:\analysis\Axel_Bisi\NWB_ks4')) {
    Write-Host "ERROR: M:\analysis\Axel_Bisi\NWB_ks4 not reachable (drive not mapped, or no user logged on). Aborting."
    Stop-Transcript | Out-Null
    exit 1
}

Invoke-DatasetBuild -Command 'build-ssl-ephys-dataset' -DatasetName 'ssl_ephys'
Invoke-DatasetBuild -Command 'build-ssl-behavior-dataset' -DatasetName 'ssl_behavior'

Write-Host "=== SSL KS4 build run finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Stop-Transcript | Out-Null

if (-not $script:allSucceeded) { exit 1 }
exit 0
