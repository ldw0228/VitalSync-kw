$ErrorActionPreference = 'Continue'

$python = 'python'
$syncScript = Join-Path $PSScriptRoot 'sync_subject.py'
$datasetRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'HAI_EXPERIMENT'
$logPath = Join-Path $PSScriptRoot 'sync_results\batch_reverse_log.txt'

$lines = @()
foreach ($number in 28..1) {
    $prefix = ('S{0:D2}_' -f $number)
    $subject = Get-ChildItem -LiteralPath $datasetRoot -Directory |
        Where-Object { $_.Name.StartsWith($prefix) } |
        Select-Object -First 1

    if (-not $subject) {
        $line = ('S{0:D2}: SUBJECT_FOLDER_MISSING' -f $number)
        Write-Output $line
        $lines += $line
        continue
    }

    Write-Output ("분석 중: {0}" -f $subject.Name)
    $output = & $python $syncScript $subject.Name 2>&1
    if ($LASTEXITCODE -eq 0) {
        $resultPath = Join-Path $PSScriptRoot ("sync_results\{0}\sync_result.json" -f $subject.Name)
        $result = Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json
        $line = ("{0}: OK markers={1} offset={2:+0.0;-0.0;0.0}s duration={3:0.0}s" -f $subject.Name, $result.marker_count, $result.offset_s, $result.radar_duration_s)
    } else {
        $lastOutput = ($output | Select-Object -Last 1)
        $line = ("{0}: FAILED {1}" -f $subject.Name, $lastOutput)
    }
    Write-Output $line
    $lines += $line
}

$lines | Set-Content -LiteralPath $logPath -Encoding utf8
Write-Output ("로그: {0}" -f $logPath)
