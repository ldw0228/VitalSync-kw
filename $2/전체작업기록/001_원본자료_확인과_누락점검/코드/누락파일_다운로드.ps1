param(
    [string]$ManifestPath = "$PSScriptRoot\missing_drive_files.json",
    [string]$DatasetRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) 'HAI_EXPERIMENT')
)

$ErrorActionPreference = 'Stop'
$items = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$datasetResolved = (Resolve-Path -LiteralPath $DatasetRoot).Path
$completed = 0
$skipped = 0
$failed = @()

foreach ($item in $items) {
    $completed++
    $recordingDir = Join-Path (Join-Path (Join-Path $datasetResolved $item.subject) ([string]$item.radar)) $item.recording
    if (-not (Test-Path -LiteralPath $recordingDir -PathType Container)) {
        $failed += "$($item.subject) radar$($item.radar): recording folder missing"
        continue
    }

    $destination = Join-Path $recordingDir $item.name
    $partial = "$destination.part"
    if (Test-Path -LiteralPath $destination) {
        $skipped++
        Write-Output "[$completed/$($items.Count)] SKIP $($item.subject) radar$($item.radar)"
        continue
    }

    try {
        if (-not (Test-Path -LiteralPath $partial)) {
            $url = "https://drive.google.com/uc?export=download&id=$($item.id)"
            Write-Output "[$completed/$($items.Count)] DOWNLOAD $($item.subject) radar$($item.radar)"
            & curl.exe -L --fail --retry 5 --retry-all-errors --connect-timeout 30 --max-time 180 --silent --show-error --output $partial $url
            if ($LASTEXITCODE -ne 0) { throw "curl exit code $LASTEXITCODE" }
        }

        $downloaded = Get-Item -LiteralPath $partial
        if ($downloaded.Length -lt 1MB) { throw "downloaded file is unexpectedly small ($($downloaded.Length) bytes)" }
        Move-Item -LiteralPath $partial -Destination $destination
        Write-Output "[$completed/$($items.Count)] OK $($item.subject) radar$($item.radar) $($downloaded.Length) bytes"
    }
    catch {
        $failed += "$($item.subject) radar$($item.radar): $($_.Exception.Message)"
        Write-Output "[$completed/$($items.Count)] FAILED $($item.subject) radar$($item.radar): $($_.Exception.Message)"
    }
}

Write-Output "SUMMARY total=$($items.Count) skipped=$skipped failed=$($failed.Count)"
if ($failed.Count -gt 0) {
    $failed | ForEach-Object { Write-Output "ERROR $_" }
    exit 1
}
