# ==============================================================================
# Hadoop Streaming Job Runner for Windows / WSL
# Project: US Flight Delay Prediction CA3 Big Data Analytics
# ==============================================================================

param(
    [string]$GroupKey = "AIRLINE",
    [string]$InputCsv = "data\flights_sample.csv"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " US Flight Delay Analytics — Hadoop Streaming Job (Windows)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Grouping Key: $GroupKey"
Write-Host "Input CSV:    $InputCsv"

# Check if WSL is available
$wslCheck = Get-Command wsl -ErrorAction SilentlyContinue

if ($wslCheck) {
    Write-Host "`n[WSL Detected] Submitting job to Ubuntu via WSL2..." -ForegroundColor Green
    $linuxPath = $InputCsv -replace "\\", "/"
    $scriptLinux = "./hadoop_streaming/run_hadoop_job.sh"
    
    # Run inside WSL
    wsl bash -c "chmod +x $scriptLinux && $scriptLinux '$GroupKey' '$linuxPath'"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nHadoop Streaming job executed via WSL successfully!" -ForegroundColor Green
    } else {
        Write-Host "`nHadoop execution failed or Hadoop is not running in WSL." -ForegroundColor Yellow
        Write-Host "To run in Local Simulation mode, open the Streamlit dashboard." -ForegroundColor Gray
    }
} else {
    Write-Host "`nWSL is not enabled on this Windows machine." -ForegroundColor Yellow
    Write-Host "You can execute the MapReduce local simulator directly in the dashboard" -ForegroundColor Cyan
    Write-Host "or install WSL2 with: wsl --install" -ForegroundColor Gray
}
