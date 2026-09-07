# Samanvaya — Windows PowerShell Launcher
# Usage: .\start.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "  🌙 Samanvaya (समान्वय) — Web Portal Launcher" -ForegroundColor Cyan
Write-Host "  ═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  🚀  Streamlit Portal → http://localhost:8501" -ForegroundColor Green
Write-Host ""

$py = "python"
if (Test-Path "$root\.venv\Scripts\python.exe") {
  $py = "$root\.venv\Scripts\python.exe"
} elseif (Test-Path "$root\venv\Scripts\python.exe") {
  $py = "$root\venv\Scripts\python.exe"
}

Write-Host "Starting Samanvaya Streamlit Portal..." -ForegroundColor Green
& $py -m streamlit run "$root\app.py" --server.port 8501
