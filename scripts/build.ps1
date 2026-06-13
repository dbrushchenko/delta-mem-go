# build.ps1 — Build δ-mem-go + Installer
# Run from repo root: .\scripts\build.ps1
Push-Location $PSScriptRoot\..
$ErrorActionPreference = "Stop"

$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
$env:CGO_ENABLED = "1"

Write-Host "=== δ-mem-go Build ===" -ForegroundColor Cyan

# 1. Build the main binary
Write-Host "[1/3] Building delta-mem-go.exe..."
go build -o delta-mem-go.exe ./cmd/delta-mem-go
if ($LASTEXITCODE -ne 0) { throw "Build failed" }
Write-Host "  ✓ delta-mem-go.exe built"

# 2. Stage for installer embedding
Write-Host "[2/3] Staging for installer..."
Copy-Item -Force "delta-mem-go.exe" "installer\delta-mem-go.exe"

# 3. Build installer
Write-Host "[3/3] Building delta-mem-go-setup.exe..."
go build -o delta-mem-go-setup.exe ./installer/
if ($LASTEXITCODE -ne 0) { throw "Installer build failed" }
Write-Host "  ✓ delta-mem-go-setup.exe built"

# Summary
$mainSize = (Get-Item delta-mem-go.exe).Length / 1MB
$setupSize = (Get-Item delta-mem-go-setup.exe).Length / 1MB
Write-Host "`n=== Build Complete ===" -ForegroundColor Green
Write-Host "  delta-mem-go.exe:       $([math]::Round($mainSize,1)) MB"
Write-Host "  delta-mem-go-setup.exe: $([math]::Round($setupSize,1)) MB"
Write-Host "`n  To install: Run delta-mem-go-setup.exe as Administrator"

Pop-Location
