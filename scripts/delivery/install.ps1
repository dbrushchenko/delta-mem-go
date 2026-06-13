# install.ps1 — Backup installer for δ-mem-go (when Go installer unavailable)
# Requires: delta-mem-go.exe already built in this directory
param(
    [string]$InstallDir = "$env:APPDATA\mem-go",
    [string]$ModelPath = "",
    [string]$Port = "18080",
    [string]$GrpcPort = "19090",
    [string]$Owner = $env:USERNAME
)

Write-Host "=== δ-mem-go Backup Installer ===" -ForegroundColor Cyan

# 1. Create dirs
New-Item -ItemType Directory -Path $InstallDir, "$InstallDir\data" -Force | Out-Null

# 2. Copy binary
$exe = Join-Path $PSScriptRoot "..\..\delta-mem-go.exe"
if (-not (Test-Path $exe)) { $exe = Read-Host "Path to delta-mem-go.exe" }
Copy-Item $exe "$InstallDir\delta-mem-go.exe" -Force
Write-Host "  ✓ Binary installed"

# 3. Model
if (-not $ModelPath) { $ModelPath = Read-Host "Path to nomic-embed-text-v1.5.onnx" }
Write-Host "  Model: $ModelPath"

# 4. Startup bat
$bat = @"
@echo off
start "" /B "$InstallDir\delta-mem-go.exe" --model "$ModelPath" --port $Port --grpc-port $GrpcPort --data "$InstallDir\data" > "$InstallDir\service.log" 2>&1
"@
$startupDir = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup"
Set-Content "$startupDir\delta-mem-go.bat" $bat
Write-Host "  ✓ Startup shortcut created"

# 5. Start now
Start-Process "$InstallDir\delta-mem-go.exe" -ArgumentList "--model","$ModelPath","--port",$Port,"--grpc-port",$GrpcPort,"--data","$InstallDir\data" -WindowStyle Hidden
Start-Sleep 2

# 6. Verify
try {
    $h = Invoke-RestMethod "http://localhost:$Port/health" -TimeoutSec 3
    Write-Host "  ✓ Running (uptime: $($h.uptime))" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Not responding — check $InstallDir\service.log" -ForegroundColor Red
}

# 7. Hook
Write-Host "`n  To wire kiro-cli hooks:"
Write-Host "    mem-cli.exe store --key K --content C  (gRPC, port $GrpcPort)"
Write-Host "    Or copy scripts\delivery\http-python\dmem-store.py to ~/.kiro/hooks/"
