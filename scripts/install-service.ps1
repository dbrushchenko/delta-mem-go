# install-service.ps1 — Register mem-go as NSSM Windows service
# Run as Administrator

$serviceName = "DeltaMemGo"
$installDir = "$env:APPDATA\mem-go"
$exePath = "$installDir\delta-mem-go.exe"
$modelPath = "C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx"
$dataDir = "$installDir\data"
$logFile = "$installDir\service.log"
$nssm = "C:\Windows\System32\nssm.exe"

Write-Host "=== δ-mem-go Service Installer ===" -ForegroundColor Cyan

# Create data dir
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

# Register with NSSM
& $nssm install $serviceName $exePath "--model" $modelPath "--port" "18080" "--grpc-port" "19090" "--data" $dataDir
& $nssm set $serviceName AppDirectory $installDir
& $nssm set $serviceName DisplayName "Delta-Mem-Go Thoughts Engine"
& $nssm set $serviceName Description "Persistent memory + thought synthesis for AI agents"
& $nssm set $serviceName Start SERVICE_AUTO_START
& $nssm set $serviceName AppStdout $logFile
& $nssm set $serviceName AppStderr $logFile
& $nssm set $serviceName AppRotateFiles 1
& $nssm set $serviceName AppRotateBytes 10485760

# Set PATH for CGO runtime
$mingwBin = "C:\Users\dabrush\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
& $nssm set $serviceName AppEnvironmentExtra "PATH=$mingwBin;$env:PATH"

# Start
& $nssm start $serviceName

Write-Host "`n=== Installed ===" -ForegroundColor Green
Write-Host "  Service: $serviceName"
Write-Host "  HTTP:    http://localhost:18080"
Write-Host "  gRPC:    localhost:19090"
Write-Host "  Data:    $dataDir"
Write-Host "  Log:     $logFile"
Write-Host "  Status:  nssm status $serviceName"
