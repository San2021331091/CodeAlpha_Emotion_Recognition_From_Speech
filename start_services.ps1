# CodeAlpha Emotion Recognition from Speech - Multi-Tier Service Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🚀 Launching CodeAlpha Speech Emotion Recognition System..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$root = $PSScriptRoot
$mlPython = Join-Path $root 'ml-service\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $mlPython)) { $mlPython = 'python' }
$backendRoot = Join-Path $root 'backend-gateway'
$backendTsc = Join-Path $backendRoot 'node_modules\.bin\tsc.cmd'
if (-not (Test-Path -LiteralPath $backendTsc)) {
    throw "TypeScript compiler not found at '$backendTsc'. Install backend dependencies first."
}

Write-Host "`nBuilding NestJS Backend Gateway..." -ForegroundColor Green
Push-Location $backendRoot
try {
    & $backendTsc --project .\tsconfig.json --incremental false
    $backendBuildExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
if ($backendBuildExitCode -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $backendRoot 'dist\main.js'))) {
    throw "NestJS backend build failed; refusing to start an incomplete gateway."
}
$node = (Get-Command node.exe -ErrorAction Stop).Source

# 1. Start FastAPI ML Service (Port 8000)
Write-Host "`n[1/3] Starting FastAPI ML Microservice on http://127.0.0.1:8000..." -ForegroundColor Green
$mlProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\ml-service'; `$env:PYTHONIOENCODING='utf-8'; & '$mlPython' -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload" -WindowStyle Hidden -PassThru

# Wait 2 seconds
Start-Sleep -Seconds 2

# 2. Start NestJS Backend Gateway (Port 3001)
Write-Host "[2/3] Starting NestJS Backend Gateway on http://localhost:3001..." -ForegroundColor Green
$backendProcess = Start-Process -FilePath $node -ArgumentList ".\dist\main.js" -WorkingDirectory $backendRoot -WindowStyle Hidden -PassThru

# Wait 2 seconds
Start-Sleep -Seconds 2

# 3. Start Next.js Frontend (Port 3000)
Write-Host "[3/3] Starting Next.js Modern Frontend on http://localhost:3000..." -ForegroundColor Green
$frontendProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev" -WindowStyle Hidden -PassThru

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "✨ All 3 tiers initiated successfully!" -ForegroundColor Green
Write-Host "   - Frontend UI:       http://localhost:3000" -ForegroundColor White
Write-Host "   - NestJS Gateway:    http://localhost:3001" -ForegroundColor White
Write-Host "   - FastAPI ML Docs:   http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
