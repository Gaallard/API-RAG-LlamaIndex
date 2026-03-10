# Script para ejecutar la aplicación
# Ejecutar desde PowerShell: .\run.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== Iniciando RAG FastAPI Application ===" -ForegroundColor Cyan

# Obtener el directorio actual
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Verificar si venv existe
if (-not (Test-Path "venv")) {
    Write-Host "ERROR: Entorno virtual no encontrado" -ForegroundColor Red
    Write-Host "Por favor, ejecuta primero: .\install.ps1" -ForegroundColor Yellow
    exit 1
}

# Activar venv
$activateScript = Join-Path $scriptPath "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "ADVERTENCIA: No se encontró el script de activación" -ForegroundColor Yellow
    Write-Host "Intentando ejecutar sin activar venv..." -ForegroundColor Yellow
}

# Verificar .env
if (-not (Test-Path ".env")) {
    Write-Host "ADVERTENCIA: Archivo .env no encontrado" -ForegroundColor Yellow
    if (Test-Path "env.example") {
        Write-Host "Creando .env desde env.example..." -ForegroundColor Yellow
        Copy-Item env.example .env
        Write-Host "Por favor, edita .env con tus configuraciones antes de continuar" -ForegroundColor Yellow
        Write-Host "Presiona Enter para continuar o Ctrl+C para cancelar..."
        Read-Host
    }
}

# Ejecutar aplicación
Write-Host "`nIniciando servidor..." -ForegroundColor Green
Write-Host "API disponible en: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Documentación en: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "`nPresiona Ctrl+C para detener el servidor`n" -ForegroundColor Gray

python -m uvicorn app.main:app --reload

