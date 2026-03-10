# Script de instalación para RAG FastAPI Application
# Ejecutar desde PowerShell: .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== Instalación de RAG FastAPI Application ===" -ForegroundColor Cyan

# Obtener el directorio actual
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Verificar Python
Write-Host "`n[1/5] Verificando Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python encontrado: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python no está instalado o no está en el PATH" -ForegroundColor Red
    Write-Host "Por favor, instala Python 3.11+ desde https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Crear venv si no existe
Write-Host "`n[2/5] Verificando entorno virtual..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    Write-Host "Creando entorno virtual..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: No se pudo crear el entorno virtual" -ForegroundColor Red
        exit 1
    }
    Write-Host "Entorno virtual creado" -ForegroundColor Green
} else {
    Write-Host "Entorno virtual ya existe" -ForegroundColor Green
}

# Activar venv
Write-Host "`n[3/5] Activando entorno virtual..." -ForegroundColor Yellow
$activateScript = Join-Path $scriptPath "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Entorno virtual activado" -ForegroundColor Green
} else {
    Write-Host "ADVERTENCIA: No se encontró el script de activación" -ForegroundColor Yellow
    Write-Host "Por favor, ejecuta manualmente: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

# Actualizar pip
Write-Host "`n[4/5] Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "pip actualizado" -ForegroundColor Green
} else {
    Write-Host "ADVERTENCIA: No se pudo actualizar pip" -ForegroundColor Yellow
}

# Instalar dependencias
Write-Host "`n[5/5] Instalando dependencias (esto puede tardar varios minutos)..." -ForegroundColor Yellow

# Verificar si requirements.txt existe
if (Test-Path "requirements.txt") {
    Write-Host "Instalando desde requirements.txt..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Falló la instalación desde requirements.txt" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "ADVERTENCIA: requirements.txt no encontrado" -ForegroundColor Yellow
    Write-Host "Instalando dependencias básicas..." -ForegroundColor Yellow
    
    # Instalar dependencias básicas directamente
    python -m pip install fastapi uvicorn[standard] pydantic pydantic-settings
    python -m pip install llama-index llama-index-core llama-index-embeddings-openai llama-index-readers-file
    python -m pip install qdrant-client python-multipart httpx pypdf
}

# Instalar dependencias de desarrollo
Write-Host "Instalando dependencias de desarrollo..." -ForegroundColor Yellow
python -m pip install pytest pytest-asyncio ruff --quiet

# Verificar instalación
Write-Host "`nVerificando instalación..." -ForegroundColor Yellow
try {
    $uvicornVersion = python -m uvicorn --version 2>&1
    Write-Host "uvicorn instalado correctamente" -ForegroundColor Green
} catch {
    Write-Host "ADVERTENCIA: No se pudo verificar uvicorn" -ForegroundColor Yellow
}

Write-Host "`n=== Instalación completada ===" -ForegroundColor Green
Write-Host "`nPróximos pasos:" -ForegroundColor Cyan
Write-Host "1. Activa el entorno virtual: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Crea el archivo .env: Copy-Item env.example .env" -ForegroundColor White
Write-Host "3. Edita .env con tus configuraciones (API_KEY, OPENAI_API_KEY, etc.)" -ForegroundColor White
Write-Host "4. Ejecuta la app: uvicorn app.main:app --reload" -ForegroundColor White
Write-Host "`nO ejecuta directamente: python -m uvicorn app.main:app --reload" -ForegroundColor Gray
