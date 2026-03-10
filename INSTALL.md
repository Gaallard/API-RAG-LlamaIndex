# Instrucciones de Instalación

## Pasos para instalar las dependencias

### 1. Activar el entorno virtual (si existe)
```powershell
.\venv\Scripts\Activate.ps1
```

Si no existe, créalo primero:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Actualizar pip
```powershell
python -m pip install --upgrade pip
```

### 3. Instalar dependencias

**Opción A: Usando pyproject.toml (recomendado)**
```powershell
pip install -e ".[dev]"
```

**Opción B: Usando requirements.txt**
```powershell
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx ruff
```

### 4. Verificar instalación
```powershell
uvicorn --version
```

### 5. Ejecutar la aplicación
```powershell
uvicorn app.main:app --reload
```

## Nota sobre permisos en PowerShell

Si tienes problemas al activar el venv, puede ser necesario cambiar la política de ejecución:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Configurar variables de entorno

Crea un archivo `.env` basado en `env.example`:
```powershell
Copy-Item env.example .env
```

Luego edita `.env` y configura:
- `API_KEY`: Tu clave de API para autenticación
- `OPENAI_API_KEY`: Tu clave de API de OpenAI
- `QDRANT_URL`: URL de Qdrant (default: http://localhost:6333)
- Otros parámetros según necesites

