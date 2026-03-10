# Guía Rápida de Inicio

## Instalación Rápida (PowerShell)

### Opción 1: Usar el script automático
```powershell
.\install.ps1
```

### Opción 2: Instalación manual (si el script falla)

1. **Crear y activar entorno virtual:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. **Instalar dependencias:**
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest pytest-asyncio ruff
```

3. **Crear archivo .env:**
```powershell
Copy-Item env.example .env
# Edita .env con tus configuraciones
```

4. **Ejecutar la aplicación:**
```powershell
# Opción A: Usar el script
.\run.ps1

# Opción B: Comando directo
python -m uvicorn app.main:app --reload
```

## Solución de Problemas

### Error: "uvicorn no se reconoce"
**Solución:** Usa `python -m uvicorn` en lugar de solo `uvicorn`:
```powershell
python -m uvicorn app.main:app --reload
```

### Error: "No se puede ejecutar scripts"
**Solución:** Cambia la política de ejecución:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error: Path con caracteres especiales
**Solución:** Los scripts `install.ps1` y `run.ps1` ahora manejan esto automáticamente.

### Error: "ModuleNotFoundError"
**Solución:** Asegúrate de que el entorno virtual esté activado:
```powershell
.\venv\Scripts\Activate.ps1
```

## Verificación

Para verificar que todo está instalado correctamente:
```powershell
python -m pip list | Select-String "fastapi|uvicorn|llama-index|qdrant"
```

Deberías ver las dependencias listadas.

