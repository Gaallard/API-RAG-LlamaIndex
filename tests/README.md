# Tests

Tests para la aplicación RAG FastAPI usando pytest y httpx TestClient.

## Estructura

- `conftest.py`: Fixtures compartidas (mocks, client, test data)
- `test_health.py`: Tests de health checks
- `test_ingest.py`: Tests de ingesta de documentos
- `test_query.py`: Tests de queries RAG

## Fixtures Principales

- `test_api_key`: API key para tests
- `test_headers`: Headers con API key
- `mock_embeddings`: Mock del modelo de embeddings (evita llamadas a OpenAI)
- `mock_llm`: Mock del LLM (evita llamadas a OpenAI)
- `client`: AsyncClient de httpx para hacer requests
- `sample_txt_content`: Contenido de texto de ejemplo para tests
- `sample_txt_file`: Archivo de texto de ejemplo

## Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_health.py
pytest tests/test_ingest.py
pytest tests/test_query.py

# Con verbose
pytest -v

# Con coverage
pytest --cov=app --cov-report=html
```

## Tests Implementados

### test_health_ok
- Valida que `/health/` responde 200
- Verifica estructura de respuesta (status, vector_store, collection)

### test_ingest_txt
- Sube un archivo .txt pequeño
- Valida que `ingested >= 1`
- Usa mocks para evitar dependencias externas

### test_query
- Ingesta un documento
- Hace una query sobre contenido del documento
- Valida que `answer` no esté vacío
- Valida que `sources` tenga al menos 1 elemento

## Mocks

Los tests usan mocks para:
- **Embeddings**: Retorna vectores fake (1536 dimensiones)
- **LLM**: Retorna respuestas fake sin llamar a OpenAI
- **Qdrant**: Simula almacenamiento sin conexión real

Esto permite ejecutar tests sin:
- API keys de OpenAI
- Servidor Qdrant corriendo
- Dependencias externas

## Configuración

Los tests usan configuración de test automática:
- API key de test: `test-api-key-12345`
- Directorio temporal para datos
- Mocks de servicios externos

No se requiere archivo `.env.test` - los tests configuran todo automáticamente.

