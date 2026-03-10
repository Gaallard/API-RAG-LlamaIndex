# RAG FastAPI Application

A FastAPI-based RAG (Retrieval-Augmented Generation) application using LlamaIndex and Qdrant for document ingestion and querying.

## Features

- **Document Ingestion**: Upload and process documents (PDF, TXT, DOCX, etc.)
- **Vector Search**: Query documents using semantic search
- **Qdrant Integration**: Vector database for storing embeddings
- **LlamaIndex**: Document processing and embedding generation
- **API Authentication**: API key-based authentication
- **Structured Logging**: JSON-formatted logs with request IDs
- **Health Checks**: Health, readiness, and liveness endpoints

## Project Structure

```
.
├── app/
    ├── core/
    │   ├── config.py          # Application settings
    │   ├── logging.py          # Logging configuration
    │   └── security.py         # API key authentication
    ├── routers/
    │   ├── health.py           # Health check endpoints
    │   ├── ingest.py           # Document ingestion endpoints
    │   ├── query.py            # Query endpoints
    │   └── stats.py            # Statistics endpoints
    ├── services/
    │   ├── ingest_service.py   # Document ingestion logic
    │   └── query_service.py    # Query logic
    ├── storage/
    │   └── qdrant_store.py     # Qdrant client wrapper
    └── main.py                 # FastAPI application
├── tests/
    ├── test_health.py
    ├── test_ingest.py
    └── test_query.py
├── pyproject.toml
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.11+ (for local development)
- Docker and Docker Compose (for containerized deployment)
- Qdrant server running (default: http://localhost:6333) - included in docker-compose
- OpenAI API key (optional, can use Ollama instead)

## Quick Start with Docker (Recommended)

The easiest way to run the application is using Docker Compose:

1. **Create `.env` file**:
   ```bash
   cp env.example .env
   ```
   Edit `.env` and configure:
   - `API_KEY`: Your secret API key
   - `OPENAI_API_KEY`: Your OpenAI API key (required for embeddings)
   - `QDRANT_URL`: Will be set automatically by docker-compose to `http://qdrant:6333`

2. **Start services**:
   ```bash
   docker-compose up -d
   ```

3. **Check services**:
   ```bash
   docker-compose ps
   ```

4. **View logs**:
   ```bash
   docker-compose logs -f api
   ```

5. **Access the API**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Qdrant Dashboard: http://localhost:6333/dashboard

6. **Stop services**:
   ```bash
   docker-compose down
   ```

**Note**: Data is persisted in Docker volumes. To remove all data:
```bash
docker-compose down -v
```

## Local Installation (Without Docker)

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   
   **Windows (PowerShell):**
   ```powershell
   # Opción 1: Script automático (recomendado)
   .\install.ps1
   
   # Opción 2: Manual
   .\venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install pytest pytest-asyncio ruff
   ```
   
   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   pip install -e ".[dev]"
   # O alternativamente:
   pip install -r requirements.txt
   pip install pytest pytest-asyncio ruff
   ```

5. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set your configuration:
   - `API_KEY`: Your secret API key for authentication
   - `OPENAI_API_KEY`: Your OpenAI API key (optional)
   - `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
   - `QDRANT_COLLECTION`: Collection name (default: rag_documents)
   - Other settings as needed

6. **Start Qdrant server** (if not already running):
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Running the Application

### Docker Compose (Recommended)

**Start all services** (API + Qdrant):
```bash
docker-compose up -d
```

**View logs**:
```bash
docker-compose logs -f api
docker-compose logs -f qdrant
```

**Stop services**:
```bash
docker-compose down
```

**Rebuild after code changes**:
```bash
docker-compose up -d --build
```

**Access services**:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Qdrant: http://localhost:6333
- Qdrant Dashboard: http://localhost:6333/dashboard

### Local Development Mode

**Windows (PowerShell):**
```powershell
# Opción 1: Script automático
.\run.ps1

# Opción 2: Comando directo
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Linux/Mac:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# O alternativamente:
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Note**: For local development, you need Qdrant running separately:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Production Mode

**With Docker**:
```bash
docker-compose up -d
```

**Without Docker**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Checks

- `GET /health/` - Basic health check
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check

### Document Ingestion

- `POST /ingest/` - Upload and ingest a document
  - Requires: `X-API-Key` header
  - Body: multipart/form-data with `file` field
  - Returns: Document ID and status

- `GET /ingest/status/{document_id}` - Get ingestion status

### Query

- `POST /query/` - Query documents
  - Requires: `X-API-Key` header
  - Body: JSON with `query` (string) and `top_k` (int, optional)
  - Returns: List of matching documents with scores

- `GET /query/similarity/{document_id}` - Find similar documents

### Statistics

- `GET /stats/` - Get system statistics
  - Requires: `X-API-Key` header
  - Returns: Total documents, collection info, etc.

## Testing

Run tests with pytest:

```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_health.py
```

## Development

### Code Formatting

The project uses `ruff` for linting and formatting:

```bash
ruff check .
ruff format .
```

### Environment Variables

All configuration is done through environment variables. See `.env.example` for available options.

### Adding New Features

The scaffold includes:
- ✅ Basic FastAPI structure
- ✅ API key authentication
- ✅ Structured logging with request IDs
- ✅ Health check endpoints
- ✅ Document ingestion endpoints (stub)
- ✅ Query endpoints (stub)
- ✅ Statistics endpoints (stub)
- ✅ Qdrant integration (stub)
- ⏳ Full LlamaIndex integration (TODO)
- ⏳ Complete RAG pipeline (TODO)

## Docker Configuration

### Dockerfile

The `Dockerfile` uses a multi-stage build:
- **Builder stage**: Installs Python dependencies
- **Production stage**: Minimal image with only runtime dependencies

### docker-compose.yml

The `docker-compose.yml` defines two services:

1. **qdrant**: Vector database service
   - Ports: 6333 (REST), 6334 (gRPC)
   - Volume: `qdrant_storage` for persistence
   - Health check enabled

2. **api**: FastAPI application
   - Port: 8000
   - Volume: `./data` for document storage
   - Environment: Loads from `.env` file
   - Depends on: Qdrant service

### Environment Variables

Create a `.env` file based on `env.example`:

```env
# Required
API_KEY=your-secret-api-key-here

# Required for embeddings
OPENAI_API_KEY=your-openai-api-key-here

# Qdrant (automatically set by docker-compose)
QDRANT_URL=http://qdrant:6333

# Optional
CORS_ORIGINS=*
LOG_LEVEL=INFO
```

### Docker Commands

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (deletes data)
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build

# Execute commands in container
docker-compose exec api python -m pytest
```

### Data Persistence

- **Qdrant data**: Stored in Docker volume `qdrant_storage`
- **Documents**: Stored in `./data` directory (mounted as volume)
- **Environment**: Loaded from `.env` file

## Development

### Running Tests

**With Docker**:
```bash
docker-compose exec api pytest
```

**Local**:
```bash
pytest
```

See `tests/README.md` for more details.

## Next Steps

The implementation is complete with:
- ✅ Full LlamaIndex integration
- ✅ Complete Qdrant operations
- ✅ Document ingestion with chunking and embeddings
- ✅ RAG query with sources
- ✅ Streaming SSE support
- ✅ Health checks and statistics
- ✅ API key authentication
- ✅ Docker support

## License

This project is for technical assessment purposes.

