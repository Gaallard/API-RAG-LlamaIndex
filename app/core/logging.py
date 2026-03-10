"""Structured logging configuration with JSON format and request ID middleware."""

import json
import logging
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

# Context variable to store request ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_var.get("")
        if request_id:
            log_data["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging() -> None:
    """Configure application logging."""
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to each request."""

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Add request ID to context and response headers."""
        # Generate or get request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Set in context variable
        request_id_var.set(request_id)

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get("")

