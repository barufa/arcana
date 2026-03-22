import sys
import time
from contextlib import contextmanager
from typing import Any

from loguru import logger

# Remove default handler, add JSONL sink to stderr
logger.remove()
logger.add(sys.stderr, serialize=True, level="INFO")


@contextmanager
def log_operation(
    operation: str,
    *,
    metadata: dict[str, Any] | None = None,
):
    """Context manager that logs an operation with its duration.

    Usage:
        with log_operation("search", metadata={"library": "fastmcp"}):
            results = do_search()
    """
    start = time.perf_counter()
    try:
        yield
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.bind(operation=operation, duration_ms=duration_ms, metadata=metadata).error(
            f"{operation} failed"
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.bind(operation=operation, duration_ms=duration_ms, metadata=metadata).info(
            f"{operation} completed"
        )
