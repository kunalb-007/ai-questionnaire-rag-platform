"""
logging_config.py — Configure application-wide logging.

We use Python's standard logging library with a consistent format that
includes timestamp, log level, logger name, and message. Each request
carries a request_id (injected by middleware) so log lines from the
same request can be correlated.

What we log:
  - Request start/end with latency.
  - Document filename and chunk count on ingestion.
  - Retrieval and LLM latency on query.
  - Errors with full tracebacks (at ERROR level).

What we deliberately do NOT log:
  - API keys or secrets (they live only in Settings).
  - Full document text (potentially confidential).
  - User queries beyond INFO-level acknowledgement (privacy consideration).

Interview note on log levels:
  DEBUG — noisy; useful during development, disabled in production.
  INFO  — normal operations; one line per significant step.
  WARNING — unexpected but handled situations (e.g., empty PDF page).
  ERROR — failures that affect the user; always include the exception.
"""

import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    """
    Set up root logger with a single StreamHandler to stdout.

    Args:
        log_level: string level name from settings (DEBUG/INFO/WARNING/ERROR).

    Called once at application startup from main.py.
    All other modules use logging.getLogger(__name__) — they inherit this
    configuration automatically through the root logger hierarchy.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # override any prior basicConfig calls
    )

    # Suppress noisy third-party loggers in production
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)