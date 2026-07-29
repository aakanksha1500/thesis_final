"""
Shared logger for all modules in the project.
So every subsequent model can use the same logger instance
without circular dependencies. All agents, the orchestrator, and the
evaluation pipeline use get_logger() to obtain the logger instance.

Log level is controlled by DEBUG env var (set in .env file).
DEBUG=true  -> DEBUG + above
DEBUG=false -> INFO + above
"""

from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with a consistent format.
    Safe to call multiple times; will return the same logger instance for the same name.
    Handlers are added only once to prevent duplicate logs.

    Usage:
        from utlis.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Agent initialised")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # DEBUG env var - set in .env via pythondotenv - controls log level
        debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

        # Prevent propagation to the root logger to avoid duplicate logs
        logger.propagate = False

    return logger
