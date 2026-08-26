"""
FallGuard AI - Structured Application Logging
"""

import logging
import sys
from backend.core.config import settings

def setup_logger(name: str = "FallGuard") -> logging.Logger:
    """Configure structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

app_logger = setup_logger("FallGuardAPI")
