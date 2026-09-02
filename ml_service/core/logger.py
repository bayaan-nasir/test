"""
core/logger.py
Centralised logging using loguru.
Import `logger` anywhere — do not use print() in production code.
"""
import sys
from loguru import logger
from core.config import settings

# Remove default handler
logger.remove()

# Console handler — colourful in debug, clean in production
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)

logger.add(
    sys.stdout,
    format=log_format,
    level="DEBUG" if settings.debug else "INFO",
    colorize=True,
)

# File handler — always INFO and above
logger.add(
    "logs/ml_service.log",
    format=log_format,
    level="INFO",
    rotation="10 MB",
    retention="14 days",
    compression="zip",
)
