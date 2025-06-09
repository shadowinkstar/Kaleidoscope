from loguru import logger
import sys
from pathlib import Path
import os
from typing import List

# store log messages in memory so that UI can display them
LOG_BUFFER: List[str] = []

def init_logger():
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "kaleidoscope.log"

    logger.remove()
    logger.add(sys.stdout, level=level, colorize=True, enqueue=True)
    logger.add(log_file, level="DEBUG", rotation="1 MB", encoding="utf-8", enqueue=True)
    logger.add(LOG_BUFFER.append, format="{message}", level=level, enqueue=True)

def get_logs(start: int = 0) -> list[str]:
    """Return log messages starting from ``start``."""
    return LOG_BUFFER[start:]

init_logger()
