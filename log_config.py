from loguru import logger
import sys
from pathlib import Path
import os

def init_logger():
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "kaleidoscope.log"

    logger.remove()
    logger.add(sys.stdout, level=level, colorize=True, enqueue=True)
    logger.add(log_file, level="DEBUG", rotation="1 MB", encoding="utf-8", enqueue=True)

init_logger()
