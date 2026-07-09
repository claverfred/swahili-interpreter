"""
Centralized logger using loguru.
Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting pipeline...")
"""
import sys
import os
from loguru import logger


def get_logger(name: str = "swahili_interpreter"):
    """Return a configured loguru logger."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()  # Remove default handler

    # Console — clean, coloured
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level:<8}</level> | "
               "<cyan>{name}</cyan> | "
               "{message}",
        level="INFO",
        colorize=True,
    )

    # File — full detail
    logger.add(
        f"{log_dir}/pipeline.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )

    return logger.bind(name=name)
