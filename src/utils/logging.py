from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path) -> None:
    if logging.getLogger("swingquant").handlers:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / "swingquant.log")
    file_handler.setFormatter(formatter)

    logger = logging.getLogger("swingquant")
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"swingquant.{name}")
