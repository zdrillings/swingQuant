from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import time
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def retry(
    retries: int = 3,
    backoff_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger=None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a function with exponential backoff."""

    if retries < 1:
        raise ValueError("retries must be at least 1")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            delay = backoff_seconds
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == retries:
                        raise
                    if logger is not None:
                        logger.warning(
                            "Retrying %s after failure %s/%s: %s",
                            func.__name__,
                            attempt,
                            retries,
                            exc,
                        )
                    sleep_fn(delay)
                    delay *= backoff_multiplier
            raise RuntimeError("retry wrapper exhausted unexpectedly")

        return wrapper

    return decorator
