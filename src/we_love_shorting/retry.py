# src/we_love_shorting/retry.py
"""Generic retry decorator for transient failures (rate limits, flaky network, etc.)."""

import logging
import time
from functools import wraps

log = logging.getLogger(__name__)


def retry(times: int = 4, exceptions: tuple = (Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == times - 1:
                        raise
                    wait = 2**attempt
                    log.warning(
                        "%s failed (%s), retrying in %ss",
                        getattr(func, "__name__", repr(func)),
                        e,
                        wait,
                    )
                    time.sleep(wait)

        return wrapper

    return decorator
