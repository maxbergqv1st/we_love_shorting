import urllib.error
from unittest.mock import Mock

import pytest

from we_love_shorting.retry import retry


def test_retry_succeeds_after_transient_failures():
    # Mock function that fails 2 times, then 3rd is success
    attempts = [
        urllib.error.HTTPError(url="", code=429, msg="", hdrs=None, fp=None),
        urllib.error.HTTPError(url="", code=429, msg="", hdrs=None, fp=None),
        "success",
    ]
    mock_func = Mock(side_effect=attempts)
    wrapped = retry(times=4, exceptions=(urllib.error.HTTPError,))(mock_func)

    result = wrapped()

    assert result == "success"
    assert mock_func.call_count == 3  # Two failed attempts + one success


def test_retry_gives_up_after_max_attempts():
    mock_func = Mock(
        side_effect=urllib.error.HTTPError(url="", code=429, msg="", hdrs=None, fp=None)
    )
    wrapped = retry(times=3, exceptions=(urllib.error.HTTPError,))(mock_func)

    with pytest.raises(urllib.error.HTTPError):
        wrapped()

    assert mock_func.call_count == 3  # Gives up after 3 retries


def test_retry_does_not_catch_unlisted_exceptions():
    # VlaueError is not in the exceptions tuple -> Should not be catched/retried
    mock_func = Mock(side_effect=ValueError("bad query"))
    wrapped = retry(times=4, exceptions=(urllib.error.HTTPError,))(mock_func)

    with pytest.raises(ValueError):
        wrapped()

    assert mock_func.call_count == 1  # No retries
