from datetime import datetime, timedelta, timezone
from typing import Tuple

import pytest

import conbench.util


@pytest.mark.parametrize(
    "teststring",
    ["2022-03-03T19:48:06", "2022-03-03T19:48:06Z", "2022-03-03T19:48:06+02:00"],
)
def test_tznaive_iso8601_to_tzaware_dt(teststring: str):
    # Confirm that
    # - UTC tzinfo is attached when input is naive.
    # - UTC tzinfo in input is OK.
    # - any non-UTC tzinfo in the input gets replaced with
    #   UTC, retaining the numbers.
    dt = conbench.util.tznaive_iso8601_to_tzaware_dt(teststring)
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2022
    assert dt.second == 6
    assert dt.hour == 19


@pytest.mark.parametrize(
    "param",
    [
        # Confirm that tz-aware dt obj does not crash function, and that
        # tz info is indeed retained, with Z technique for UTC.
        (
            datetime(2000, 10, 10, 17, 30, 55, tzinfo=timezone.utc),
            "2000-10-10T17:30:55Z",
        ),
        (
            datetime(2000, 10, 10, 17, 30, 55, tzinfo=timezone(timedelta(hours=5))),
            "2000-10-10T17:30:55+05:00",
        ),
        # Confirm taht tz-naive dt objects are interpreted in UTC.
        (datetime(2000, 10, 10, 17, 30, 55), "2000-10-10T17:30:55Z"),
        # Confirm that fractions of seconds are ignored in output.
        (datetime(2000, 10, 10, 17, 30, 55, 1337), "2000-10-10T17:30:55Z"),
    ],
)
def test_tznaive_dt_to_aware_iso8601_for_api(param: Tuple[datetime, str]):
    assert conbench.util.tznaive_dt_to_aware_iso8601_for_api(param[0]) == param[1]


def test_tznaive_iso8601_to_tzaware_dt_nat_single():
    """NaT (pandas null timestamp stringified) must not crash the converter.

    When pandas NaT values are serialized via .isoformat() the result is the
    string 'NaT'.  datetime.fromisoformat('NaT') raises ValueError, so the
    converter must handle this gracefully and return None.
    """
    result = conbench.util.tznaive_iso8601_to_tzaware_dt("NaT")
    assert result is None


def test_tznaive_iso8601_to_tzaware_dt_nat_in_list():
    """When a list of timestamp strings contains 'NaT' entries, the converter
    must return None for those positions while still converting valid strings.
    """
    inputs = ["2022-03-03T19:48:06", "NaT", "2023-01-01T00:00:00"]
    results = conbench.util.tznaive_iso8601_to_tzaware_dt(inputs)

    assert len(results) == 3
    assert results[0].tzinfo == timezone.utc
    assert results[0].year == 2022
    assert results[1] is None
    assert results[2].tzinfo == timezone.utc
    assert results[2].year == 2023


def test_tznaive_iso8601_to_tzaware_dt_all_nat():
    """Edge case: every entry is NaT."""
    results = conbench.util.tznaive_iso8601_to_tzaware_dt(["NaT", "NaT"])
    assert results == [None, None]
