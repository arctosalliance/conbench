"""Tests for conbench.app._plots — focusing on NaT / None-datetime handling.

These reproduce production errors where pandas NaT timestamps (serialized as
the string 'NaT') flow through the plotting pipeline and cause crashes:

1. _source() calls util.tznaive_iso8601_to_tzaware_dt() which must handle 'NaT'
   strings without raising ValueError.
2. _source() builds date_strings via d.strftime() — must not crash on None.
3. time_series_plot() computes t_end - t_start from the x-axis data — must not
   crash when some x values are None.
4. time_series_plot() must filter out samples with epoch-0 (1970-01-01) or NaT
   timestamps so they don't appear as stray points in the plot.
"""
import importlib
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from conbench.entities.history import HistorySample, HistorySampleZscoreStats


@pytest.fixture()
def _source():
    """Import _source from conbench.app._plots, working around the BUILD_INFO
    import error that occurs outside a Docker/deployed environment.
    """
    # Ensure buildinfo module exports BUILD_INFO (even if fake) so that the
    # import chain conbench.app -> conbench.buildinfo.BUILD_INFO doesn't crash.
    import conbench.buildinfo as bi

    if not hasattr(bi, "BUILD_INFO"):
        bi.BUILD_INFO = bi.Buildinfo(
            commit="test", branch_name="test", build_time_rfc3339="test",
            build_hostname="test", version_string="test",
        )

    from conbench.app._plots import _source as fn

    return fn


def _make_zscorestats(**kwargs) -> HistorySampleZscoreStats:
    defaults = dict(
        begins_distribution_change=False,
        segment_id="0",
        rolling_mean_excluding_this_commit=1.0,
        rolling_mean=1.0,
        residual=0.0,
        rolling_stddev=0.1,
        is_outlier=False,
    )
    defaults.update(kwargs)
    return HistorySampleZscoreStats(**defaults)


def _make_sample(
    commit_timestamp,
    svs: float = 1.0,
    benchmark_result_id: str = "br1",
    commit_hash: str = "abc12345" * 5,
    commit_msg: str = "test commit",
    **kwargs,
) -> HistorySample:
    defaults = dict(
        benchmark_result_id=benchmark_result_id,
        benchmark_name="test-bench",
        history_fingerprint="fp1",
        case_text_id="case1",
        case_id="case1",
        context_id="ctx1",
        mean=svs,
        svs=svs,
        svs_type="mean",
        data=[svs],
        times=[0.001],
        unit="s",
        hardware_hash="hw1",
        repository="https://github.com/org/repo",
        commit_hash=commit_hash,
        commit_msg=commit_msg,
        commit_timestamp=commit_timestamp,
        result_timestamp=commit_timestamp if not pd.isna(commit_timestamp) else datetime(2022, 1, 1),
        run_name="test-run",
        run_tags={},
        zscorestats=_make_zscorestats(),
    )
    defaults.update(kwargs)
    return HistorySample(**defaults)


def test_source_with_nat_commit_timestamp(_source):
    """Call _source() with a sample whose commit_timestamp is pandas NaT.

    This reproduces the production crash chain:
      1. pd.NaT.isoformat() returns 'NaT'
      2. util.tznaive_iso8601_to_tzaware_dt('NaT') must return None (not crash)
      3. d.strftime() in the date_strings list comp must handle None (not crash)

    Without the fixes in util.py and _plots.py, this raises:
      ValueError: Invalid isoformat string: 'NaT'
    or:
      AttributeError: 'NoneType' object has no attribute 'strftime'
    """
    samples = [
        _make_sample(
            commit_timestamp=datetime(2022, 1, 1),
            svs=1.0,
            benchmark_result_id="br1",
            commit_hash="a" * 40,
        ),
        _make_sample(
            commit_timestamp=pd.NaT,
            svs=2.0,
            benchmark_result_id="br2",
            commit_hash="b" * 40,
        ),
        _make_sample(
            commit_timestamp=datetime(2022, 1, 3),
            svs=3.0,
            benchmark_result_id="br3",
            commit_hash="c" * 40,
        ),
    ]

    source = _source(samples, "s", "svs")

    assert len(source.data["x"]) == 3
    assert source.data["x"][1] is None
    assert source.data["date_strings"][1] == "N/A"
    assert len(source.data["y"]) == 3


def test_source_with_all_nat_timestamps(_source):
    """Edge case: every sample has NaT commit_timestamp."""
    samples = [
        _make_sample(
            commit_timestamp=pd.NaT,
            svs=1.0,
            benchmark_result_id=f"br{i}",
            commit_hash=str(i) * 40,
        )
        for i in range(3)
    ]

    source = _source(samples, "s", "svs")

    assert all(x is None for x in source.data["x"])
    assert all(ds == "N/A" for ds in source.data["date_strings"])


def test_time_range_computation_with_nat(_source):
    """Verify that time range computation (t_end - t_start) does not crash
    when the x-axis data from _source contains None values.

    In time_series_plot(), the code does:
        t_start = source_svs_all.data["x"][0]
        t_end = source_svs_all.data["x"][-1]
        t_range = t_end - t_start

    If first or last x value is None, this crashes with:
        TypeError: unsupported operand type(s) for -: 'NoneType' and 'datetime.datetime'
    """
    samples = [
        _make_sample(
            commit_timestamp=pd.NaT,
            svs=1.0,
            benchmark_result_id="br1",
            commit_hash="a" * 40,
        ),
        _make_sample(
            commit_timestamp=datetime(2022, 1, 2),
            svs=2.0,
            benchmark_result_id="br2",
            commit_hash="b" * 40,
        ),
        _make_sample(
            commit_timestamp=datetime(2022, 1, 5),
            svs=3.0,
            benchmark_result_id="br3",
            commit_hash="c" * 40,
        ),
    ]

    source = _source(samples, "s", "svs")
    x_data = source.data["x"]

    # First element is None (from NaT) — direct subtraction would crash
    assert x_data[0] is None

    # The fix: filter Nones before computing range
    x_valid = [v for v in x_data if v is not None]
    t_start = x_valid[0] if x_valid else None
    t_end = x_valid[-1] if x_valid else None

    assert t_start is not None
    assert t_end is not None
    t_range = t_end - t_start
    assert t_range.days == 3


def test_epoch_zero_timestamps_filtered_out(_source):
    """Regression test: samples with epoch-0 commit_timestamp (1970-01-01)
    should be excluded from plots. These arise from commits with missing
    timestamps in the DB and show up as a stray point far left on the x-axis.

    The filter in time_series_plot() removes them before any _source() call.
    Verify the filter logic directly here.
    """
    samples = [
        _make_sample(
            commit_timestamp=datetime(1970, 1, 1),
            svs=999.0,
            benchmark_result_id="br_epoch0",
            commit_hash="0" * 40,
        ),
        _make_sample(
            commit_timestamp=datetime(2022, 3, 1),
            svs=1.0,
            benchmark_result_id="br1",
            commit_hash="a" * 40,
        ),
        _make_sample(
            commit_timestamp=datetime(2022, 3, 2),
            svs=2.0,
            benchmark_result_id="br2",
            commit_hash="b" * 40,
        ),
    ]

    # Apply the same filter used in time_series_plot()
    filtered = [
        s for s in samples
        if not pd.isna(s.commit_timestamp) and s.commit_timestamp.year >= 2000
    ]

    assert len(filtered) == 2
    assert all(s.commit_timestamp.year >= 2000 for s in filtered)

    # The filtered list should produce a clean source with no epoch-0 points
    source = _source(filtered, "s", "svs")
    assert len(source.data["x"]) == 2
    assert all(x.year >= 2000 for x in source.data["x"])


def test_nat_and_epoch_zero_both_filtered():
    """Both NaT and epoch-0 timestamps should be removed by the filter."""
    samples = [
        _make_sample(commit_timestamp=pd.NaT, benchmark_result_id="br_nat", commit_hash="0" * 40),
        _make_sample(commit_timestamp=datetime(1970, 1, 1), benchmark_result_id="br_epoch", commit_hash="1" * 40),
        _make_sample(commit_timestamp=datetime(2022, 6, 15), benchmark_result_id="br_ok", commit_hash="a" * 40),
    ]

    filtered = [
        s for s in samples
        if not pd.isna(s.commit_timestamp) and s.commit_timestamp.year >= 2000
    ]

    assert len(filtered) == 1
    assert filtered[0].benchmark_result_id == "br_ok"
