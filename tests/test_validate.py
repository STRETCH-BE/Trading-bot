from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import FIXTURES, H1
from trading_bot.data import schema
from trading_bot.data.dumps import ingest
from trading_bot.data.normalize import read_dump_csv
from trading_bot.data.validate import validate_frame, validate_store


@pytest.fixture
def clean():
    return read_dump_csv(str(FIXTURES / "XBTEUR_60.csv"))


def _codes(report, severity=None):
    return [i.code for i in report.issues if severity in (None, i.severity)]


def test_clean_frame_passes(clean):
    report = validate_frame(clean, H1, dataset="XBTEUR/1h")
    assert report.ok
    assert report.issues == []
    assert report.stats["rows"] == 48
    assert report.stats["expected_rows"] == 48
    assert report.stats["missing_rows"] == 0


def test_duplicates_are_errors(clean):
    df = pd.concat([clean, clean.tail(2)], ignore_index=True)
    report = validate_frame(df, H1)
    assert not report.ok
    assert "duplicate_timestamps" in _codes(report, "error")


def test_non_monotonic_is_error(clean):
    df = pd.concat([clean.tail(5), clean.head(43)], ignore_index=True)
    report = validate_frame(df, H1)
    assert "non_monotonic" in _codes(report, "error")


def test_gap_detection(clean):
    df = pd.concat([clean.iloc[:10], clean.iloc[13:]], ignore_index=True)
    report = validate_frame(df, H1)
    assert not report.ok
    assert "gaps" in _codes(report, "error")
    assert report.stats["missing_rows"] == 3
    assert report.stats["gap_runs"] == 1
    assert "3 candle(s) missing" in report.errors[0].message


def test_off_grid_timestamp_is_error(clean):
    df = clean.copy()
    shifted = df.loc[20, schema.TIMESTAMP] + pd.Timedelta(minutes=30)
    df.loc[20, schema.TIMESTAMP] = shifted
    report = validate_frame(df, H1)
    assert "off_grid" in _codes(report, "error")
    # exactly the shifted row is named, and the grid candle it vacated counts
    # as missing despite the fractional-step diffs around it
    [off_grid_issue] = [i for i in report.issues if i.code == "off_grid"]
    assert "1 timestamp(s)" in off_grid_issue.message
    assert str(shifted) in off_grid_issue.message
    assert report.stats["missing_rows"] == 1


def test_uniformly_shifted_series_is_off_grid(clean):
    df = clean.copy()
    df[schema.TIMESTAMP] = df[schema.TIMESTAMP] + pd.Timedelta(minutes=17)
    report = validate_frame(df, H1)
    assert not report.ok
    [off_grid_issue] = [i for i in report.issues if i.code == "off_grid"]
    assert "48 timestamp(s)" in off_grid_issue.message


def test_null_timestamp_is_error_not_crash(clean):
    df = clean.copy()
    df.loc[5, schema.TIMESTAMP] = pd.NaT
    report = validate_frame(df, H1)
    assert "null_timestamps" in _codes(report, "error")
    # remaining checks still run on the surviving rows: dropping row 5 leaves
    # a genuine one-candle gap
    assert report.stats["missing_rows"] == 1


def test_zero_volume_mid_active_is_warning(clean):
    df = clean.copy()
    df.loc[24, schema.VOLUME] = 0.0
    report = validate_frame(df, H1)
    assert report.ok  # warning, not error
    assert "zero_volume_active" in _codes(report, "warning")
    assert report.stats["zero_volume_active_rows"] == 1


def test_zero_volume_in_leading_dead_zone_not_flagged(clean):
    df = clean.copy()
    df.loc[:9, schema.VOLUME] = 0.0  # inactive start of listing
    report = validate_frame(df, H1)
    assert "zero_volume_active" not in _codes(report)
    assert report.stats["zero_volume_rows"] == 10
    assert report.stats["zero_volume_active_rows"] == 0


def test_zero_volume_in_trailing_dead_zone_not_flagged(clean):
    df = clean.copy()
    df.loc[38:, schema.VOLUME] = 0.0
    report = validate_frame(df, H1)
    assert "zero_volume_active" not in _codes(report)


def test_zero_volume_active_fraction_boundary(clean):
    # volumes [1,0,0,0,1]: with window=2 the middle zero sees exactly half of
    # its neighbours trading on each side — flagged at threshold 0.5 (>=),
    # not flagged just above it
    df = clean.head(5).copy()
    df[schema.VOLUME] = [1.0, 0.0, 0.0, 0.0, 1.0]
    at_threshold = validate_frame(df, H1, active_window=2, min_active_fraction=0.5)
    assert at_threshold.stats["zero_volume_active_rows"] == 1
    above_threshold = validate_frame(df, H1, active_window=2, min_active_fraction=0.6)
    assert above_threshold.stats["zero_volume_active_rows"] == 0


def test_empty_frame_is_warning():
    report = validate_frame(schema.empty_frame(), H1)
    assert report.ok
    assert "empty" in _codes(report, "warning")


def test_validate_store_over_fixture_ingest(store):
    ingest(FIXTURES, store)
    reports = validate_store(store)
    assert len(reports) == 4
    assert all(rep.ok for rep in reports)
    assert all(not rep.warnings for rep in reports)
    assert {rep.dataset for rep in reports} == {
        "XBTEUR/1h", "XBTEUR/1d", "ETHEUR/1h", "ETHEUR/1d",
    }
