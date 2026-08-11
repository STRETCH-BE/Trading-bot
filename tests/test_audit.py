"""Data-quality audit, and the guarantee that nothing is ever filled in."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pandas as pd
import pytest

from trading_bot.data import schema
from trading_bot.data.audit import audit_frame, audit_store, render
from trading_bot.data.dumps import ingest
from trading_bot.data.normalize import read_dump_csv

from .conftest import D1, FIXTURES, H1, XBTEUR
from .synthetic import make_candles

NOW = datetime(2024, 2, 1, tzinfo=UTC)


def _hourly(n: int) -> pd.DataFrame:
    return make_candles(
        [(100.0, 101.0, 99.0, 100.5, 5.0)] * n, step=pd.Timedelta(hours=1)
    )


def test_clean_frame_audits_clean():
    a = audit_frame(_hourly(48), H1, dataset="X/1h", now=NOW)
    assert a.rows == 48
    assert a.expected_rows == 48
    assert a.missing_intervals == 0
    assert a.gap_runs == []
    assert a.integrity_ok
    assert a.coverage_pct == 100.0
    assert a.filled_rows == 0


def test_missing_intervals_are_counted_and_located():
    df = _hourly(48)
    holed = pd.concat([df.iloc[:10], df.iloc[13:]], ignore_index=True)
    a = audit_frame(holed, H1, dataset="X/1h", now=NOW)

    assert a.rows == 45
    assert a.expected_rows == 48
    assert a.missing_intervals == 3
    assert len(a.gap_runs) == 1
    assert a.gap_runs[0].missing == 3
    assert a.coverage_pct == pytest.approx(93.75)


def test_gap_runs_are_reported_longest_first():
    df = _hourly(100)
    holed = pd.concat(
        [df.iloc[:10], df.iloc[12:40], df.iloc[48:]], ignore_index=True
    )
    a = audit_frame(holed, H1, dataset="X/1h", now=NOW)
    assert a.missing_intervals == 10

    # parse the rendered gap-run block rather than substring-hunting the page
    lines = render([a], top_gaps=10).splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("missing")) + 2
    counts = [int(ln.split()[0]) for ln in lines[start:] if ln.strip()]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 8


@pytest.mark.parametrize(
    "column, value, field",
    [
        (schema.HIGH, 1.0, "high_lt_low"),
        (schema.CLOSE, 500.0, "close_out_of_range"),
        (schema.OPEN, 500.0, "open_out_of_range"),
        (schema.VOLUME, -1.0, "negative_volume"),
    ],
)
def test_integrity_violations_are_detected(column, value, field):
    """One violation at a time — overlapping breakages mask each other."""
    df = _hourly(10)
    df.loc[4, column] = value
    a = audit_frame(df, H1, dataset="X/1h", now=NOW)

    assert getattr(a, field) >= 1
    assert not a.integrity_ok


def test_non_positive_price_is_detected():
    df = _hourly(10)
    df.loc[3, [schema.OPEN, schema.HIGH, schema.LOW, schema.CLOSE]] = 0.0
    a = audit_frame(df, H1, dataset="X/1h", now=NOW)
    assert a.non_positive_price == 1
    assert not a.integrity_ok


def test_clean_frame_reports_integrity_ok():
    assert audit_frame(_hourly(10), H1, dataset="X/1h", now=NOW).integrity_ok


def test_staleness_is_measured_from_the_last_candle():
    df = _hourly(24)  # starts 2024-01-01 00:00, ends 2024-01-01 23:00
    a = audit_frame(df, H1, dataset="X/1h", now=datetime(2024, 1, 11, 23, tzinfo=UTC))
    assert a.staleness == pd.Timedelta(days=10)


def test_source_counts_split_dump_and_rest():
    df = _hourly(10)
    df.loc[5:, schema.SOURCE] = schema.SOURCE_REST
    a = audit_frame(df, H1, dataset="X/1h", now=NOW)
    assert a.source_counts == {"dump": 5, "rest": 5}


def test_empty_dataset_audits_without_crashing():
    a = audit_frame(schema.empty_frame(), H1, dataset="X/1h", now=NOW)
    assert a.rows == 0
    assert a.first is None


# --- the no-fill guarantee ---------------------------------------------------


def test_ingest_never_invents_a_candle(store, tmp_path):
    """Row count out must equal row count in, exactly. No gap is ever filled."""
    csv = tmp_path / "XBTEUR_60.csv"
    base = 1704067200
    # deliberately skip hours 3, 4, 5 and 9
    kept = [0, 1, 2, 6, 7, 8, 10, 11]
    csv.write_text(
        "".join(f"{base + h * 3600},100,101,99,100.5,5.0,7\n" for h in kept)
    )

    ingest(csv, store, allow_partial=True)
    out = store.read(XBTEUR, H1)

    assert len(out) == len(kept), "ingest changed the row count"
    expected_ts = {
        pd.Timestamp(base + h * 3600, unit="s", tz="UTC") for h in kept
    }
    assert set(out[schema.TIMESTAMP]) == expected_ts, "a timestamp was invented"

    a = audit_frame(out, H1, dataset="XBTEUR/1h", now=NOW)
    assert a.missing_intervals == 4  # 3,4,5 and 9 — reported, not filled
    assert a.filled_rows == 0


def test_no_forward_fill_or_interpolation_in_the_data_package():
    """Structural: the ingest path must contain no filling primitives at all."""
    import pathlib

    import trading_bot.data as data_pkg

    root = pathlib.Path(data_pkg.__file__).parent
    # Call sites, not words: prose about interpolation is fine, calling it is not.
    banned = (
        ".ffill(", ".bfill(", ".backfill(", ".interpolate(",
        ".resample(", ".asfreq(", ".reindex(",
        'method="ffill"', "method='ffill'",
        'method="bfill"', "method='bfill'",
        'method="pad"', "method='pad'",
    )
    offenders = []
    for path in root.glob("*.py"):
        if path.name == "data_quality_only.py":
            continue  # audited separately; not on the ingest path
        text = path.read_text()
        offenders += [f"{path.name}:{tok}" for tok in banned if tok in text]
    assert not offenders, f"filling primitive found on the data path: {offenders}"


def test_the_no_fill_check_would_catch_a_real_fill(tmp_path):
    """Control: the detector must fire on an actual filling call."""
    offender = tmp_path / "bad.py"
    offender.write_text("import pandas as pd\ndef f(df):\n    return df.ffill()\n")
    banned = (".ffill(", ".interpolate(")
    assert any(tok in offender.read_text() for tok in banned)


def test_fixture_ingest_round_trips_row_for_row(store):
    """Every fixture CSV's line count must survive ingestion unchanged."""
    ingest(FIXTURES, store, allow_partial=True)
    for pair, tf, name in [
        (XBTEUR, H1, "XBTEUR_60.csv"),
        (XBTEUR, D1, "XBTEUR_1440.csv"),
    ]:
        source_rows = len(read_dump_csv(str(FIXTURES / name)))
        assert len(store.read(pair, tf)) == source_rows


def test_audit_render_states_zero_filled_rows():
    a = audit_frame(_hourly(24), H1, dataset="X/1h", now=NOW)
    assert "rows filled/interpolated: 0" in render([a])


def test_audit_store_covers_every_dataset(store):
    ingest(FIXTURES, store, allow_partial=True)
    assert len(audit_store(store)) == 4


def test_dump_rows_carry_null_vwap_after_ingest(store):
    ingest(FIXTURES, store, allow_partial=True)
    df = store.read(XBTEUR, H1)
    assert df[schema.VWAP].isna().all()
    assert (df[schema.SOURCE] == "dump").all()


def test_read_dump_csv_of_a_gapped_file_keeps_the_gap():
    csv = io.StringIO("1704067200,1,2,0.5,1.5,3,7\n1704074400,1,2,0.5,1.5,3,7\n")
    df = read_dump_csv(csv)
    assert len(df) == 2
    delta = df[schema.TIMESTAMP].iloc[1] - df[schema.TIMESTAMP].iloc[0]
    assert delta == pd.Timedelta(hours=2)  # the missing 01:00 stays missing
