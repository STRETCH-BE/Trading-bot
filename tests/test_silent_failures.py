"""One test per audit finding, proving the failure is now loud.

Governing principle under test: NO OPERATION MAY REPORT SUCCESS WHEN IT DID
LESS THAN ASKED. Each of these previously returned a clean exit code while
quietly doing nothing, or less than nothing.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import pandas as pd
import pytest

from tests.helpers import FakeExchange
from trading_bot.backtest import BacktestConfig, backtest, compute_metrics
from trading_bot.backtest.config import ConfigNotFoundError
from trading_bot.backtest.report import format_report
from trading_bot.data import schema
from trading_bot.data.dumps import DownloadError, PartialIngestError, download_dump, ingest
from trading_bot.data.normalize import from_ccxt
from trading_bot.data.store import ParquetStore
from trading_bot.data.update import ClockDriftError, update
from trading_bot.risk.limits import RiskLimits
from trading_bot.strategies.donchian import DonchianParams
from trading_bot.strategies.voltrend import VolTrendParams

from .conftest import FIXTURES, H1, XBTEUR
from .synthetic import flat_candles

TS = pd.Timestamp("2024-01-03 00:00:00", tz="UTC")


# --- FINDING 5: clock drift --------------------------------------------------


def test_finding_5_clock_drift_raises_with_the_computed_delta(store, api_rows):
    """Was: fetched=12 appended=0 warnings=0. Now: raises with the drift."""
    from trading_bot.data.normalize import read_dump_csv

    store.write(XBTEUR, H1, read_dump_csv(str(FIXTURES / "XBTEUR_60.csv")))
    behind = datetime(2024, 1, 2, 23, 30, tzinfo=UTC)  # ~12h behind the fixture

    with pytest.raises(ClockDriftError) as exc:
        update(store, FakeExchange(api_rows), XBTEUR, H1, now=behind)

    message = str(exc.value)
    assert "dated AFTER the local clock" in message
    assert "11.5h" in message or "behind by at least" in message
    assert "chrony" in message  # tells the operator what to fix


def test_finding_5_a_correct_clock_still_works(store, api_rows):
    """Control: the guard must not fire during normal operation."""
    from trading_bot.data.normalize import read_dump_csv

    store.write(XBTEUR, H1, read_dump_csv(str(FIXTURES / "XBTEUR_60.csv")))
    result = update(
        store, FakeExchange(api_rows), XBTEUR, H1,
        now=datetime(2024, 1, 3, 11, 30, tzinfo=UTC),
    )
    assert result.appended == 11


def test_finding_5_fetched_but_stored_nothing_is_warned(store, api_rows):
    """Legitimate no-op, but it must appear in the warnings, not vanish."""
    from trading_bot.data.normalize import read_dump_csv

    store.write(XBTEUR, H1, read_dump_csv(str(FIXTURES / "XBTEUR_60.csv")))
    now = datetime(2024, 1, 3, 0, 30, tzinfo=UTC)
    available = [r for r in api_rows if r[0] <= now.timestamp() * 1000]
    result = update(store, FakeExchange(available), XBTEUR, H1, now=now)

    assert result.fetched > 0
    assert result.appended == 0
    assert any("stored none" in w for w in result.warnings)


# --- FINDING 6: config path --------------------------------------------------


@pytest.mark.parametrize(
    "loader",
    [BacktestConfig.load, DonchianParams.load, VolTrendParams.load, RiskLimits.load],
)
def test_finding_6_missing_config_path_raises(loader, tmp_path):
    """Was: silently returned defaults, so a typo ran different economics."""
    with pytest.raises(ConfigNotFoundError, match="config file not found"):
        loader(tmp_path / "typo.yaml")


def test_finding_6_defaults_are_still_reachable_deliberately():
    assert BacktestConfig.defaults() == BacktestConfig()


def test_finding_6_startup_banner_shows_the_numbers_in_use():
    banner = BacktestConfig.load("config.yaml").banner("config.yaml")
    assert "RESOLVED CONFIGURATION" in banner
    assert "config.yaml" in banner
    assert "26.0 bps" in banner  # taker fee
    assert "slippage" in banner
    assert "effective fee rate" in banner
    assert "holdout start" in banner


# --- FINDING 8: partial ingest -----------------------------------------------


def test_finding_8_partial_ingest_fails(store, tmp_path):
    """Was: exit 0 with three of four datasets silently absent."""
    archive = tmp_path / "partial.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(FIXTURES / "XBTEUR_60.csv", arcname="XBTEUR_60.csv")

    with pytest.raises(PartialIngestError, match="3 of 4 requested"):
        ingest(archive, store)


def test_finding_8_allow_partial_is_an_explicit_opt_in(store, tmp_path):
    archive = tmp_path / "partial.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(FIXTURES / "XBTEUR_60.csv", arcname="XBTEUR_60.csv")
    assert ingest(archive, store, allow_partial=True) == {("XBTEUR", "1h"): 48}


# --- FINDING 9: pagination cap -----------------------------------------------


def test_finding_9_page_cap_warns_loudly(store, monkeypatch, api_rows):
    """Was: stopped at the cap and returned partial data as if complete."""
    from trading_bot.data import update as update_mod

    monkeypatch.setattr(update_mod, "_MAX_PAGES", 2)
    monkeypatch.setattr(update_mod, "KRAKEN_OHLC_LIMIT", 2)

    # an exchange with plenty of pages left when the cap hits
    long_rows = [
        [int((TS + pd.Timedelta(hours=i)).timestamp() * 1000), 1.0, 2.0, 0.5, 1.5, 3.0]
        for i in range(50)
    ]
    result = update_mod.update(
        store, FakeExchange(long_rows), XBTEUR, H1,
        now=datetime(2024, 1, 10, tzinfo=UTC),
    )
    assert any("safety cap" in w and "INCOMPLETE" in w for w in result.warnings)


# --- FINDING 10: download validation -----------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = {"content-type": "application/zip"}

    def raise_for_status(self): ...
    def iter_content(self, n): yield self._body
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeSession:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def get(self, url, **kw):
        return _FakeResponse(self._body)


def test_finding_10_html_error_page_is_not_written_as_a_zip(tmp_path):
    """Was: a Drive interstitial was saved as .zip and reported success."""
    dest = tmp_path / "dump.zip"
    html = b"<!DOCTYPE html><html><body>Google Drive can't scan this file</body></html>"

    with pytest.raises(DownloadError, match="not a zip archive"):
        download_dump("https://example.com/x.zip", dest, session=_FakeSession(html))

    assert not dest.exists(), "a bad download must leave no file behind"


def test_finding_10_a_real_zip_is_written(tmp_path):
    dest = tmp_path / "ok.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("XBTEUR_60.csv", "1704067200,1,2,0.5,1.5,3,7\n")
    download_dump("https://example.com/x.zip", dest, session=_FakeSession(buf.getvalue()))
    assert dest.exists() and zipfile.is_zipfile(dest)


# --- FINDING 11: skipped orders ----------------------------------------------


def test_finding_11_excessive_skips_are_surfaced_in_the_report():
    """Was: a counter nobody reads while the strategy sat flat forever."""
    candles = flat_candles(60, price=100_000.0)  # ordermin unaffordable at this size
    config = BacktestConfig(starting_capital=100.0, min_rebalance_delta=0.0)
    signals = pd.Series([float(i % 2) * 0.001 for i in range(60)])

    result = backtest(candles, lambda c: signals, config, pair="XBTEUR")
    metrics = compute_metrics(result, config)

    assert result.skipped_orders > 0
    assert metrics.skipped_orders_excessive
    assert metrics.skipped_order_fraction == 1.0
    report = format_report(metrics)
    assert "ORDERS ARE BEING SKIPPED" in report
    assert "NOT the strategy you validated" in report


def test_finding_11_normal_runs_do_not_warn():
    candles = flat_candles(60, price=100.0)
    config = BacktestConfig(starting_capital=10_000.0)
    signals = pd.Series([1.0] * 60)
    metrics = compute_metrics(
        backtest(candles, lambda c: signals, config, pair="XBTEUR"), config
    )
    assert not metrics.skipped_orders_excessive
    assert "ORDERS ARE BEING SKIPPED" not in format_report(metrics)


# --- FINDING 13: vwap and count ----------------------------------------------


def test_finding_13_vwap_and_count_survive_the_ccxt_path():
    """Was: 8-column Kraken rows truncated to 6, discarding both."""
    rows = [[1704067200, 40000.0, 40500.0, 39800.0, 40250.0, 40133.77, 123.456, 987]]
    out = from_ccxt(rows)

    assert out[schema.VWAP].iloc[0] == 40133.77  # the pagination cross-check
    assert out[schema.TRADES].iloc[0] == 987  # the liquidity measure
    assert out[schema.VOLUME].iloc[0] == 123.456  # not overwritten by vwap


def test_finding_13_six_column_ccxt_rows_still_work():
    out = from_ccxt([[1704067200000, 1.0, 2.0, 0.5, 1.5, 3.0]])
    assert pd.isna(out[schema.VWAP].iloc[0])  # genuinely absent, not invented
    assert out[schema.VOLUME].iloc[0] == 3.0


def test_finding_13_count_feeds_the_liquidity_floor(store):
    """count is not decorative: the Stage 4 liquidity floor is computed from it."""
    from trading_bot.data.liquidity import FloorRule, daily_metrics

    rows = [
        [int((TS + pd.Timedelta(days=i)).timestamp()), 1.0, 2.0, 0.5, 1.5, 1.2, 3.0, 900]
        for i in range(40)
    ]
    df = from_ccxt(rows)
    metrics = daily_metrics(df, FloorRule(min_median_trades=500.0))
    assert metrics["roll_median"].dropna().iloc[-1] == 900.0


def test_all_findings_have_a_named_test():
    """Meta-guard: every finding fixed in this batch is represented here."""
    import pathlib

    text = pathlib.Path(__file__).read_text()
    for finding in (5, 6, 8, 9, 10, 11, 13):
        assert f"FINDING {finding}:" in text, f"finding {finding} lost its test"


def test_store_fixture_is_used(store):
    assert isinstance(store, ParquetStore)
