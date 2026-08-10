from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_bot.data import schema
from trading_bot.data.store import ParquetStore

FIXTURES = Path(__file__).parent / "fixtures"

XBTEUR = schema.PAIRS["XBTEUR"]
ETHEUR = schema.PAIRS["ETHEUR"]
H1 = schema.TIMEFRAMES["1h"]
D1 = schema.TIMEFRAMES["1d"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Hard guarantee that no test opens a network connection."""
    import socket

    def guard(self, *args, **kwargs):
        raise RuntimeError("network access attempted from a test")

    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "parquet")


@pytest.fixture
def api_rows() -> list[list[float]]:
    """Recorded-format ccxt fetch_ohlcv rows continuing after XBTEUR_60.csv."""
    return json.loads((FIXTURES / "ccxt_kraken_xbteur_1h.json").read_text())
