from __future__ import annotations

import pandas as pd

from tests.conftest import FIXTURES, H1, XBTEUR
from tests.helpers import FakeExchange
from trading_bot.data import cli
from trading_bot.data.dumps import ingest
from trading_bot.data.store import ParquetStore


def test_ingest_then_report(tmp_path, capsys):
    data_dir = tmp_path / "parquet"

    assert cli.main(["ingest", str(FIXTURES), "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "XBTEUR/1h: +48 candles" in out
    assert (data_dir / "XBTEUR" / "1h.parquet").exists()
    assert (data_dir / "ETHEUR" / "1d.parquet").exists()

    assert cli.main(["report", "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "XBTEUR/1h" in out
    assert "2024-01-01 00:00:00" in out
    assert "OK" in out


def test_ingest_selection_flags(tmp_path, capsys):
    data_dir = tmp_path / "parquet"
    code = cli.main(
        ["ingest", str(FIXTURES), "--data-dir", str(data_dir), "--pair", "ETHEUR",
         "--timeframe", "1d"]
    )
    assert code == 0
    assert (data_dir / "ETHEUR" / "1d.parquet").exists()
    assert not (data_dir / "XBTEUR" / "1h.parquet").exists()


def test_ingest_no_match_fails(tmp_path, capsys):
    code = cli.main(["ingest", str(tmp_path), "--data-dir", str(tmp_path / "p")])
    assert code == 1
    assert "no matching dump files" in capsys.readouterr().err


def test_validate_exit_codes(tmp_path, capsys):
    data_dir = tmp_path / "parquet"
    store = ParquetStore(data_dir)
    ingest(FIXTURES, store)
    assert cli.main(["validate", "--data-dir", str(data_dir)]) == 0
    capsys.readouterr()

    # punch a hole in one dataset -> validation must fail
    df = store.read(XBTEUR, H1)
    store.write(XBTEUR, H1, pd.concat([df.iloc[:10], df.iloc[13:]], ignore_index=True))
    assert cli.main(["validate", "--data-dir", str(data_dir)]) == 1
    out = capsys.readouterr().out
    assert "missing" in out


def test_validate_empty_store_fails(tmp_path, capsys):
    assert cli.main(["validate", "--data-dir", str(tmp_path / "nowhere")]) == 1
    assert "store is empty" in capsys.readouterr().out


def test_update_command_gap_exits_nonzero_and_continues(tmp_path, capsys, monkeypatch, api_rows):
    data_dir = tmp_path / "parquet"
    store = ParquetStore(data_dir)
    ingest(FIXTURES, store, pairs=[XBTEUR])  # both 1h and 1d present

    # depth=6 starts the 1h API window at 06:00, beyond the stored 1h tail;
    # the 1d dataset sees an empty fetch and stays untouched
    monkeypatch.setattr(cli, "make_kraken_exchange", lambda: FakeExchange(api_rows, depth=6))
    code = cli.main(["update", "--data-dir", str(data_dir), "--pair", "XBTEUR"])
    assert code == 1
    captured = capsys.readouterr()
    assert "GAP:" in captured.err
    assert "XBTEUR/1d" in captured.out  # the loop went on to the next dataset
    assert len(store.read(XBTEUR, H1)) == 48  # nothing written past the gap


def test_update_command_uses_exchange_factory(tmp_path, capsys, monkeypatch, api_rows):
    data_dir = tmp_path / "parquet"
    store = ParquetStore(data_dir)
    ingest(FIXTURES, store, pairs=[XBTEUR], timeframes=[H1])

    monkeypatch.setattr(cli, "make_kraken_exchange", lambda: FakeExchange(api_rows))
    code = cli.main(
        ["update", "--data-dir", str(data_dir), "--pair", "XBTEUR", "--timeframe", "1h"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "fetched 12, appended 12" in out  # all recorded candles closed long ago
    assert len(store.read(XBTEUR, H1)) == 60
