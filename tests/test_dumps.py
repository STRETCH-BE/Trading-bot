from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from tests.conftest import D1, FIXTURES, H1, XBTEUR
from trading_bot.data.dumps import _gdrive_confirm_params, _gdrive_file_id, ingest


def _dump_files() -> list[Path]:
    return sorted(FIXTURES.glob("*_*.csv"))


def test_ingest_from_directory(store):
    results = ingest(FIXTURES, store)
    assert results == {
        ("XBTEUR", "1h"): 48,
        ("XBTEUR", "1d"): 30,
        ("ETHEUR", "1h"): 48,
        ("ETHEUR", "1d"): 30,
    }
    assert len(store.read(XBTEUR, H1)) == 48


def test_ingest_from_zip_with_nested_folders(store, tmp_path):
    archive = tmp_path / "Kraken_OHLCVT_Q1_2024.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in _dump_files():
            zf.write(path, arcname=f"Kraken_OHLCVT/{path.name}")
    results = ingest(archive, store)
    assert len(results) == 4
    assert len(store.read(XBTEUR, D1)) == 30


def test_ingest_from_extracted_directory_with_nested_folder(store, tmp_path):
    """Extracting the quarterly zip leaves CSVs inside a folder — must be found."""
    nested = tmp_path / "Kraken_OHLCVT"
    nested.mkdir()
    for path in _dump_files():
        shutil.copy(path, nested / path.name)
    results = ingest(tmp_path, store)
    assert len(results) == 4
    assert len(store.read(XBTEUR, H1)) == 48


def test_ingest_partial_zip(store, tmp_path):
    """Quarterly archives don't always contain every pair/timeframe file."""
    archive = tmp_path / "partial.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(FIXTURES / "XBTEUR_60.csv", arcname="XBTEUR_60.csv")
    results = ingest(archive, store)
    assert results == {("XBTEUR", "1h"): 48}
    assert store.read(XBTEUR, D1).empty


def test_ingest_single_csv(store, tmp_path):
    target = tmp_path / "XBTEUR_60.csv"
    shutil.copy(FIXTURES / "XBTEUR_60.csv", target)
    results = ingest(target, store)
    assert results == {("XBTEUR", "1h"): 48}


def test_ingest_single_csv_with_unrecognised_name_raises(store, tmp_path):
    import pytest

    target = tmp_path / "xbteur-hourly.csv"
    shutil.copy(FIXTURES / "XBTEUR_60.csv", target)
    with pytest.raises(ValueError, match="does not match any selected dataset"):
        ingest(target, store)


def test_ingest_restricted_selection(store):
    results = ingest(FIXTURES, store, pairs=[XBTEUR], timeframes=[H1])
    assert results == {("XBTEUR", "1h"): 48}


def test_reingest_is_idempotent(store):
    ingest(FIXTURES, store)
    results = ingest(FIXTURES, store)
    assert all(added == 0 for added in results.values())
    assert len(store.read(XBTEUR, H1)) == 48


def test_ingest_empty_directory_yields_nothing(store, tmp_path):
    assert ingest(tmp_path, store) == {}


def test_gdrive_file_id():
    assert (
        _gdrive_file_id("https://drive.google.com/file/d/1AbC-xyz_123/view?usp=sharing")
        == "1AbC-xyz_123"
    )
    assert _gdrive_file_id("https://drive.google.com/uc?export=download&id=XYZ") == "XYZ"
    assert _gdrive_file_id("https://example.com/dump.zip") is None


def test_gdrive_confirm_params():
    html = (
        '<form action="https://drive.usercontent.google.com/download" method="get">'
        '<input type="hidden" name="id" value="XYZ">'
        '<input type="hidden" name="confirm" value="t">'
        '<input type="hidden" name="uuid" value="123-456"></form>'
    )
    params = _gdrive_confirm_params(html)
    assert params["confirm"] == "t"
    assert params["uuid"] == "123-456"
