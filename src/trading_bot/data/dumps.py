"""Ingest Kraken's downloadable historical OHLCVT dumps.

Kraken publishes quarterly zip archives of per-pair CSVs (no header;
``timestamp,open,high,low,close,volume,trades``; interval encoded in the file
name in minutes, e.g. ``XBTEUR_60.csv``). The archives are distributed via
Google Drive links from:
https://support.kraken.com/articles/360047124832

Because the Drive links rotate every quarter, ``download_dump`` takes the URL
as an argument; ``ingest`` accepts the downloaded zip, an extracted directory,
or a single CSV.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import requests

from trading_bot.data import schema
from trading_bot.data.normalize import read_dump_csv
from trading_bot.data.store import ParquetStore

_CHUNK = 1 << 20

_GDRIVE_ID_PATTERNS = (
    re.compile(r"drive\.google\.com/file/d/([\w-]+)"),
    re.compile(r"drive\.google\.com/uc\?.*\bid=([\w-]+)"),
)


def download_dump(url: str, dest: str | Path, *, session: requests.Session | None = None) -> Path:
    """Download a dump archive to ``dest``. Understands Google Drive share links.

    For Drive links this handles the "can't scan for viruses" confirmation
    page that Drive returns for large files.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()

    file_id = _gdrive_file_id(url)
    if file_id:
        url = "https://drive.usercontent.google.com/download"
        params: dict[str, str] = {"id": file_id, "export": "download"}
        resp = sess.get(url, params=params, stream=True, timeout=60)
        resp.raise_for_status()
        if "text/html" in resp.headers.get("content-type", ""):
            # Large-file confirmation page: re-submit with the tokens it embeds.
            params |= _gdrive_confirm_params(resp.text)
            resp = sess.get(url, params=params, stream=True, timeout=60)
            resp.raise_for_status()
    else:
        resp = sess.get(url, stream=True, timeout=60)
        resp.raise_for_status()

    with resp, open(dest, "wb") as fh:
        for chunk in resp.iter_content(_CHUNK):
            fh.write(chunk)
    return dest


def _gdrive_file_id(url: str) -> str | None:
    for pattern in _GDRIVE_ID_PATTERNS:
        if match := pattern.search(url):
            return match.group(1)
    return None


def _gdrive_confirm_params(html: str) -> dict[str, str]:
    params = dict(re.findall(r'name="(\w+)"\s+value="([^"]*)"', html))
    params.setdefault("confirm", "t")
    return params


def ingest(
    source: str | Path,
    store: ParquetStore,
    pairs: Iterable[schema.Pair] = (),
    timeframes: Iterable[schema.Timeframe] = (),
) -> dict[tuple[str, str], int]:
    """Ingest dump CSVs from ``source`` (zip archive, directory, or single CSV).

    Returns ``{(pair, timeframe): rows_added}`` for every dataset found.
    Datasets not present in the source are skipped silently — Kraken ships
    one archive per quarter, and not every quarter contains every file.
    """
    pairs = list(pairs) or list(schema.PAIRS.values())
    timeframes = list(timeframes) or list(schema.TIMEFRAMES.values())
    source = Path(source)
    results: dict[tuple[str, str], int] = {}

    is_zip = source.is_file() and zipfile.is_zipfile(source)
    for pair in pairs:
        for timeframe in timeframes:
            name = timeframe.dump_filename(pair)
            df = None
            if source.is_dir():
                # extracted archives keep their internal folder, so search deep
                candidate = next(iter(sorted(source.rglob(name))), None)
                if candidate is not None:
                    df = read_dump_csv(str(candidate))
            elif is_zip:
                with zipfile.ZipFile(source) as zf:
                    member = _find_member(zf, name)
                    if member:
                        with zf.open(member) as fh:
                            df = read_dump_csv(fh)
            elif source.name == name:
                df = read_dump_csv(str(source))

            if df is not None:
                added = store.upsert(pair, timeframe, df)
                results[(pair.kraken_name, timeframe.name)] = added

    if not results and source.is_file() and not is_zip:
        expected = ", ".join(
            tf.dump_filename(p) for p in pairs for tf in timeframes
        )
        raise ValueError(
            f"{source.name} does not match any selected dataset "
            f"(expected one of: {expected})"
        )
    return results


def _find_member(zf: zipfile.ZipFile, name: str) -> str | None:
    """Find ``name`` in the archive regardless of internal folder structure."""
    for member in zf.namelist():
        if member.rsplit("/", 1)[-1] == name:
            return member
    return None
