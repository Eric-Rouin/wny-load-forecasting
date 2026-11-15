"""Download and parse NYISO public load data.

Datasets (all public, no account needed):
  palIntegrated  hourly integrated actual load by zone
  isolf          NYISO's own load forecast (each daily file forecasts ~7 days ahead)

Monthly archives live at  http://mis.nyiso.com/public/csv/<ds>/<YYYYMM01><ds>_csv.zip
Recent daily files are also posted to NYISO's OASIS site, used as a fallback.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

from . import config

MIS_BASE = "http://mis.nyiso.com/public/csv"
OASIS_BASE = "https://oasis-postings.prod.nysdi.com"
OASIS_FOLDERS = {"palIntegrated": "INT_RT_ACT_LOAD", "isolf": "ISO_LF"}
HEADERS = {"User-Agent": "wny-load-forecast (portfolio project)"}


def _get(url: str, retries: int = 3) -> bytes | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def _months(start: date, end: date):
    d = date(start.year, start.month, 1)
    while d <= end:
        yield d
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def download(dataset: str, start: str = config.START_DATE, end: date | None = None,
             refresh_months: int = 1) -> None:
    """Cache one zip per month in data/raw/<dataset>/.

    The last `refresh_months` months (counting the current one) are downloaded again every
    time; older months only if they are missing. Refreshing the previous month as well
    completes a month whose file was first saved part-way through it.
    """
    end = end or config.local_today()
    out_dir = config.DATA_RAW / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    months = list(_months(pd.Timestamp(start).date(), end))
    recent = set(months[-refresh_months:]) if refresh_months > 0 else set()

    for m in months:
        path = out_dir / f"{m:%Y%m}.zip"
        if path.exists() and m not in recent:
            continue
        content = _get(f"{MIS_BASE}/{dataset}/{m:%Y%m01}{dataset}_csv.zip")
        if content is None:
            content = _zip_from_daily_files(dataset, m, end)
        if content is None:
            print(f"  {dataset} {m:%Y-%m}: not available")
            continue
        path.write_bytes(content)
        print(f"  {dataset} {m:%Y-%m}: saved")


def _zip_from_daily_files(dataset: str, month: date, end: date) -> bytes | None:
    """Fallback: build a month zip from individual daily CSVs (MIS, then OASIS)."""
    buf = io.BytesIO()
    found = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        d = month
        while d.month == month.month and d <= end:
            name = f"{d:%Y%m%d}{dataset}.csv"
            content = _get(f"{MIS_BASE}/{dataset}/{name}", retries=1)
            if content is None and dataset in OASIS_FOLDERS:
                content = _get(
                    f"{OASIS_BASE}/{OASIS_FOLDERS[dataset]}/CSV/{d:%Y/%m/%d}/{dataset}-{d:%Y%m%d}.csv",
                    retries=1,
                )
            if content is not None:
                zf.writestr(name, content)
                found += 1
            d += timedelta(days=1)
    return buf.getvalue() if found else None


def _read_zip_csvs(dataset: str) -> list[tuple[str, pd.DataFrame]]:
    frames = []
    for path in sorted((config.DATA_RAW / dataset).glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".csv"):
                    frames.append((name, pd.read_csv(zf.open(name))))
    return frames


def _to_utc(stamps: pd.Series, tz_labels: pd.Series | None) -> pd.Series:
    """NYISO stamps are local prevailing time; use the EST/EDT label when present."""
    naive = pd.to_datetime(stamps, format="mixed")
    if tz_labels is not None:
        offset = tz_labels.str.upper().map({"EST": 5, "EDT": 4})
        if offset.notna().all():
            return (naive + pd.to_timedelta(offset, unit="h")).dt.tz_localize("UTC")
    try:
        local = naive.dt.tz_localize(config.LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")
    except ValueError:  # fall-back day without a repeated 1 AM row
        local = naive.dt.tz_localize(config.LOCAL_TZ, ambiguous=True, nonexistent="shift_forward")
    return local.dt.tz_convert("UTC")


def load_actuals() -> pd.Series:
    """Hourly actual load for the zone (MW), indexed by UTC hour-beginning."""
    parts = []
    for _, df in _read_zip_csvs("palIntegrated"):
        df = df[df["Name"].str.upper() == config.ZONE_NAME_LOAD]
        if df.empty:
            continue
        ts = _to_utc(df["Time Stamp"], df.get("Time Zone"))
        parts.append(pd.Series(df["Integrated Load"].to_numpy(), index=ts.to_numpy()))
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = pd.DatetimeIndex(s.index, tz="UTC").floor("h")
    s.name = "load_mw"
    return s.groupby(level=0).mean()


def load_nyiso_forecast(days_ahead: int = 1) -> pd.Series:
    """NYISO's zone forecast issued `days_ahead` days before each target day."""
    parts = []
    for name, df in _read_zip_csvs("isolf"):
        issue_day = pd.Timestamp(name[:8]).date()
        col = next((c for c in df.columns if c.strip().lower() == config.ZONE_NAME_FORECAST.lower()), None)
        if col is None:
            continue
        local = pd.to_datetime(df["Time Stamp"], format="mixed")
        keep = local.dt.date == issue_day + timedelta(days=days_ahead)
        if not keep.any():
            continue
        ts = _to_utc(df.loc[keep, "Time Stamp"], None)
        parts.append(pd.Series(df.loc[keep, col].to_numpy(), index=ts.to_numpy()))
    s = pd.concat(parts).sort_index()
    s.index = pd.DatetimeIndex(s.index, tz="UTC").floor("h")
    s = s[~s.index.duplicated(keep="last")]
    s.name = "nyiso_forecast_mw"
    return s
