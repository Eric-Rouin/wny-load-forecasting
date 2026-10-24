"""Hourly weather for Buffalo from Open-Meteo (free, no API key).

- archive API: historical reanalysis, used for training and backtests
- forecast API: weather forecasts, used by the live daily job

Caveat worth knowing (and mentioning in interviews): training on observed /
reanalysis weather but forecasting with *forecast* weather makes the backtest
slightly optimistic, because real weather forecasts have their own errors.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests

from . import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _request(url: str, params: dict) -> pd.DataFrame:
    base = {
        "latitude": config.WEATHER_LAT,
        "longitude": config.WEATHER_LON,
        "hourly": ",".join(config.WEATHER_VARS),
        "timezone": "GMT",
    }
    r = requests.get(url, params={**base, **params}, timeout=120)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df.index = pd.to_datetime(df.pop("time")).dt.tz_localize("UTC")
    return df


def download_history(start: str = config.START_DATE, end: date | None = None) -> pd.DataFrame:
    """Fetch history year by year and cache to data/raw/weather_history.parquet."""
    end = end or date.today() - timedelta(days=6)  # archive lags real time by a few days
    out = config.DATA_RAW / "weather_history.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    year_start = pd.Timestamp(start).date()
    while year_start <= end:
        year_end = min(date(year_start.year, 12, 31), end)
        print(f"  weather {year_start} to {year_end}")
        chunks.append(_request(ARCHIVE_URL, {"start_date": str(year_start), "end_date": str(year_end)}))
        year_start = year_end + timedelta(days=1)

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(out)
    return df


def fetch_forecast(past_days: int = 7, forecast_days: int = 3) -> pd.DataFrame:
    """Recent past plus forecast hours, for the live job."""
    return _request(FORECAST_URL, {"past_days": past_days, "forecast_days": forecast_days})


def load_history() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_RAW / "weather_history.parquet")
