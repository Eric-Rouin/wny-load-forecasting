"""Build the modeling table: one row per target hour, features known at issue time.

Leakage rule: the forecast for day D is issued at config.ISSUE_HOUR local time
on day D-1, so every load-based feature must come from hours that had finished
by then. Weather features use the target hour's weather; in production that is
a weather *forecast* (see data_weather.py for the caveat).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

try:
    import holidays
except ImportError:  # optional dependency
    holidays = None

BASE_TEMP_C = 18.3  # 65 F, the usual heating/cooling degree base


def build_table(load: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    idx = pd.date_range(load.index.min(), load.index.max(), freq="h", tz="UTC")
    load = load.reindex(idx)
    df = pd.DataFrame(index=idx)
    df["load_mw"] = load

    local = idx.tz_convert(config.LOCAL_TZ)
    wall = local.tz_localize(None)
    day = wall.normalize()

    # Calendar
    df["hour"] = wall.hour
    df["dow"] = wall.dayofweek
    df["month"] = wall.month
    doy = wall.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_weekend"] = (wall.dayofweek >= 5).astype(int)
    if holidays is not None:
        hol = holidays.country_holidays("US", subdiv="NY", years=range(wall.year.min(), wall.year.max() + 1))
        df["is_holiday"] = pd.Index(wall.date).isin(list(hol.keys())).astype(int)
    else:
        df["is_holiday"] = 0

    # Load lags that are always known at issue time (D-1, ISSUE_HOUR).
    df["load_same_hour_d2"] = load.shift(48)
    df["load_same_hour_d7"] = load.shift(168)

    # Same hour yesterday is only known for hours that finished before issue time.
    known_yesterday = wall.hour <= config.ISSUE_HOUR - 1
    df["load_same_hour_d1"] = load.shift(24).where(known_yesterday)

    # Last observed hour before issue time (D-1 at ISSUE_HOUR-1), and mean of day D-2.
    last_known_wall = day - pd.Timedelta(days=1) + pd.Timedelta(hours=config.ISSUE_HOUR - 1)
    last_known_utc = last_known_wall.tz_localize(config.LOCAL_TZ, nonexistent="shift_forward", ambiguous=True).tz_convert("UTC")
    df["load_last_known"] = load.reindex(last_known_utc).to_numpy()

    daily_mean = load.groupby(local.tz_localize(None).normalize()).mean()
    df["load_mean_d2"] = daily_mean.reindex(day - pd.Timedelta(days=2)).to_numpy()

    # Weather for the target hour
    w = weather.reindex(idx)
    for col in config.WEATHER_VARS:
        if col in w:
            df[col] = w[col]
    if "temperature_2m" in w:
        t = w["temperature_2m"]
        df["hdd"] = (BASE_TEMP_C - t).clip(lower=0)
        df["cdd"] = (t - BASE_TEMP_C).clip(lower=0)
        df["temp_mean_24h"] = t.rolling(24, min_periods=12).mean()  # buildings respond slowly
        df["temp_x_hour_cdd"] = df["cdd"] * (df["hour"].between(12, 20)).astype(int)

    return df


FEATURES = [
    "hour", "dow", "month", "doy_sin", "doy_cos", "is_weekend", "is_holiday",
    "load_same_hour_d1", "load_same_hour_d2", "load_same_hour_d7",
    "load_last_known", "load_mean_d2",
    *config.WEATHER_VARS, "hdd", "cdd", "temp_mean_24h", "temp_x_hour_cdd",
]
