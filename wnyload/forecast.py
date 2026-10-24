"""python -m wnyload.forecast [--day YYYY-MM-DD]  -- forecast every hour of one day.

The default target is tomorrow. The model is retrained on everything observed before
the issue time (config.ISSUE_HOUR local time on the day before the target), so the
same command can be pointed at a past day and checked against what actually happened.

Weather for the target day comes from Open-Meteo's forecast API. For a past day the
archive (reanalysis) is used where it exists, which makes that check a bit optimistic.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from . import charts, config, data_nyiso, data_weather
from .backtest import make_model
from .features import FEATURES, build_table

FORECASTS = config.REPORTS / "forecasts"


def issue_time(day: date) -> pd.Timestamp:
    """UTC moment the forecast for `day` is issued: ISSUE_HOUR local on the day before."""
    local = pd.Timestamp(day - timedelta(days=1)) + pd.Timedelta(hours=config.ISSUE_HOUR)
    return local.tz_localize(config.LOCAL_TZ).tz_convert("UTC")


def target_hours(day: date) -> pd.DatetimeIndex:
    """UTC hour-beginnings of every local hour of `day` (23 or 25 on DST change days)."""
    start = pd.Timestamp(day).tz_localize(config.LOCAL_TZ)
    end = pd.Timestamp(day + timedelta(days=1)).tz_localize(config.LOCAL_TZ)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def get_weather() -> pd.DataFrame:
    """Archive weather where it exists, Open-Meteo forecast for the recent past and next days."""
    history = data_weather.load_history()
    recent = data_weather.fetch_forecast(past_days=14, forecast_days=3)
    return history.combine_first(recent)


def refresh_recent_data() -> None:
    """Re-download the current month of NYISO load and forecasts (past months stay cached)."""
    this_month = date.today().replace(day=1).isoformat()
    print("Refreshing NYISO data for the current month")
    data_nyiso.download("palIntegrated", start=this_month)
    data_nyiso.download("isolf", start=this_month)


def forecast_day(day: date, refresh: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Hourly forecast for `day`, plus the full actual-load series it was built from."""
    if refresh:
        refresh_recent_data()
    load = data_nyiso.load_actuals()
    issue = issue_time(day)
    hours = target_hours(day)

    # Only load that had finished by the issue time is visible to the model.
    known = load[load.index + pd.Timedelta(hours=1) <= issue]
    if known.index.max() < issue - pd.Timedelta(hours=1):
        print(f"Warning: load data only runs through {known.index.max():%Y-%m-%d %H:%M} UTC; "
              "the most recent-load features will be missing.")

    weather = get_weather()
    missing = weather.reindex(hours)["temperature_2m"].isna().sum()
    if missing:
        raise SystemExit(f"No weather for {missing} of {len(hours)} target hours; "
                         f"Open-Meteo only forecasts a few days ahead.")

    table = build_table(known.reindex(known.index.union(hours)), weather)
    train = table[table.index < issue].dropna(subset=["load_mw", "temperature_2m"])
    model = make_model()
    model.fit(train[FEATURES], train["load_mw"])

    out = pd.DataFrame(index=hours)
    out["forecast_mw"] = model.predict(table.loc[hours, FEATURES])
    try:
        out["nyiso_forecast_mw"] = data_nyiso.load_nyiso_forecast(days_ahead=1).reindex(hours)
    except ValueError:
        out["nyiso_forecast_mw"] = float("nan")
    out["actual_mw"] = load.reindex(hours)
    out["temperature_f"] = weather.reindex(hours)["temperature_2m"] * 9 / 5 + 32
    print(f"Trained on {len(train):,} hours observed before {issue.tz_convert(config.LOCAL_TZ):%Y-%m-%d %H:%M %Z}")
    return out, load


def save(day: date, out: pd.DataFrame, load: pd.Series) -> None:
    FORECASTS.mkdir(parents=True, exist_ok=True)
    table = out.copy()
    table.insert(0, "hour_local", table.index.tz_convert(config.LOCAL_TZ).strftime("%Y-%m-%d %H:%M %Z"))
    table.round(1).to_csv(FORECASTS / f"forecast_{day}.csv", index_label="hour_utc")
    recent = load[(load.index >= out.index[0] - pd.Timedelta(days=3)) & (load.index < out.index[0])]
    charts.forecast_chart(out, recent, FORECASTS / f"forecast_{day}.png")


def report(day: date, out: pd.DataFrame) -> None:
    local = out.index.tz_convert(config.LOCAL_TZ)
    peak = out["forecast_mw"].idxmax()
    print(f"\nZone A (West) forecast for {day:%A %Y-%m-%d}")
    print(f"  Peak {out['forecast_mw'].max():,.0f} MW at {peak.tz_convert(config.LOCAL_TZ):%H:%M}, "
          f"low {out['forecast_mw'].min():,.0f} MW, total {out['forecast_mw'].sum() / 1000:,.1f} GWh")
    shown = out.set_index(local.strftime("%H:%M")).round(0)
    print(shown.to_string(na_rep="-"))
    for col, name in [("forecast_mw", "Model"), ("nyiso_forecast_mw", "NYISO")]:
        both = out[["actual_mw", col]].dropna()
        if len(both):
            mape = ((both[col] - both["actual_mw"]).abs() / both["actual_mw"]).mean() * 100
            print(f"  {name} error so far: {mape:.2f}% MAPE over {len(both)} hours")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", type=date.fromisoformat, default=date.today() + timedelta(days=1),
                        help="target day, YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--no-refresh", action="store_true", help="skip re-downloading this month's NYISO data")
    args = parser.parse_args()

    out, load = forecast_day(args.day, refresh=not args.no_refresh)
    save(args.day, out, load)
    report(args.day, out)
    print(f"\nSaved to {FORECASTS}")


if __name__ == "__main__":
    main()
