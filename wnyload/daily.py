"""python -m wnyload.daily  -- the once-a-day job: forecast tomorrow, score past forecasts.

Run it every morning shortly after config.ISSUE_HOUR (local time). Each run
  1. downloads new NYISO load and forecasts and new weather (everything, on a first run),
  2. forecasts every hour of tomorrow into reports/live/forecasts/,
  3. scores every live forecast whose day is complete in the actual-load data, rebuilding
     reports/live/scores.csv, so days missed while the machine was off are scored later,
  4. appends one line to reports/live/runs.csv and its full output to reports/live/daily.log.

Only this job writes to reports/live/. Its forecasts were issued the day before with
weather *forecasts*, so their scores are the model's real accuracy. Backfills made with
`python -m wnyload.forecast --day ...` stay in reports/forecasts/.

Exits with status 1 if any step failed, so a scheduler can flag the run.
"""
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, data_nyiso
from .forecast import forecast_day, report, save

LIVE = config.REPORTS / "live"
LIVE_FORECASTS = LIVE / "forecasts"
SCORES = LIVE / "scores.csv"
RUNS = LIVE / "runs.csv"
LOG = LIVE / "daily.log"

MODELS = {"model": "forecast_mw", "nyiso": "nyiso_forecast_mw"}
RUN_FIELDS = ["run_at", "target_day", "status", "load_through", "train_hours",
              "forecast_peak_mw", "days_scored", "error"]


def score_forecasts(load: pd.Series, folder=LIVE_FORECASTS) -> pd.DataFrame:
    """One row per forecast day whose every hour has an actual load value."""
    try:
        nyiso_latest = data_nyiso.load_nyiso_forecast(days_ahead=1)
    except ValueError:
        nyiso_latest = pd.Series(dtype=float)

    rows = []
    for path in sorted(folder.glob("forecast_*.csv")):
        day = date.fromisoformat(path.stem.removeprefix("forecast_"))
        f = pd.read_csv(path, index_col="hour_utc")
        f.index = pd.to_datetime(f.index, utc=True)
        actual = load.reindex(f.index)
        if actual.isna().any():
            continue  # day not over yet, or NYISO hasn't posted all of it
        # NYISO's forecast as it stood when ours was issued; if it wasn't posted by then,
        # fall back to the day-ahead file downloaded since.
        f["nyiso_forecast_mw"] = f["nyiso_forecast_mw"].fillna(nyiso_latest.reindex(f.index))

        row = {"day": day, "hours": len(f), "actual_peak_mw": actual.max()}
        for name, col in MODELS.items():
            err = f[col] - actual
            complete = err.notna().all()
            row[f"{name}_mape_pct"] = (err.abs() / actual).mean() * 100 if complete else float("nan")
            row[f"{name}_mae_mw"] = err.abs().mean() if complete else float("nan")
            row[f"{name}_bias_mw"] = err.mean() if complete else float("nan")
            row[f"{name}_peak_mw"] = f[col].max()
        rows.append(row)

    columns = ["day", "hours", "actual_peak_mw"] + [
        f"{name}_{stat}" for name in MODELS for stat in ("mape_pct", "mae_mw", "bias_mw", "peak_mw")]
    return pd.DataFrame(rows, columns=columns)


def print_score_summary(scores: pd.DataFrame) -> None:
    if scores.empty:
        print("No completed live forecasts to score yet.")
        return
    last = scores.iloc[-1]
    print(f"Latest scored day {last['day']}: model {last['model_mape_pct']:.2f}% MAPE, "
          f"NYISO {last['nyiso_mape_pct']:.2f}%")
    for days in (7, 30):
        if days > 7 and len(scores) <= 7:
            break  # the 7-day line already covers every scored day
        recent = scores.tail(days)
        print(f"  Last {len(recent)} scored days: model {recent['model_mape_pct'].mean():.2f}%, "
              f"NYISO {recent['nyiso_mape_pct'].mean():.2f}% MAPE")


def append_run(row: dict) -> None:
    new = not RUNS.exists()
    with RUNS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RUN_FIELDS})


def run(refresh: bool = True) -> int:
    now = datetime.now(ZoneInfo(config.LOCAL_TZ))
    target = config.local_today() + timedelta(days=1)
    record = {"run_at": f"{now:%Y-%m-%d %H:%M:%S %Z}", "target_day": target}
    errors = []
    print(f"\n=== Daily run {record['run_at']}: forecasting {target} ===")
    if now.hour < config.ISSUE_HOUR:
        print(f"Warning: running before the {config.ISSUE_HOUR}:00 issue time, so the most recent "
              "load hours aren't available yet.")

    load = None
    try:
        out, load = forecast_day(target, refresh=refresh)
        save(target, out, load, LIVE_FORECASTS)
        report(target, out)
        record["train_hours"] = out.attrs["train_hours"]
        record["load_through"] = f"{out.attrs['load_through'].tz_convert(config.LOCAL_TZ):%Y-%m-%d %H:%M %Z}"
        record["forecast_peak_mw"] = round(out["forecast_mw"].max())
    except (Exception, SystemExit) as exc:  # forecast_day exits when weather is missing
        traceback.print_exc()
        errors.append(f"forecast: {exc}")

    try:
        if load is None:
            load = data_nyiso.load_actuals()
        scores = score_forecasts(load)
        scores.round(2).to_csv(SCORES, index=False)
        record["days_scored"] = len(scores)
        print()
        print_score_summary(scores)
    except Exception as exc:
        traceback.print_exc()
        errors.append(f"scoring: {exc}")

    record["status"] = "error" if errors else "ok"
    record["error"] = "; ".join(errors)
    append_run(record)
    print(f"Run {record['status']}" + (f": {record['error']}" if errors else ""))
    return 1 if errors else 0


class _Tee:
    """Write to the console and the log file at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-refresh", action="store_true", help="skip downloading new data (for testing)")
    args = parser.parse_args()

    LIVE_FORECASTS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        out, err = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
        with redirect_stdout(out), redirect_stderr(err):
            code = run(refresh=not args.no_refresh)
    sys.exit(code)


if __name__ == "__main__":
    main()
