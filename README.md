# Western New York Day-Ahead Load Forecast

Forecasting hourly electricity demand for **NYISO Zone A (West)**, the Buffalo region,
one day ahead, and comparing the model honestly against simple baselines and NYISO's
own published forecast. The end goal is a self-updating dashboard hosted on a home server
that publishes a forecast every day and tracks its accuracy over time.

## The problem

Grid operators have to match electricity supply to demand every hour. Day-ahead load
forecasts drive which power plants are scheduled, so better forecasts mean lower costs and
fewer reliability problems, especially during heat waves and cold snaps.

## Setup

**Forecast time.** The forecast for every hour of day D is issued at 9:00 AM local time on
day D-1 (`ISSUE_HOUR` in `wnyload/config.py`). Every load-based feature uses only hours that
had finished by then, so the backtest never sees information the model wouldn't have had.

**Data (all public, free)**
| Source | What | Used for |
|---|---|---|
| NYISO `palIntegrated` | Hourly actual load by zone | Target and lag features |
| NYISO `isolf` | NYISO's own load forecast | Benchmark to beat |
| Open-Meteo archive / forecast | Hourly Buffalo (KBUF) weather | Weather features |

**Features:** hour, day of week, season, holidays; load from the same hour 1 day (when
known), 2 days and 7 days earlier, last observed load, mean load two days earlier;
temperature, humidity, dew point, cloud cover, wind, solar radiation, heating and cooling
degree hours, 24-hour mean temperature.

**Validation:** walk-forward. Each month, the model is retrained on everything observed
before that month's first forecast, then forecasts the whole month. Results are reported
over the last 24 months on hours where every model has a forecast.

**Baselines**
- *Seasonal naive:* the same hour one week earlier
- *NYISO:* NYISO's day-ahead forecast for Zone A

## Running it

```bash
pip install -r requirements.txt
python -m wnyload.download        # NYISO load + forecasts + weather, cached in data/raw
python -m wnyload.run_backtest    # features, walk-forward backtest, report in reports/
```

## Results

_To be filled in after the first full backtest:_ MAPE / MAE for each model, monthly error
chart, and notes on where the model misses (storms, holidays, heat waves).

## Known limitations

- **Weather forecast vs. actual weather.** The backtest uses historical (reanalysis)
  weather, but the live system must use weather *forecasts*, which have their own errors.
  Backtest accuracy is therefore somewhat optimistic; the live dashboard measures the
  real number.
- **Issue time vs. NYISO.** NYISO updates its forecast several times a day. The comparison
  uses the forecast file published on day D-1, which may have been issued at a different
  hour than this model's 9 AM issue time.
- **One weather point.** Zone A is large; Buffalo airport weather is a simplification.

## Roadmap

1. First backtest and error analysis
2. Feature and model improvements (hyperparameter tuning, more weather points, solar data from NYSERDA's IEDR)
3. Daily job: fetch data, forecast tomorrow, score yesterday
4. Dashboard (Streamlit or FastAPI) in Docker on a home server, published through Cloudflare Tunnel
