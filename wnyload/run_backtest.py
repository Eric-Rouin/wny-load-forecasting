"""python -m wnyload.run_backtest  -- build features and run the walk-forward backtest."""
import json

from . import backtest, charts, config, data_nyiso, data_weather, features


def main(test_months: int = 24) -> None:
    load = data_nyiso.load_actuals()
    print(f"Load: {len(load):,} hours, {load.index.min():%Y-%m-%d} to {load.index.max():%Y-%m-%d}")
    try:
        nyiso = data_nyiso.load_nyiso_forecast(days_ahead=1)
        print(f"NYISO day-ahead forecast: {len(nyiso):,} hours")
    except ValueError:
        nyiso = None
        print("NYISO forecast files not found; skipping that baseline")
    weather = data_weather.load_history()

    table = features.build_table(load, weather)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_parquet(config.DATA_PROCESSED / "model_table.parquet")

    print(f"Walk-forward backtest over the last {test_months} months")
    pred = backtest.walk_forward(table, nyiso, test_months=test_months)
    overall, monthly = backtest.summarize(pred)
    backtest.save_report(pred, overall, monthly)
    charts.make_backtest_charts(pred, table)
    print(json.dumps(overall, indent=2))
    print(f"Report written to {config.REPORTS}")


if __name__ == "__main__":
    main()
