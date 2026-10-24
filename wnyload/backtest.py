"""Walk-forward backtest: retrain each month, forecast the next month, compare to baselines.

Models compared on exactly the same hours:
  seasonal_naive   same hour one week earlier
  nyiso            NYISO's own forecast issued the day before
  gbm              gradient boosting on features.FEATURES
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .features import FEATURES

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
from sklearn.ensemble import HistGradientBoostingRegressor


def make_model():
    if LGBMRegressor is not None:
        return LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                             min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=0, verbose=-1)
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                         min_samples_leaf=40, random_state=0)


def metrics(y: pd.Series, p: pd.Series) -> dict:
    err = p - y
    return {
        "MAE_MW": float(np.abs(err).mean()),
        "RMSE_MW": float(np.sqrt((err ** 2).mean())),
        "MAPE_pct": float((np.abs(err) / y).mean() * 100),
        "hours": int(len(y)),
    }


def walk_forward(table: pd.DataFrame, nyiso: pd.Series | None, test_months: int = 24) -> pd.DataFrame:
    data = table.dropna(subset=["load_mw"]).copy()
    local_month = data.index.tz_convert(config.LOCAL_TZ).tz_localize(None).to_period("M")
    months = sorted(local_month.unique())[-test_months:]

    preds = []
    for m in months:
        test_mask = local_month == m
        test_start = data.index[test_mask].min()
        # Train only on targets that were observed before the first test issue time.
        train = data[data.index < test_start - pd.Timedelta(days=1)]
        model = make_model()
        model.fit(train[FEATURES], train["load_mw"])
        # Only score hours that have weather (the archive lags real time by ~6 days).
        test = data[test_mask].dropna(subset=["temperature_2m"])
        if test.empty:
            continue
        out = pd.DataFrame(index=test.index)
        out["actual"] = test["load_mw"]
        out["gbm"] = model.predict(test[FEATURES])
        out["seasonal_naive"] = test["load_same_hour_d7"]
        preds.append(out)
        print(f"  {m}: trained on {len(train):,} hours")

    pred = pd.concat(preds)
    if nyiso is not None:
        pred["nyiso"] = nyiso.reindex(pred.index)
    return pred


def summarize(pred: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    models = [c for c in ["seasonal_naive", "nyiso", "gbm"] if c in pred]
    common = pred.dropna(subset=["actual", *models])
    overall = {m: metrics(common["actual"], common[m]) for m in models}
    month = common.index.tz_convert(config.LOCAL_TZ).tz_localize(None).to_period("M")
    monthly = (
        common.groupby(month)
        .apply(lambda g: pd.Series({m: (np.abs(g[m] - g["actual"]) / g["actual"]).mean() * 100 for m in models}))
    )
    monthly.index = monthly.index.astype(str)
    return overall, monthly


def save_report(pred: pd.DataFrame, overall: dict, monthly: pd.DataFrame) -> None:
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(config.REPORTS / "backtest_predictions.parquet")
    monthly.round(2).to_csv(config.REPORTS / "backtest_monthly_mape.csv")
    (config.REPORTS / "backtest_metrics.json").write_text(json.dumps(overall, indent=2))
