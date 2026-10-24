"""python -m wnyload.charts  -- redraw the backtest result charts from saved reports.

Charts are written to reports/figures/. Every chart uses the same colors for the same
forecast (model blue, NYISO orange, seasonal naive aqua, actual load in ink), so they
read as one set in the README.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # quiet when a listed font is absent
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

FIGURES = config.REPORTS / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COLORS = {"actual": INK, "gbm": "#2a78d6", "nyiso": "#eb6834", "seasonal_naive": "#1baf7a"}
LABELS = {"actual": "Actual", "gbm": "Model", "nyiso": "NYISO forecast", "seasonal_naive": "Same hour last week"}
MODELS = ["seasonal_naive", "nyiso", "gbm"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 13,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "ytick.left": False,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})


def _wall(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC index -> local wall-clock time without tz, for plotting."""
    return index.tz_convert(config.LOCAL_TZ).tz_localize(None)


def _ape(pred: pd.DataFrame, model: str) -> pd.Series:
    return (pred[model] - pred["actual"]).abs() / pred["actual"] * 100


def _common(pred: pd.DataFrame) -> pd.DataFrame:
    """Hours where every model has a forecast, so comparisons are like for like."""
    return pred.dropna(subset=["actual", *[m for m in MODELS if m in pred]])


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _header(ax, title: str, subtitle: str, legend: bool = True, title_size: float = 13) -> None:
    """Title, subtitle and legend row stacked above the plot, so the legend never covers data."""
    sub_offset = 26 if legend else 6
    ax.set_title(title, pad=sub_offset + 20, fontsize=title_size)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, sub_offset),
                textcoords="offset points", color=INK_2, fontsize=9.5, va="bottom")
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=len(handles), borderaxespad=0.2,
                  handlelength=1.6, columnspacing=1.6, borderpad=0)


# Backtest charts -------------------------------------------------------------

def accuracy_chart(pred: pd.DataFrame) -> None:
    p = _common(pred)
    models = sorted([m for m in MODELS if m in p], key=lambda m: -_ape(p, m).mean())
    mape = [_ape(p, m).mean() for m in models]
    mae = [(p[m] - p["actual"]).abs().mean() for m in models]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    y = np.arange(len(models))
    ax.barh(y, mape, height=0.5, color=[COLORS[m] for m in models])
    for yi, v, mw in zip(y, mape, mae):
        ax.text(v + max(mape) * 0.015, yi, f"{v:.2f}%  ({mw:.0f} MW avg miss)", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, [LABELS[m] for m in models])
    ax.set_xlim(0, max(mape) * 1.45)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Mean absolute percentage error (lower is better)")
    start, end = _wall(p.index[[0, -1]])
    _header(ax, "Day-ahead forecast error, Zone A (West)",
            f"Walk-forward backtest, {start:%b %Y} to {end:%b %Y}, {len(p):,} hours", legend=False)
    _save(fig, "accuracy.png")


def monthly_chart(pred: pd.DataFrame) -> None:
    p = _common(pred)
    month = _wall(p.index).to_period("M")
    fig, ax = plt.subplots(figsize=(9, 4))
    for m in [m for m in MODELS if m in p]:
        s = _ape(p, m).groupby(month).mean()
        ax.plot(s.index.to_timestamp(), s.values, color=COLORS[m], marker="o", markersize=4.5, label=LABELS[m])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MAPE (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    _header(ax, "Forecast error by month", "Each month's model was trained only on data available before that month")
    _save(fig, "monthly_error.png")


def by_hour_chart(pred: pd.DataFrame) -> None:
    p = _common(pred)
    hour = _wall(p.index).hour
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for m in [m for m in ["nyiso", "gbm"] if m in p]:
        s = _ape(p, m).groupby(hour).mean()
        ax.plot(s.index, s.values, color=COLORS[m], marker="o", markersize=4.5, label=LABELS[m])
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 24, 3), [f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}" for h in range(0, 24, 3)])
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("MAPE (%)")
    _header(ax, "Forecast error by hour of day",
            "The model keeps up with NYISO at night but misses by more in daylight hours")
    _save(fig, "error_by_hour.png")


def by_temperature_chart(pred: pd.DataFrame, temperature_c: pd.Series) -> None:
    p = _common(pred).copy()
    p["temp_f"] = temperature_c.reindex(p.index) * 9 / 5 + 32
    p = p.dropna(subset=["temp_f"])
    edges = np.arange(-10, 101, 10)
    p["bin"] = pd.cut(p["temp_f"].clip(edges[0], edges[-1] - 0.01), edges, right=False)
    counts = p.groupby("bin", observed=True).size()
    keep = counts[counts >= 50].index

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for m in [m for m in ["nyiso", "gbm"] if m in p]:
        s = _ape(p, m).groupby(p["bin"], observed=True).mean().reindex(keep)
        ax.plot(range(len(s)), s.values, color=COLORS[m], marker="o", markersize=4.5, label=LABELS[m])
    ax.set_xticks(range(len(keep)), [f"{int(b.left)}–{int(b.right)}°F" for b in keep])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MAPE (%)")
    _header(ax, "Forecast error by temperature",
            "Buffalo airport temperature at the target hour; bins with under 50 hours left out")
    _save(fig, "error_by_temperature.png")


def _day_lines(ax, p: pd.DataFrame) -> None:
    t = _wall(p.index)
    for m in ["actual", "nyiso", "gbm"]:
        if m in p:
            ax.plot(t, p[m], color=COLORS[m], lw=2.4 if m == "actual" else 1.8, label=LABELS[m])


def week_chart(pred: pd.DataFrame) -> None:
    p = _common(pred)
    day = _wall(p.index).normalize()
    full_days = day.value_counts()
    full_days = full_days[full_days >= 23].index.sort_values()
    last7 = full_days[-7:]
    week = p[day.isin(last7)]

    fig, ax = plt.subplots(figsize=(10, 4))
    _day_lines(ax, week)
    ax.set_ylabel("Load (MW)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%b %d"))
    mapes = ", ".join(f"{LABELS[m]} {_ape(week, m).mean():.1f}%" for m in ["gbm", "nyiso"] if m in week)
    _header(ax, "Actual load vs. day-ahead forecasts, most recent backtest week", f"MAPE this week: {mapes}")
    _save(fig, "sample_week.png")


def worst_days(pred: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    p = _common(pred)
    day = _wall(p.index).date
    daily = pd.DataFrame({m: _ape(p, m).groupby(day).mean() for m in MODELS if m in p})
    daily["actual_peak_mw"] = p["actual"].groupby(day).max()
    return daily.sort_values("gbm", ascending=False).head(n).round(2)


def worst_days_chart(pred: pd.DataFrame, n: int = 3) -> None:
    p = _common(pred)
    worst = worst_days(pred, n)
    day = _wall(p.index).date
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.4))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.70, bottom=0.09, wspace=0.22)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, worst.index):
        _day_lines(ax, p[day == d])
        ts = pd.Timestamp(d)
        _header(ax, f"{ts:%a %b} {ts.day}, {ts.year}",
                f"MAPE: model {worst.loc[d, 'gbm']:.1f}%, NYISO {worst.loc[d, 'nyiso']:.1f}%",
                legend=False, title_size=11)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_hour_label))
    axes[0].set_ylabel("Load (MW)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("The model's worst days in the backtest", x=0.05, y=0.995, ha="left", va="top", fontsize=13, color=INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.905), ncol=3,
               handlelength=1.6, columnspacing=1.6, borderpad=0)
    _save(fig, "worst_days.png")


def _hour_label(x, _pos) -> str:
    h = mdates.num2date(x).hour
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def make_backtest_charts(pred: pd.DataFrame, table: pd.DataFrame) -> None:
    accuracy_chart(pred)
    monthly_chart(pred)
    by_hour_chart(pred)
    by_temperature_chart(pred, table["temperature_2m"])
    week_chart(pred)
    worst_days_chart(pred)
    worst_days(pred).to_csv(config.REPORTS / "worst_days.csv", index_label="date")
    print(f"Charts written to {FIGURES}")


# Forecast chart --------------------------------------------------------------

def forecast_chart(out: pd.DataFrame, recent_actual: pd.Series, path) -> None:
    """Last few days of actual load, then the forecast day."""
    fig, ax = plt.subplots(figsize=(10, 4))
    t_out = _wall(out.index)
    if len(recent_actual):
        ax.plot(_wall(recent_actual.index), recent_actual.values, color=INK, lw=2.4, label="Actual")
    if out["actual_mw"].notna().any():
        ax.plot(t_out, out["actual_mw"], color=INK, lw=2.4, label=None if len(recent_actual) else "Actual")
    if out["nyiso_forecast_mw"].notna().any():
        ax.plot(t_out, out["nyiso_forecast_mw"], color=COLORS["nyiso"], lw=1.8, label=LABELS["nyiso"])
    ax.plot(t_out, out["forecast_mw"], color=COLORS["gbm"], lw=2.4, label=LABELS["gbm"])

    ax.axvline(t_out[0], color=AXIS, lw=1)
    ax.text(t_out[0], 1.0, "  forecast day", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=9, va="top")
    peak = out["forecast_mw"].idxmax()
    ax.annotate(f"Peak {out.loc[peak, 'forecast_mw']:,.0f} MW", xy=(_wall(pd.DatetimeIndex([peak]))[0], out.loc[peak, "forecast_mw"]),
                xytext=(0, 10), textcoords="offset points", ha="center", color=INK, fontsize=9.5)
    ax.margins(y=0.15)
    ax.set_ylabel("Load (MW)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%b %d"))
    day = t_out[0]
    _header(ax, f"Zone A (West) load forecast for {day:%A, %B} {day.day}, {day.year}",
            f"Issued {config.ISSUE_HOUR} AM the day before, with the last 3 days of actual load for context")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pred = pd.read_parquet(config.REPORTS / "backtest_predictions.parquet")
    table = pd.read_parquet(config.DATA_PROCESSED / "model_table.parquet")
    make_backtest_charts(pred, table)


if __name__ == "__main__":
    main()
