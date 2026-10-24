"""Project-wide settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# NYISO load zone A ("West"): Buffalo / Western New York.
ZONE_NAME_LOAD = "WEST"      # value in the "Name" column of palIntegrated files
ZONE_NAME_FORECAST = "West"  # column name in isolf files

# Weather point: Buffalo Niagara International Airport (KBUF).
WEATHER_LAT = 42.94
WEATHER_LON = -78.73
LOCAL_TZ = "America/New_York"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]

# History to download and use.
START_DATE = "2018-01-01"

# Forecast setup: the forecast for every hour of day D is issued at
# ISSUE_HOUR (local time) on day D-1. Features may only use load observed
# at or before that moment. This is what keeps the backtest honest.
ISSUE_HOUR = 9
