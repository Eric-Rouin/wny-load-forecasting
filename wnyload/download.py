"""python -m wnyload.download  -- fetch NYISO load, NYISO forecasts and weather."""
from . import data_nyiso, data_weather


def main() -> None:
    print("NYISO actual load (palIntegrated)")
    data_nyiso.download("palIntegrated")
    print("NYISO load forecast (isolf)")
    data_nyiso.download("isolf")
    print("Weather history (Open-Meteo)")
    data_weather.download_history()
    print("Done.")


if __name__ == "__main__":
    main()
