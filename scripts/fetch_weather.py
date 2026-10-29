"""Download hourly Washington DC weather (2021-2025) from Open-Meteo into data/raw/weather.csv.

Usage:
    python scripts/fetch_weather.py
    python scripts/fetch_weather.py --start 2024-01-01 --end 2024-02-29   # small test

Source: https://open-meteo.com/en/docs/historical-weather-api (free, no account).
Times are local DC time (America/New_York) to match the trip timestamps.
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 38.9072, -77.0369  # downtown Washington DC
VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"]
OUT = Path("data/raw/weather.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": args.start,
        "end_date": args.end,
        "hourly": ",".join(VARS),
        "timezone": "America/New_York",
    }
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json()["hourly"]

    df = pd.DataFrame(hourly).rename(columns={
        "time": "hour",
        "temperature_2m": "temperature",
        "wind_speed_10m": "wind",
        "relative_humidity_2m": "humidity",
    })
    df["hour"] = pd.to_datetime(df["hour"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved {len(df):,} hourly rows to {OUT}")
    print(f"Range: {df['hour'].min()} -> {df['hour'].max()}")
    print("Missing values per column:")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
