"""Turn raw trips + weather into the station x hour dataset.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --min-active 0.5

Reads:   data/raw/trips/*.csv, data/raw/weather.csv
Writes:  data/processed/demand.npy    (N stations, T hours, 2) uint16 -> [departures, arrivals]
         data/processed/hours.csv     one row per hour: weather, calendar, outage flag
         data/processed/stations.csv  one row per kept station (same order as demand.npy)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

RAW = Path("data/raw")
OUT = Path("data/processed")
COLS = ["started_at", "ended_at", "start_station_id", "end_station_id",
        "start_station_name", "start_lat", "start_lng"]


def clean_id(s):
    """Station IDs as clean strings ('31519.0' -> '31519', '' -> missing)."""
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.mask(s == "")


def load_month(path):
    df = pd.read_csv(path, usecols=COLS, dtype={"start_station_id": "string",
                     "end_station_id": "string", "start_station_name": "string"})
    df["start_station_id"] = clean_id(df["start_station_id"])
    df["end_station_id"] = clean_id(df["end_station_id"])
    df["sh"] = pd.to_datetime(df["started_at"], format="ISO8601").dt.floor("h")
    df["eh"] = pd.to_datetime(df["ended_at"], format="ISO8601").dt.floor("h")

    s = df.dropna(subset=["start_station_id"])
    e = df.dropna(subset=["end_station_id"])
    dep = s.groupby(["start_station_id", "sh"]).size()
    arr = e.groupby(["end_station_id", "eh"]).size()
    info = s.groupby("start_station_id").agg(
        name=("start_station_name", "last"),
        lat=("start_lat", "median"),
        lng=("start_lng", "median"))
    stats = {"trips": len(df), "no_start_station": len(df) - len(s),
             "no_end_station": len(df) - len(e)}
    return dep, arr, info, stats


def to_grid(counts, st_index, t0, T, N):
    """Series indexed by (station, hour) -> dense (N, T) array."""
    d = counts.rename("n").reset_index()
    d.columns = ["sid", "hour", "n"]
    d = d[d["sid"].isin(st_index.keys())]
    t = ((d["hour"] - t0) // pd.Timedelta("1h")).astype(int)
    d = d[(t >= 0) & (t < T)]
    t = t[d.index]
    grid = np.zeros((N, T), dtype=np.int32)
    grid[d["sid"].map(st_index).to_numpy(), t.to_numpy()] = d["n"].to_numpy()
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-active", type=float, default=0.5,
                    help="keep stations with trips on at least this share of days")
    args = ap.parse_args()

    files = sorted((RAW / "trips").glob("*.csv"))
    if not files:
        raise SystemExit("No trip CSVs in data/raw/trips. Run download_trips.py first.")

    deps, arrs, infos, totals = [], [], [], {}
    for f in files:
        dep, arr, info, stats = load_month(f)
        deps.append(dep); arrs.append(arr); infos.append(info)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        print(f"[read] {f.name}: {stats['trips']:,} trips")

    # trips crossing month boundaries can share a key across files -> sum
    dep = pd.concat(deps).groupby(level=[0, 1]).sum()
    arr = pd.concat(arrs).groupby(level=[0, 1]).sum()
    info = pd.concat(infos).groupby(level=0).agg(
        name=("name", "last"), lat=("lat", "median"), lng=("lng", "median"))

    # hour grid: first day of earliest month -> last hour of latest month
    start_hours = dep.index.get_level_values(1)
    t0 = start_hours.min().to_period("M").start_time
    t1 = start_hours.max().to_period("M").end_time.floor("h")
    hours = pd.date_range(t0, t1, freq="h")
    T, n_days = len(hours), len(hours) // 24

    # station activity: share of days with any departure or arrival
    act = pd.concat([dep.reset_index().iloc[:, :2].set_axis(["sid", "hour"], axis=1),
                     arr.reset_index().iloc[:, :2].set_axis(["sid", "hour"], axis=1)])
    act = act[(act["hour"] >= t0) & (act["hour"] <= t1)]
    act["day"] = act["hour"].dt.normalize()
    g = act.groupby("sid")["day"]
    st = pd.DataFrame({"active_frac": g.nunique() / n_days,
                       "first_seen": g.min(), "last_seen": g.max()})
    keep = st[st["active_frac"] >= args.min_active].index.sort_values()
    N = len(keep)
    st_index = {sid: i for i, sid in enumerate(keep)}

    demand = np.stack([to_grid(dep, st_index, t0, T, N),
                       to_grid(arr, st_index, t0, T, N)], axis=-1)
    assert demand.max() < 65535
    demand = demand.astype(np.uint16)

    # outages: hours with zero trips system-wide, or days far below normal
    system = dep.groupby(level=1).sum().reindex(hours, fill_value=0)
    daily = system.resample("D").sum()
    ref = daily.rolling(29, center=True, min_periods=7).median()
    bad_days = daily.index[daily < 0.1 * ref]
    outage = (system.to_numpy() == 0) | hours.normalize().isin(bad_days)

    # weather (past observed values only; dataset.py decides what the model sees)
    w = pd.read_csv(RAW / "weather.csv", parse_dates=["hour"])
    w = w.drop_duplicates("hour").set_index("hour").reindex(hours)
    n_wx_missing = int(w.isna().any(axis=1).sum())
    w = w.interpolate(limit=3).ffill().bfill()

    holidays = USFederalHolidayCalendar().holidays(t0, t1)
    hours_df = pd.DataFrame({
        "hour": hours,
        "temperature": w["temperature"].to_numpy(),
        "precipitation": w["precipitation"].to_numpy(),
        "wind": w["wind"].to_numpy(),
        "humidity": w["humidity"].to_numpy(),
        "hour_of_day": hours.hour,
        "weekday": hours.weekday,
        "month": hours.month,
        "holiday": hours.normalize().isin(holidays).astype(int),
        "outage": outage.astype(int),
    })

    stations = st.loc[keep].join(info, how="left")
    stations["total_departures"] = demand[:, :, 0].sum(axis=1)
    stations["total_arrivals"] = demand[:, :, 1].sum(axis=1)
    stations = stations.reset_index().rename(columns={"index": "station_id", "sid": "station_id"})

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "demand.npy", demand)
    hours_df.to_csv(OUT / "hours.csv", index=False)
    stations.to_csv(OUT / "stations.csv", index=False)

    kept = int(demand[:, :, 0].sum())
    print("\n=== SUMMARY ===")
    print(f"Hours:      {T:,}  ({hours[0]} -> {hours[-1]})")
    print(f"Stations:   {N:,} kept of {len(st):,} (active on >= {args.min_active:.0%} of days)")
    print(f"Trips:      {totals['trips']:,} total")
    print(f"            {totals['no_start_station']:,} with no start station (dockless, dropped)")
    print(f"            {totals['no_end_station']:,} with no end station (dockless, dropped)")
    print(f"            {kept:,} departures at kept stations")
    print(f"Zero cells: {(demand[:, :, 0] == 0).mean():.1%} of station-hours have no departures")
    print(f"Outages:    {int(outage.sum()):,} hours flagged ({len(bad_days)} low-trip days)")
    print(f"Weather:    {n_wx_missing} hours filled by interpolation")
    print(f"Saved to {OUT}/  demand.npy {demand.shape}, hours.csv, stations.csv")


if __name__ == "__main__":
    main()
