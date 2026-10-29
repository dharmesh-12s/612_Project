"""Exploratory data analysis for the interim report.

Usage:  python scripts/eda.py
Reads data/processed/, writes figures and a stats table to results/eda/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA, OUT = Path("data/processed"), Path("results/eda")
SPLIT_COLORS = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"saved {OUT / name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    demand = np.load(DATA / "demand.npy")                     # (N, T, 2)
    hours = pd.read_csv(DATA / "hours.csv", parse_dates=["hour"])
    st = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    N, T, _ = demand.shape

    dep = demand[:, :, 0].astype(np.float32)
    hours["system_dep"] = dep.sum(0)
    hours["split"] = np.select([hours.hour.dt.year <= 2023, hours.hour.dt.year == 2024],
                               ["train", "val"], "test")
    ok = hours["outage"] == 0

    # 1. daily cycle: weekday vs weekend
    h = hours[ok].assign(weekend=hours.weekday >= 5)
    prof = h.groupby(["weekend", "hour_of_day"])["system_dep"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prof.index, prof[False], marker="o", ms=3, label="Weekday")
    ax.plot(prof.index, prof[True], marker="o", ms=3, label="Weekend")
    ax.set(title="Average system departures by hour of day", xlabel="Hour of day",
           ylabel="Departures per hour (all stations)", xticks=range(0, 24, 3))
    ax.legend()
    save(fig, "1_daily_cycle.png")

    # 2. weekly cycle (hour of week)
    h["how"] = h["weekday"] * 24 + h["hour_of_day"]
    wk = h.groupby("how")["system_dep"].mean()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(wk.index, wk.values)
    ax.set_xticks(range(0, 168, 24), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set(title="Average system departures over the week", ylabel="Departures per hour")
    save(fig, "2_weekly_cycle.png")

    # 3. monthly totals with splits (growth + seasonality)
    m = hours.set_index("hour")["system_dep"].resample("MS").sum()
    split_of = lambda d: "train" if d.year <= 2023 else ("val" if d.year == 2024 else "test")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(m.index, m.values / 1e3, width=25, color=[SPLIT_COLORS[split_of(d)] for d in m.index])
    from matplotlib.patches import Patch
    labels = {"train": "Train 2021-23", "val": "Val 2024", "test": "Test 2025"}
    ax.legend(handles=[Patch(color=c, label=labels[s]) for s, c in SPLIT_COLORS.items()])
    ax.set(title=f"Monthly departures at the {N} modeled stations", ylabel="Thousands of trips")
    save(fig, "3_monthly_totals.png")

    # 4. busiest stations
    top = st.nlargest(20, "total_departures").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["name"], top["total_departures"] / 1e3)
    ax.set(title="20 busiest stations (2021-2025)", xlabel="Thousands of departures")
    ax.tick_params(axis="y", labelsize=7)
    save(fig, "4_top_stations.png")

    # 5. station map
    fig, ax = plt.subplots(figsize=(6.5, 7))
    sc = ax.scatter(st["lng"], st["lat"], s=4 + 60 * st["total_departures"] / st["total_departures"].max(),
                    c=np.log10(st["total_departures"] + 1), cmap="viridis", alpha=0.8)
    plt.colorbar(sc, ax=ax, label="log10 total departures", shrink=0.7)
    ax.set(title=f"{N} stations (size and color = total departures)", xlabel="Longitude", ylabel="Latitude")
    ax.set_aspect(1 / np.cos(np.radians(38.9)))
    save(fig, "5_station_map.png")

    # 6. weather vs demand (daytime hours only, so night doesn't dominate)
    d = hours[ok & hours.hour_of_day.between(7, 20)].copy()
    d["temp_bin"] = (d["temperature"] // 3) * 3
    tb = d.groupby("temp_bin")["system_dep"].agg(["mean", "size"])
    tb = tb[tb["size"] >= 30]
    rain = d.assign(rain=np.where(d.precipitation > 0.5, "Rain (>0.5 mm/h)", "Dry"))
    rb = rain.groupby("rain")["system_dep"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(tb.index + 1.5, tb["mean"], marker="o")
    axes[0].set(title="Demand vs temperature (7am-8pm)", xlabel="Temperature (°C)",
                ylabel="Avg system departures per hour")
    axes[1].bar(rb.index, rb.values, color=["#4C78A8", "#9ECAE9"])
    axes[1].set(title="Dry vs rainy hours (7am-8pm)")
    save(fig, "6_weather.png")

    # 7. how sparse station-hours are
    counts = np.bincount(np.minimum(dep[:, ok.to_numpy()].astype(int).ravel(), 15), minlength=16)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(range(16), counts / counts.sum() * 100)
    ax.set_xticks(range(16), [str(i) for i in range(15)] + ["15+"])
    ax.set(title="Departures per station-hour", xlabel="Trips in the hour", ylabel="% of station-hours")
    save(fig, "7_count_distribution.png")

    # stats table
    yr = hours.groupby(hours.hour.dt.year)["system_dep"].sum()
    stats = {
        "Stations modeled": N,
        "Hours": f"{T:,} ({hours.hour.iloc[0]:%Y-%m-%d} to {hours.hour.iloc[-1]:%Y-%m-%d})",
        "Departures at modeled stations": f"{int(demand[:, :, 0].sum(dtype=np.int64)):,}",
        **{f"Departures {y}": f"{int(v):,}" for y, v in yr.items()},
        "Growth 2021 -> 2025": f"{yr.iloc[-1] / yr.iloc[0] - 1:+.0%}",
        "Mean departures per station-hour": f"{dep.mean():.2f}",
        "Station-hours with zero departures": f"{(dep == 0).mean():.1%}",
        "Max departures in one station-hour": int(dep.max()),
        "Peak system hour (avg weekday)": f"{prof[False].idxmax()}:00",
        "Outage hours masked": int(hours.outage.sum()),
        "Stations opened after Jan 2021": int((pd.to_datetime(st.first_seen) > "2021-02-01").sum()),
        "Busiest station": f"{top.iloc[-1]['name']} ({int(top.iloc[-1]['total_departures']):,})",
    }
    s = pd.Series(stats, name="value")
    s.to_csv(OUT / "dataset_stats.csv", header=True, index_label="stat")
    print("\n=== DATASET STATS ===")
    print(s.to_string())


if __name__ == "__main__":
    main()
