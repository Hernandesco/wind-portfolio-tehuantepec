"""
Load raw ERA5 hourly wind speed/direction JSON for the 20 Tehuantepec grid
points, convert to capacity factors via the V150-4.2 MW power curve, and
save a compact processed dataset (capacity factors + wind direction).
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from power_curve import capacity_factor

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

SITES = {
    "T01": (15.993, -94.428), "T02": (15.993, -94.683), "T03": (15.993, -94.939),
    "T04": (15.993, -95.109), "T05": (16.063, -95.369), "T06": (16.274, -94.444),
    "T07": (16.274, -94.530), "T08": (16.344, -94.791), "T09": (16.274, -95.128),
    "T10": (16.274, -95.385), "T11": (16.485, -94.371), "T12": (16.485, -94.629),
    "T13": (16.485, -94.886), "T14": (16.485, -95.143), "T15": (16.485, -95.400),
    "T16": (16.766, -94.388), "T17": (16.766, -94.560), "T18": (16.766, -94.904),
    "T19": (16.766, -95.163), "T20": (16.696, -95.415),
}


def main():
    cf_data = {}
    dir_data = {}
    speed_data = {}
    time_index = None
    for site_id, (lat, lon) in SITES.items():
        fname = f"{site_id}_{lat}_{lon}.json"
        path = os.path.join(RAW_DIR, fname)
        with open(path) as f:
            raw = json.load(f)
        hourly = raw["hourly"]
        if time_index is None:
            time_index = pd.to_datetime(hourly["time"])
        speed = np.array(hourly["wind_speed_100m"], dtype=float)
        direction = np.array(hourly["wind_direction_100m"], dtype=float)
        cf = capacity_factor(speed)
        cf_data[site_id] = cf
        dir_data[site_id] = direction
        speed_data[site_id] = speed
        print(f"{site_id}: mean CF = {cf.mean()*100:.1f}%, std = {cf.std()*100:.1f}%")

    cf_df = pd.DataFrame(cf_data, index=time_index)
    dir_df = pd.DataFrame(dir_data, index=time_index)
    speed_df = pd.DataFrame(speed_data, index=time_index)

    cf_df.to_parquet(os.path.join(OUT_DIR, "capacity_factors.parquet"))
    dir_df.to_parquet(os.path.join(OUT_DIR, "wind_direction.parquet"))
    speed_df.to_parquet(os.path.join(OUT_DIR, "wind_speed.parquet"))

    site_stats = pd.DataFrame({
        "lat": {s: SITES[s][0] for s in SITES},
        "lon": {s: SITES[s][1] for s in SITES},
        "mean_cf_pct": cf_df.mean() * 100,
        "std_cf_pct": cf_df.std() * 100,
    })
    site_stats.to_csv(os.path.join(OUT_DIR, "site_stats.csv"))
    print("\nSaved processed capacity factors, wind direction/speed, and site stats to", OUT_DIR)


if __name__ == "__main__":
    main()
