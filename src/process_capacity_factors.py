"""
Build hub-height (80 m) wind speeds from ERA5 100 m and 10 m series via an hourly
power-law shear exponent, convert to capacity factors with the Acciona AW-77/1500
power curve, and save a compact processed dataset.

Inputs
  data/processed/wind_speed_100m.parquet, wind_direction.parquet  (ERA5 100 m, 20 sites)
  data/raw_10m/*.json                                              (ERA5 10 m, see download_era5_10m.py)
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from power_curve import capacity_factor, HUB_HEIGHT_M

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RAW10 = os.path.join(ROOT, "data", "raw_10m")

SITES = {
    "T01": (15.993, -94.428), "T02": (15.993, -94.683), "T03": (15.993, -94.939),
    "T04": (15.993, -95.109), "T05": (16.063, -95.369), "T06": (16.274, -94.444),
    "T07": (16.274, -94.530), "T08": (16.344, -94.791), "T09": (16.274, -95.128),
    "T10": (16.274, -95.385), "T11": (16.485, -94.371), "T12": (16.485, -94.629),
    "T13": (16.485, -94.886), "T14": (16.485, -95.143), "T15": (16.485, -95.400),
    "T16": (16.766, -94.388), "T17": (16.766, -94.560), "T18": (16.766, -94.904),
    "T19": (16.766, -95.163), "T20": (16.696, -95.415),
}


def hub_speed(v100, v10, hub=HUB_HEIGHT_M):
    """Extrapolate 100 m speed to hub height with an hourly shear exponent from 10/100 m."""
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.log(v100 / v10) / np.log(100.0 / 10.0)
    valid = np.isfinite(alpha) & (v10 > 0.5) & (v100 > 0.5)
    alpha = np.where(valid, np.clip(alpha, 0.0, 0.6), np.nan)
    alpha = np.where(np.isnan(alpha), np.nanmedian(alpha), alpha)
    return v100 * (hub / 100.0) ** alpha, alpha


def main():
    v100_df = pd.read_parquet(os.path.join(PROC, "wind_speed_100m.parquet"))
    dir_df = pd.read_parquet(os.path.join(PROC, "wind_direction.parquet"))
    hub, cf, alpha_mean = {}, {}, {}
    for site, (lat, lon) in SITES.items():
        raw = json.load(open(os.path.join(RAW10, f"{site}_{lat}_{lon}.json")))
        v10 = np.array(raw["hourly"]["wind_speed_10m"], dtype=float)
        v100 = v100_df[site].values
        v80, alpha = hub_speed(v100, v10)
        hub[site] = v80
        cf[site] = capacity_factor(v80)
        alpha_mean[site] = float(np.mean(alpha))
        print(f"{site}: mean v100={v100.mean():.2f} v80={v80.mean():.2f} m/s, "
              f"alpha={alpha_mean[site]:.3f}, CF={cf[site].mean()*100:.1f}% std={cf[site].std()*100:.1f}%")

    idx = v100_df.index
    hub_df = pd.DataFrame(hub, index=idx)
    cf_df = pd.DataFrame(cf, index=idx)
    hub_df.to_parquet(os.path.join(PROC, "wind_speed_hub80.parquet"))
    cf_df.to_parquet(os.path.join(PROC, "capacity_factors.parquet"))
    stats = pd.DataFrame({
        "lat": {s: SITES[s][0] for s in SITES},
        "lon": {s: SITES[s][1] for s in SITES},
        "mean_v80_ms": hub_df.mean(),
        "mean_shear_alpha": pd.Series(alpha_mean),
        "mean_cf_pct": cf_df.mean() * 100,
        "std_cf_pct": cf_df.std() * 100,
    })
    stats.to_csv(os.path.join(PROC, "site_stats.csv"))
    print("saved processed data")


if __name__ == "__main__":
    main()
