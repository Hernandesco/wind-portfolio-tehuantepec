"""
Level 2: wake-risk-aware turbine layout optimization, using ERA5-derived hub-height
wind speed/direction bins and the Bastankhah-Gaussian wake model. Parametric grid
search over row spacing and stagger (rotation fixed to the site's dominant wind axis);
used as a library by two_level_loop.py and as a script for the best site (T03).
"""
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from wake_model import turbine_layout, effective_speeds, D
from power_curve import capacity_factor, RATED_POWER_MW

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SITE = "T03"
N_ROWS, N_COLS = 4, 4  # 16 turbines
N_DIR_BINS = 16
N_SPEED_BINS = 20
HOURS_PER_YEAR = 8760.0
AEP_TOLERANCE = 0.005  # risk-adjusted layout may lose at most 0.5% AEP
MAX_AREA_D2 = 36.0     # land-use budget: along_D * cross_D <= 36 D^2 per turbine (~7 MW/km^2 for 1.5 MW, D = 77 m)
SWEEP_AREAS = [25.0, 30.0, 36.0, 50.0]

ALONG_D = [5, 6, 7, 8, 9, 10, 11, 12]
CROSS_D = [3, 4, 5, 6, 7, 8, 9, 10]
STAGGER = [0.0, 0.25, 0.5, 0.75]
ROTATION = [0]

_speed = None
_direction = None


def _load():
    global _speed, _direction
    if _speed is None:
        _speed = pd.read_parquet(os.path.join(DATA_DIR, "wind_speed_hub80.parquet"))
        _direction = pd.read_parquet(os.path.join(DATA_DIR, "wind_direction.parquet"))


def build_wind_bins(site):
    """Return (bins, hour_bin_index): nonzero (direction, speed) bins with frequencies and
    the bin index of every hour in the 25-year record."""
    _load()
    speed = _speed[site].values
    direction = _direction[site].values
    dir_edges = np.linspace(0, 360, N_DIR_BINS + 1)
    spd_edges = np.linspace(0, 25, N_SPEED_BINS + 1)
    di = np.clip(np.digitize(direction, dir_edges) - 1, 0, N_DIR_BINS - 1)
    si = np.clip(np.digitize(speed, spd_edges) - 1, 0, N_SPEED_BINS - 1)
    key = di * N_SPEED_BINS + si
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    dc = 0.5 * (dir_edges[:-1] + dir_edges[1:])
    sc = 0.5 * (spd_edges[:-1] + spd_edges[1:])
    bins = [{"dir": dc[k // N_SPEED_BINS], "speed": sc[k % N_SPEED_BINS], "freq": c / len(key)}
            for k, c in zip(uniq, counts)]
    return bins, inv


def evaluate_layout(positions, bins):
    """AEP (MWh/yr), mean pairwise correlation of turbine power deficits (Eq. 12-13), wake risk
    WR = Var(total farm power deficit) = sum_mn Cov(dP_m, dP_n) in MW^2 (Eq. 14 analogue) and
    the farm-average capacity factor in every wind bin. dP_m is turbine m's power deficit
    relative to free-stream operation, evaluated over the (direction, speed) distribution."""
    n, nb = len(positions), len(bins)
    dP = np.zeros((nb, n))
    farm_cf = np.zeros(nb)
    aep = 0.0
    for bi, b in enumerate(bins):
        eff, _, _ = effective_speeds(positions, b["dir"], b["speed"])
        cf = capacity_factor(eff)
        free = float(capacity_factor(b["speed"]))
        farm_cf[bi] = cf.mean()
        aep += cf.sum() * RATED_POWER_MW * b["freq"] * HOURS_PER_YEAR
        dP[bi] = (free - cf) * RATED_POWER_MW
    w = np.array([b["freq"] for b in bins])
    w = w / w.sum()
    centered = dP - (w[:, None] * dP).sum(axis=0)
    cov = np.einsum("b,bi,bj->ij", w, centered, centered)
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 1e-12, cov / denom, 0.0)
    off = ~np.eye(n, dtype=bool)
    return aep, float(corr[off].mean()), float(cov.sum()), farm_cf


def search_site(site, bins=None, max_area=None):
    """Grid search under the land-use budget. Returns (DataFrame, aep_max_row, risk_adjusted_row,
    layouts): the risk-adjusted layout minimizes the wake risk WR among layouts whose AEP is
    within AEP_TOLERANCE of the maximum (Eq. 15 with AEP treated as a near-binding constraint)."""
    if bins is None:
        bins, _ = build_wind_bins(site)
    rows, farm = [], {}
    for rot, aD, cD, st in itertools.product(ROTATION, ALONG_D, CROSS_D, STAGGER):
        if aD * cD > (MAX_AREA_D2 if max_area is None else max_area):
            continue
        pos = turbine_layout(N_ROWS, N_COLS, along_D=aD, cross_D=cD, stagger_frac=st, rotation_deg=rot)
        aep, corr, wr, fcf = evaluate_layout(pos, bins)
        rows.append({"rotation": rot, "along_D": aD, "cross_D": cD, "stagger": st,
                     "aep_gwh": aep / 1000.0, "mean_corr": corr, "wake_risk": wr})
        farm[len(rows) - 1] = (pos, fcf)
    df = pd.DataFrame(rows)
    aep_max = df.loc[df["aep_gwh"].idxmax()]
    ok = df[df["aep_gwh"] >= aep_max["aep_gwh"] * (1 - AEP_TOLERANCE)]
    risk_adj = ok.loc[ok["wake_risk"].idxmin()]
    layouts = {"aep_maximizing": farm[aep_max.name], "risk_adjusted": farm[risk_adj.name]}
    return df, aep_max, risk_adj, layouts


def wake_loss_series(site, farm_cf_bins, hour_bin_index, bins):
    """Hourly wake-induced capacity-factor loss L_i(t): bin-center free-stream CF minus
    farm-average CF with wake interaction."""
    free = np.array([float(capacity_factor(b["speed"])) for b in bins])
    return (free - farm_cf_bins)[hour_bin_index]


def summarize(a, r):
    return {
        "aep_maximizing": a.to_dict(), "risk_adjusted": r.to_dict(),
        "correlation_reduction_pct": float((a["mean_corr"] - r["mean_corr"]) / a["mean_corr"] * 100) if a["mean_corr"] else 0.0,
        "aep_change_pct": float((r["aep_gwh"] - a["aep_gwh"]) / a["aep_gwh"] * 100),
        "wake_risk_change_pct": float((r["wake_risk"] - a["wake_risk"]) / a["wake_risk"] * 100) if a["wake_risk"] else 0.0,
    }


def main():
    bins, _ = build_wind_bins(SITE)
    print(f"{SITE}: {len(bins)} nonzero (direction, speed) bins")
    df, a, r, _ = search_site(SITE, bins)
    df.to_csv(os.path.join(RESULTS_DIR, "layout_search.csv"), index=False)
    summary = {"site": SITE, "n_turbines": N_ROWS * N_COLS, "n_configs_evaluated": len(df),
               "max_area_d2": MAX_AREA_D2, "aep_tolerance": AEP_TOLERANCE, **summarize(a, r)}
    # sensitivity to the land-use budget
    sweep = []
    for area in SWEEP_AREAS:
        d2, a2, r2, _ = search_site(SITE, bins, max_area=area)
        row = summarize(a2, r2)
        sweep.append({"max_area_d2": area, "n_configs": len(d2),
                      "aep_max_layout": f"{a2['along_D']:.0f}D x {a2['cross_D']:.0f}D, stagger {a2['stagger']}",
                      "risk_adj_layout": f"{r2['along_D']:.0f}D x {r2['cross_D']:.0f}D, stagger {r2['stagger']}",
                      "aep_max_gwh": a2["aep_gwh"], "aep_change_pct": row["aep_change_pct"],
                      "corr_aep_max": a2["mean_corr"], "corr_risk_adj": r2["mean_corr"],
                      "wr_aep_max": a2["wake_risk"], "wr_risk_adj": r2["wake_risk"]})
    pd.DataFrame(sweep).to_csv(os.path.join(RESULTS_DIR, "layout_density_sweep.csv"), index=False)
    summary["density_sweep"] = sweep
    with open(os.path.join(RESULTS_DIR, "level2_layout_result.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "density_sweep"}, indent=2, default=float))
    print(pd.DataFrame(sweep).to_string())


if __name__ == "__main__":
    main()
