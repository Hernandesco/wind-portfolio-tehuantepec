"""
Level 2: wake-risk-aware turbine layout optimization at the best Level-1
site (T03), using real ERA5-derived wind speed/direction bins and the
Bastankhah-Gaussian wake model. Parametric grid search over rotation,
row spacing, and stagger fraction (the approach found to be numerically
robust, in place of unstable free-form continuous SQP -- see paper Sec. 3.3.4).
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
N_SPEED_BINS = 8
HOURS_PER_YEAR = 8760.0


def build_wind_bins(site):
    speed = pd.read_parquet(os.path.join(DATA_DIR, "wind_speed.parquet"))[site].values
    direction = pd.read_parquet(os.path.join(DATA_DIR, "wind_direction.parquet"))[site].values

    dir_edges = np.linspace(0, 360, N_DIR_BINS + 1)
    dir_bin = np.digitize(direction, dir_edges) - 1
    dir_bin = np.clip(dir_bin, 0, N_DIR_BINS - 1)
    dir_centers = 0.5 * (dir_edges[:-1] + dir_edges[1:])

    speed_edges = np.linspace(0, 25, N_SPEED_BINS + 1)
    speed_bin = np.digitize(speed, speed_edges) - 1
    speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)
    speed_centers = 0.5 * (speed_edges[:-1] + speed_edges[1:])

    n_total = len(speed)
    bins = []
    for di in range(N_DIR_BINS):
        for si in range(N_SPEED_BINS):
            mask = (dir_bin == di) & (speed_bin == si)
            freq = mask.sum() / n_total
            if freq > 0:
                bins.append({"dir": dir_centers[di], "speed": speed_centers[si], "freq": freq})
    return bins


def evaluate_layout(positions, bins):
    """For each turbine, build its total-deficit value under every (direction,
    speed) wind bin -- a proxy 'time series' over wind conditions -- then
    compute the frequency-weighted pairwise correlation between turbines'
    deficit series (Eq. 12-13: turbines that lose power together under the
    same wind conditions are correlated risk, not just co-located)."""
    n = len(positions)
    n_bins = len(bins)
    aep_mw = 0.0
    deficit_series = np.zeros((n_bins, n))  # rows = wind bins, cols = turbines
    freqs = np.array([b["freq"] for b in bins])

    for bi, b in enumerate(bins):
        eff_speed, total_def, pair_def = effective_speeds(positions, b["dir"], b["speed"])
        cf = capacity_factor(eff_speed)
        power_mw = cf * RATED_POWER_MW
        aep_mw += power_mw.sum() * b["freq"] * HOURS_PER_YEAR
        deficit_series[bi, :] = total_def

    # frequency-weighted correlation matrix across turbines' deficit series
    w = freqs / freqs.sum()
    mean_d = (w[:, None] * deficit_series).sum(axis=0)
    centered = deficit_series - mean_d
    cov = (w[:, None, None] * (centered[:, :, None] * centered[:, None, :])).sum(axis=0)
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.where(denom > 1e-12, cov / denom, 0.0)
    off_diag_mask = ~np.eye(n, dtype=bool)
    mean_corr = float(corr_matrix[off_diag_mask].mean())
    wake_risk_wr = float(cov[off_diag_mask].sum())  # aggregate deficit covariance (Eq. 14 analogue)
    return aep_mw, mean_corr, wake_risk_wr


def main():
    bins = build_wind_bins(SITE)
    print(f"{SITE}: {len(bins)} nonzero (direction, speed) bins from {sum(b['freq'] for b in bins)*100:.1f}% of hours")

    # Rotation fixed at 0 deg (rows aligned with the site's dominant, strongly
    # bidirectional N-S "Tehuano" wind axis -- see build_wind_bins output);
    # spacing and stagger are the free design variables, matching the
    # AEP-max vs. risk-adjusted comparison reported in the paper (Table 3).
    rotations = [0]
    along_Ds = [8, 9, 10, 11, 12]
    cross_Ds = [3, 4, 5, 6, 7, 8]
    staggers = [0.0, 0.25, 0.5, 0.75]

    results = []
    for rot, aD, cD, stag in itertools.product(rotations, along_Ds, cross_Ds, staggers):
        pos = turbine_layout(N_ROWS, N_COLS, along_D=aD, cross_D=cD,
                              stagger_frac=stag, rotation_deg=rot, d=D)
        aep, corr, wr = evaluate_layout(pos, bins)
        results.append({"rotation": rot, "along_D": aD, "cross_D": cD, "stagger": stag,
                         "aep_gwh": aep / 1000.0, "mean_corr": corr, "wake_risk": wr})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "layout_search.csv"), index=False)

    aep_max_row = df.loc[df["aep_gwh"].idxmax()]

    # Risk-adjusted layout: rather than the global minimum-correlation point
    # in the whole grid (which can require large, impractical geometry changes),
    # select the best *single-parameter* perturbation of the AEP-maximizing
    # baseline -- i.e., a minimally invasive adjustment -- among those that
    # keep AEP within 0.1% of the maximum. This mirrors the paper's framing
    # of an incremental spacing/stagger adjustment (Table 3).
    params = ["rotation", "along_D", "cross_D", "stagger"]
    same_except_one = df.apply(
        lambda row: sum(row[p] != aep_max_row[p] for p in params) <= 1, axis=1
    )
    aep_tolerance = aep_max_row["aep_gwh"] * 0.999
    neighbors = df[same_except_one & (df["aep_gwh"] >= aep_tolerance) & (df.index != aep_max_row.name)]
    if len(neighbors) == 0:
        neighbors = df[df["aep_gwh"] >= aep_tolerance]
    risk_adj_row = neighbors.loc[neighbors["mean_corr"].idxmin()]

    summary = {
        "site": SITE,
        "n_turbines": N_ROWS * N_COLS,
        "n_configs_evaluated": len(df),
        "aep_maximizing": aep_max_row.to_dict(),
        "risk_adjusted": risk_adj_row.to_dict(),
        "correlation_reduction_pct": float(
            (aep_max_row["mean_corr"] - risk_adj_row["mean_corr"]) / aep_max_row["mean_corr"] * 100
        ),
        "aep_change_pct": float(
            (risk_adj_row["aep_gwh"] - aep_max_row["aep_gwh"]) / aep_max_row["aep_gwh"] * 100
        ),
    }
    with open(os.path.join(RESULTS_DIR, "level2_layout_result.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
