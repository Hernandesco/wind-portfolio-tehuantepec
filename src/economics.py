"""
Economic evaluation (NPV, LCOE, CVaR95) of the Level-1 portfolios and the
single-site benchmark, using NREL 2024 ATB land-based wind cost parameters.
"""
import os
import json
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CAPEX_USD_PER_KW = 1968.0
OPEX_USD_PER_KW_YR = 43.0
DISCOUNT_RATE = 0.08
LIFETIME_YEARS = 25
PRICE_USD_PER_MWH = 45.0
CAPACITY_MW = 200.0


def annual_cf_series(cf_series):
    """Split an hourly capacity-factor series into per-calendar-year means."""
    return cf_series.groupby(cf_series.index.year).mean()


def evaluate_portfolio(weights, cf_df):
    """weights: dict site -> fraction of installed capacity."""
    sites = list(weights.keys())
    w = np.array([weights[s] for s in sites])
    portfolio_cf_hourly = (cf_df[sites].values * w).sum(axis=1)
    portfolio_cf = pd.Series(portfolio_cf_hourly, index=cf_df.index)
    annual_cf = annual_cf_series(portfolio_cf)

    capex_total = CAPEX_USD_PER_KW * CAPACITY_MW * 1000
    opex_annual = OPEX_USD_PER_KW_YR * CAPACITY_MW * 1000

    annual_energy_mwh = annual_cf * CAPACITY_MW * 8760
    annual_revenue = annual_energy_mwh * PRICE_USD_PER_MWH
    annual_cashflow = annual_revenue - opex_annual

    years = np.arange(1, LIFETIME_YEARS + 1)
    # cycle the ~25 observed years to cover the assumed project lifetime
    cf_cycle = np.resize(annual_cashflow.values, LIFETIME_YEARS)
    disc_factors = 1.0 / (1.0 + DISCOUNT_RATE) ** years
    npv = -capex_total + (cf_cycle * disc_factors).sum()

    mean_energy_mwh = annual_energy_mwh.mean()
    disc_energy = (np.resize(annual_energy_mwh.values, LIFETIME_YEARS) * disc_factors).sum()
    disc_costs = capex_total + (np.resize(np.full(LIFETIME_YEARS, opex_annual), LIFETIME_YEARS) * disc_factors).sum()
    lcoe = disc_costs / disc_energy if disc_energy > 0 else np.nan

    # CVaR95 on annual cashflow distribution (mean of worst 5% of observed years)
    sorted_cf = np.sort(annual_cashflow.values)
    n_tail = max(1, int(np.ceil(0.05 * len(sorted_cf))))
    cvar95 = sorted_cf[:n_tail].mean()
    mean_cf = annual_cashflow.mean()
    # use abs(mean) in the denominator so the sign always reflects "worse than
    # average" (a negative mean should not flip a downside gap into a
    # positive percentage)
    cvar_vs_mean_pct = (cvar95 - mean_cf) / abs(mean_cf) * 100 if mean_cf != 0 else np.nan

    return {
        "mean_aep_gwh": float(mean_energy_mwh / 1000),
        "lcoe_usd_per_mwh": float(lcoe),
        "npv_usd_m": float(npv / 1e6),
        "cvar95_usd_m_per_yr": float(cvar95 / 1e6),
        "cvar_vs_mean_pct": float(cvar_vs_mean_pct),
    }


def main():
    cf_df = pd.read_parquet(os.path.join(DATA_DIR, "capacity_factors.parquet"))
    with open(os.path.join(RESULTS_DIR, "portfolios_converged.json")) as f:
        port_data = json.load(f)

    econ_results = {}
    for name, p in port_data["portfolios"].items():
        econ_results[name] = evaluate_portfolio(p["weights"], cf_df)
        print(name, econ_results[name])

    # single-site benchmark = the aggressive (100%-best-site) portfolio, kept
    # separately for clarity even though it is numerically identical here
    econ_results["single_site_benchmark"] = econ_results["aggressive"]

    with open(os.path.join(RESULTS_DIR, "economics_result.json"), "w") as f:
        json.dump(econ_results, f, indent=2)
    print("\nSaved economics_result.json to", RESULTS_DIR)


if __name__ == "__main__":
    main()
