# Two-Level Markowitz Wind Portfolio Optimization — Isthmus of Tehuantepec

Code and data-processing pipeline for the two-level framework described in:

> Hernandez-Escobedo, Q. (2026). *A Two-Level Markowitz Framework for Wind Farm Portfolio
> Optimization: Spatial Diversification and Wake-Risk Management under Transmission-Constrained
> Grid Access — Application to the Isthmus of Tehuantepec Wind Corridor, Oaxaca, Mexico.*
> Submitted to *Energy Economics*.

- **Level 1 (macro):** Markowitz mean-variance optimization over 20 candidate sites using 25 years
  (2000–2024) of ERA5 wind data; correlation pre-filter, k-means wind regimes, efficient frontier.
- **Level 2 (micro):** wake-risk-aware layout search with the Bastankhah-Gaussian wake model,
  using turbine power deficits (Eqs. 12–14) under a land-use budget.
- **Integration loop:** Level 2 wake-loss covariance is fed back into Level 1
  (Σ⁽ᵏ⁺¹⁾ = Σ⁽⁰⁾ + Σʷᵃᵏᵉ, Eq. 16) until the portfolio weights converge (Eq. 17).
- **Economics:** NPV, LCOE and CVaR(95%) with NREL 2024 ATB land-based wind costs.

## Data

All wind data are free and public: ERA5 reanalysis (Hersbach et al., 2020,
https://doi.org/10.1002/qj.3803) hourly 100 m wind speed/direction and 10 m wind speed, retrieved
through the [Open-Meteo Historical Weather API](https://open-meteo.com) (no authentication).

## Turbine model

Acciona AW-77/1500 (1.5 MW, rotor diameter 77 m), **80 m hub height**; datasheet cut-in / rated /
cut-out speeds 3.5 / 11.1 / 25.0 m/s with an idealized cubic ramp (no manufacturer power-curve
table was available). ERA5 provides 100 m and 10 m winds, so hub-height speed is obtained by
power-law extrapolation of the 100 m series with an **hourly shear exponent** computed from the
10 m and 100 m series (clipped to 0–0.6). Wake model: Bastankhah-Gaussian, C_T = 0.8, wake growth
k = 0.075, quadratic energy-deficit superposition (Katic et al., 1986).

## Reproducing the analysis

```bash
pip install -r requirements.txt
python src/download_era5_tehuantepec.py   # 100 m wind speed/direction (20 sites)  [optional: processed copy included]
python src/download_era5_10m.py           # 10 m wind speed (20 sites)
python src/process_capacity_factors.py    # hub-height speeds + capacity factors
python src/level1_portfolio.py            # iteration-0 portfolios (no wake feedback)
python src/level2_layout.py               # best-site layout search + land-density sweep
python src/two_level_loop.py              # iterative Level 1 <-> Level 2 integration
python src/economics.py                   # NPV / LCOE / CVaR on the converged portfolios
python src/make_figures.py && python src/make_figure1.py
```

Raw downloads are not committed (`data/raw*/` is git-ignored); the 100 m series needed by
`process_capacity_factors.py` is provided in `data/processed/`.

## Main results (this repository, `results/`)

Portfolios are minimum-variance points on the efficient frontier of the 7 sites retained by the
correlation pre-filter (rho_max = 0.85): *conservative* = E[CF] >= 20 %, *balanced* = E[CF] >= 35 %,
*aggressive* = 100 % in the highest-yield site (T03); the maximum-Sharpe portfolio is a reference.

| Portfolio | Weights | E[CF] (%) | Std (%) | Sharpe | LCOE (USD/MWh) | NPV (USD M) |
|---|---|---|---|---|---|---|
| Conservative | T03 25 %, T15 66 %, T18 9 % | 20.0 | 17.8 | 1.13 | 130.9 | -318.6 |
| Balanced | T03 42 %, T15 29 %, T18 29 % | 35.0 | 28.8 | 1.22 | 76.0 | -198.2 |
| Aggressive (single site) | T03 100 % | 52.9 | 44.6 | 1.19 | 49.5 | -44.0 |
| Max-Sharpe (reference) | T03 57 %, T18 43 % | 47.0 | 38.1 | 1.24 | 56.9 | -101.2 |

* **Diversification:** the conservative portfolio cuts the standard deviation of capacity factor by
  60.2 % relative to the single best site, at a much lower expected capacity factor.
* **Loop (Eqs. 16-17):** converges after one re-solve (max weight change 1.4e-4 < epsilon = 1e-3; selected
  sites T03, T15, T18 unchanged). The wake-induced capacity-factor loss has a standard deviation of only
  0.4-0.9 percentage points versus 10-45 points for site capacity factors, so Sigma^wake is a very small
  correction to Sigma^(0).
* **Level 2:** with a land budget of 36 D^2 per turbine (about 7 MW/km^2 for this turbine) the AEP-maximizing
  layout (6D x 6D, staggered rows) is also the minimum wake-risk layout, i.e. no trade-off appears; the same
  holds at 25 and 50 D^2. Only at 30 D^2 do the layouts differ (wake risk -14.1 % for -0.12 % AEP;
  `results/layout_density_sweep.csv`).
* The k-means clusters are wind regimes (high / moderate / low wind), not contiguous geographic zones.
* The 20 % and 35 % return floors are conventions chosen to define representative frontier points; an earlier
  development version used a percentile rule that produced a degenerate conservative portfolio and was
  replaced.

## Limitations

* ERA5 (≈0.25°) resolves the Isthmus wind corridor only coarsely; capacity factors are
  reanalysis-based, not measurement-validated.
* Idealized power curve and constant C_T; the 80 m wind speeds are extrapolated, not observed.
* Σʷᵃᵏᵉ enters Level 1 only through covariance (as in Eq. 16); mean wake losses are not deducted
  from the expected return.
* Economics use a single representative price (USD 45/MWh), not nodal prices.

## License

Code: MIT (see `LICENSE`). ERA5 data are produced by the Copernicus Climate Change Service and
are subject to the Copernicus license; only compact derived series are redistributed here.
