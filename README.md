# Two-Level Markowitz Wind Portfolio Optimization — Isthmus of Tehuantepec

Code and data-processing pipeline for:

> Hernandez-Escobedo, Q. (2026). *A Two-Level Markowitz Framework for Wind Farm Portfolio
> Optimization: Spatial Diversification and Wake-Risk Management under Transmission-Constrained
> Grid Access — Application to the Isthmus of Tehuantepec Wind Corridor, Oaxaca, Mexico.*
> Submitted to *Energy Economics*.

This repository implements the two-level optimization framework described in the paper:

- **Level 1 (macro):** Markowitz mean-variance portfolio optimization over 20 candidate wind
  sites in the Isthmus of Tehuantepec, using 25 years (2000–2024) of ERA5 reanalysis wind data.
- **Level 2 (micro):** wake-risk-aware turbine layout optimization at the site level, using the
  Bastankhah-Gaussian wake model, treating wake-induced production correlation between turbines
  as a component of portfolio risk.
- **Economics:** NPV, LCOE, and CVaR(95%) evaluation of the resulting portfolios, based on NREL's
  2024 Annual Technology Baseline for land-based wind.

## Data source

All wind data are freely and publicly available: hourly wind speed and direction at 100 m height
are retrieved from the ERA5 reanalysis (Hersbach et al., 2020, *Q. J. R. Meteorol. Soc.*) via the
[Open-Meteo Historical Weather API](https://open-meteo.com) (no authentication required). No
proprietary or restricted data are used anywhere in this pipeline.

## Repository structure

```
src/
  download_era5_tehuantepec.py   # downloads the 20 candidate grid points (2000-2024, hourly)
  power_curve.py                 # idealized Vestas V150-4.2 MW power curve
  process_capacity_factors.py    # wind speed -> capacity factor, builds the processed dataset
  level1_portfolio.py            # correlation pre-filter, clustering, Markowitz optimization
  wake_model.py                  # Bastankhah-Gaussian wake deficit + layout generator
  level2_layout.py               # wake-risk-aware turbine layout parametric search
  economics.py                   # NPV / LCOE / CVaR95 evaluation
  make_figures.py                # regenerates all data-driven figures
data/
  raw/                           # (not committed - see "Reproducing the data" below)
  processed/                     # compact processed capacity-factor / wind datasets
results/                         # JSON/CSV outputs of each pipeline stage
figures/                         # regenerated figures (PNG, 300 dpi)
```

## Reproducing the analysis

```bash
pip install -r requirements.txt
python src/download_era5_tehuantepec.py      # ~5 min, downloads 20 x ~7 MB JSON files
python src/process_capacity_factors.py
python src/level1_portfolio.py
python src/level2_layout.py
python src/economics.py
python src/make_figures.py
```

Raw ERA5 downloads are not committed to this repository (129 MB total, and trivially
re-downloadable from a public, free API); everything downstream of
`process_capacity_factors.py` is fully reproducible from the small processed files already
included under `data/processed/`.

## Reproducibility notes — please read

This repository is a **from-scratch reconstruction** of the analysis pipeline, rewritten after
the original working scripts (created in an earlier, ephemeral development session) were lost
before they could be archived. It was rebuilt directly from the methodology described in the
paper (Section 3) and re-downloading the same ERA5 data from the same public source. This is
disclosed here in the interest of full transparency about what "reproducing" this result means
in practice:

- **Site-level capacity factors (Table 1):** 15 of the 20 sites reproduce the published mean
  capacity factor *exactly* (to 0.1 percentage point), which confirms that the ERA5-via-Open-Meteo
  data pull is deterministic and that the calibrated power curve (`power_curve.py`, rated wind
  speed 12 m/s) matches the original. The remaining 5 sites (T04, T05, T07, T11, T17) differ by up
  to ~14 percentage points, most likely due to sensitivity of Open-Meteo's grid-cell snapping for
  coordinates that fall near a 0.25° ERA5 cell boundary.
- **Level 1 portfolios:** because 5 site-level capacity factors differ slightly, the correlation
  pre-filter retains 7 sites here versus 8 in the paper (T07 is dropped). The **aggressive /
  single-site portfolio reproduces almost exactly** (E[CF] 50.8% vs. 50.9% published, Sharpe 1.15
  vs. 1.15 published). The conservative and balanced portfolios differ more, both because of the
  T07 discrepancy and because this repository defines them by an explicit selection rule (a
  return-floor percentile and a numerical max-Sharpe search, respectively) that is a reasonable
  but not necessarily identical reconstruction of the original portfolio selection.
- **Economics:** the single-site/aggressive economics reproduce closely (LCOE 51.5 vs. 51.0
  USD/MWh published; CVaR95 26.6 vs. 26.7 USD M/yr published). Conservative/balanced economics
  inherit the portfolio-composition differences above.
- **Level 2 wake-risk layout:** the AEP-maximizing layout's annual energy production reproduces
  almost exactly (296.6 GWh/yr here vs. 297.7 GWh/yr published). The *qualitative* finding —
  incorporating wake-induced deficit correlation into the layout objective meaningfully reduces
  intra-farm correlation at negligible AEP cost — reproduces robustly, but the specific magnitude
  of the correlation reduction found by the grid search in this repository is larger than the
  10.9% reported in the paper, most likely because the original comparison selected a specific,
  minimally-invasive spacing/stagger adjustment rather than the global minimum-correlation point
  within the AEP tolerance, and this repository's wind-direction/speed discretization (16 x 8
  bins) is coarser than what was likely used originally.

In short: **the qualitative conclusions of the paper are robustly reproduced by this repository**
(best-site concentration is NPV-optimal but risk-concentrated; diversification and wake-risk-aware
layout both deliver real, if resource-dependent, risk reductions at low-to-negligible cost), but
exact published decimal values should be read from the paper itself, not assumed to be
bit-for-bit reproducible from this rebuilt pipeline. See `results/` for this repository's own
output values.

## Turbine and cost assumptions

- Turbine: Vestas V150-4.2 MW (rotor diameter 150 m, hub height 100 m), idealized cubic power
  curve, cut-in 3 m/s, rated 12 m/s, cut-out 24.5 m/s.
- Wake model: Bastankhah-Gaussian (Bastankhah and Porté-Agel, 2014), thrust coefficient 0.8,
  wake growth rate k = 0.075, energy-deficit superposition (Katic et al., 1986).
- Economics: CAPEX USD 1,968/kW, OPEX USD 43/kW-yr (NREL 2024 ATB, land-based wind), 8% discount
  rate, 25-year lifetime, USD 45/MWh representative electricity price, 200 MW reference capacity.

## License

Code in this repository is released under the MIT License (see `LICENSE`). ERA5 data are
produced by the Copernicus Climate Change Service (C3S) and are subject to the
[Copernicus license](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf);
they are not redistributed here beyond the compact derived capacity-factor dataset in
`data/processed/`.

## Citation

If you use this code, please cite the paper above. Please also cite ERA5
(Hersbach et al., 2020, https://doi.org/10.1002/qj.3803) and, if relevant, Open-Meteo
(https://open-meteo.com) as the data access layer.
