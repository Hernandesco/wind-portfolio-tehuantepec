"""
Iterative two-level integration (paper Section 3.4, Eqs. 16-17).

Step 1-2  Solve Level 1 with the raw ERA5 covariance Sigma^(0).
Step 3    For every site selected by Level 1, use its risk-adjusted Level 2 layout (other sites
          keep the AEP-maximizing layout) and compute the hourly wake-induced capacity-factor
          loss L_i(t); Sigma^wake is the covariance matrix of these loss series across sites.
Step 4    Sigma^(k+1) = Sigma^(0) + Sigma^wake                                    (Eq. 16)
Step 5-6  Re-solve Level 1 and repeat until max ||w^(k) - w^(k-1)|| < epsilon     (Eq. 17)
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import level1_portfolio as l1
import level2_layout as l2

DATA_DIR = l1.DATA_DIR
RESULTS_DIR = l1.RESULTS_DIR
EPSILON = 1e-3
MAX_ITER = 10
SELECT_THRESHOLD = 0.005


def main():
    cf_df = pd.read_parquet(os.path.join(DATA_DIR, "capacity_factors.parquet"))
    site_stats = pd.read_csv(os.path.join(DATA_DIR, "site_stats.csv"), index_col=0)
    retained, _ = l1.correlation_prefilter(cf_df, l1.RHO_MAX)
    regimes = l1.assign_regimes(cf_df, retained)
    mu = cf_df[retained].mean().values
    cov0 = cf_df[retained].cov().values

    # Level 2 for every retained site (cached): wake-loss series for both layout types
    loss = {}
    layout_info = {}
    for s in retained:
        bins, hour_idx = l2.build_wind_bins(s)
        df, a, r, lay = l2.search_site(s, bins)
        loss[s] = {key: l2.wake_loss_series(s, lay[key][1], hour_idx, bins) for key in lay}
        layout_info[s] = {
            "aep_maximizing": {k: float(a[k]) for k in ("along_D", "cross_D", "stagger", "aep_gwh", "mean_corr")},
            "risk_adjusted": {k: float(r[k]) for k in ("along_D", "cross_D", "stagger", "aep_gwh", "mean_corr")},
            "mean_wake_loss_aep_max_pct": float(loss[s]["aep_maximizing"].mean() * 100),
            "mean_wake_loss_risk_adj_pct": float(loss[s]["risk_adjusted"].mean() * 100),
        }
        print(f"Level 2 done for {s}: corr {a['mean_corr']:.3f} -> {r['mean_corr']:.3f}")

    sigma = cov0.copy()
    history, prev = [], None
    converged_ws = None
    for k in range(MAX_ITER + 1):
        ws = l1.solve_portfolios(mu, sigma)
        selected = sorted({s for w in ws.values() for s, wi in zip(retained, w) if wi > SELECT_THRESHOLD})
        entry = {"iteration": k, "selected_sites": selected,
                 "weights": {name: {s: float(wi) for s, wi in zip(retained, w) if wi > SELECT_THRESHOLD}
                             for name, w in ws.items()}}
        if prev is not None:
            entry["max_weight_change"] = float(max(np.linalg.norm(ws[n] - prev[n]) for n in ws))
        history.append(entry)
        print(f"iter {k}: selected={selected}, change={entry.get('max_weight_change')}")
        if prev is not None and entry["max_weight_change"] < EPSILON:
            converged_ws = ws
            break
        prev = ws
        L = np.column_stack([loss[s]["risk_adjusted" if s in selected else "aep_maximizing"] for s in retained])
        sigma_wake = np.cov(L, rowvar=False)
        sigma = cov0 + sigma_wake
    converged = converged_ws is not None
    final_ws = converged_ws if converged else ws

    portfolios = {k: l1.summarize(w, mu, cov0, retained, regimes) for k, w in final_ws.items()}
    with open(os.path.join(RESULTS_DIR, "portfolios_converged.json"), "w") as f:
        json.dump({"rho_max": l1.RHO_MAX, "retained_sites": retained, "regimes": regimes,
                   "portfolios": portfolios}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "loop_result.json"), "w") as f:
        json.dump({"epsilon": EPSILON, "converged": converged, "iterations_run": len(history) - 1,
                   "history": history, "layout_info": layout_info,
                   "sigma_wake_diag_std_pct": [float(np.sqrt(np.diag(sigma - cov0))[i] * 100)
                                               for i in range(len(retained))],
                   "sites": retained}, f, indent=2)
    print("converged:", converged, "after", len(history) - 1, "re-solves")


if __name__ == "__main__":
    main()
