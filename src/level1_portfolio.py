"""
Level 1: Markowitz mean-variance portfolio optimization over Tehuantepec
candidate sites. Correlation pre-filter (rho_max), k-means regime clustering,
efficient frontier, and three named portfolios (conservative/balanced/aggressive).
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RHO_MAX = 0.85
CONSERVATIVE_RETURN_FLOOR = 0.20  # conservative portfolio: minimum variance s.t. E[CF] >= 20%
BALANCED_RETURN_FLOOR = 0.35      # balanced portfolio: minimum variance s.t. E[CF] >= 35%


def correlation_prefilter(cf_df, rho_max=RHO_MAX):
    corr = cf_df.corr()
    means = cf_df.mean()
    sites = list(cf_df.columns)
    dropped = set()
    for i, si in enumerate(sites):
        for sj in sites[i + 1:]:
            if si in dropped or sj in dropped:
                continue
            if corr.loc[si, sj] >= rho_max:
                dropped.add(sj if means[si] >= means[sj] else si)
    return [s for s in sites if s not in dropped], corr


def assign_regimes(cf_df, retained, n_clusters=3):
    """k-means on the capacity-factor series; clusters named by their mean capacity factor
    (the clusters are wind regimes, not contiguous geographic zones)."""
    X = cf_df[retained].T.values
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X).labels_
    mean_cf = [cf_df[retained].values[:, labels == k].mean() for k in range(n_clusters)]
    order = np.argsort(mean_cf)[::-1]
    names = {int(k): n for k, n in zip(order, ["High-wind regime", "Moderate-wind regime", "Low-wind regime"])}
    return {s: names[int(l)] for s, l in zip(retained, labels)}


def min_variance_weights(cov, target_return, mu):
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: w @ mu - target_return}]
    res = minimize(lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else None


def sharpe(w, mu, cov):
    std = np.sqrt(w @ cov @ w)
    return (w @ mu) / std if std > 0 else 0.0


def solve_portfolios(mu, cov):
    """Representative frontier points: conservative (E[CF] >= 20%), balanced (E[CF] >= 35%),
    aggressive (100% in the highest-yield site) and, for reference, the maximum-Sharpe portfolio."""
    n = len(mu)
    res = minimize(lambda w: -sharpe(w, mu, cov), np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                   options={"maxiter": 1000, "ftol": 1e-12})
    w_agg = np.zeros(n)
    w_agg[int(np.argmax(mu))] = 1.0
    return {"conservative": min_variance_weights(cov, CONSERVATIVE_RETURN_FLOOR, mu),
            "balanced": min_variance_weights(cov, BALANCED_RETURN_FLOOR, mu),
            "aggressive": w_agg, "max_sharpe": res.x}


def summarize(w, mu, cov, sites, regimes):
    reg = {}
    for s, wi in zip(sites, w):
        reg[regimes[s]] = reg.get(regimes[s], 0.0) + float(wi)
    return {
        "E_CF_pct": float(w @ mu * 100),
        "std_pct": float(np.sqrt(w @ cov @ w) * 100),
        "sharpe": float(sharpe(w, mu, cov)),
        "weights": {s: float(wi) for s, wi in zip(sites, w) if wi > 0.005},
        "regime_weights": {k: v for k, v in reg.items() if v > 0.005},
    }


def main():
    cf_df = pd.read_parquet(os.path.join(DATA_DIR, "capacity_factors.parquet"))
    site_stats = pd.read_csv(os.path.join(DATA_DIR, "site_stats.csv"), index_col=0)

    retained, _ = correlation_prefilter(cf_df, RHO_MAX)
    print(f"Retained {len(retained)}/{cf_df.shape[1]} sites (rho_max={RHO_MAX}):", retained)
    regimes = assign_regimes(cf_df, retained)
    print("Regimes:", regimes)

    mu = cf_df[retained].mean().values
    cov = cf_df[retained].cov().values

    rows = []
    for t in np.linspace(mu.min(), mu.max(), 40):
        w = min_variance_weights(cov, t, mu)
        if w is not None:
            rows.append({"target": t, "ret": w @ mu, "std": np.sqrt(w @ cov @ w), "sharpe": sharpe(w, mu, cov)})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "efficient_frontier.csv"), index=False)

    ws = solve_portfolios(mu, cov)
    portfolios = {k: summarize(w, mu, cov, retained, regimes) for k, w in ws.items()}
    for k, v in portfolios.items():
        print(k, v)

    with open(os.path.join(RESULTS_DIR, "portfolios.json"), "w") as f:
        json.dump({"rho_max": RHO_MAX, "retained_sites": retained, "regimes": regimes,
                   "portfolios": portfolios}, f, indent=2)
    print("saved portfolios.json / efficient_frontier.csv")


if __name__ == "__main__":
    main()
