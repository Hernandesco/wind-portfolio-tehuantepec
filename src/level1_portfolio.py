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
                # drop the one with lower expected return
                if means[si] >= means[sj]:
                    dropped.add(sj)
                else:
                    dropped.add(si)
    retained = [s for s in sites if s not in dropped]
    return retained, corr


def assign_regimes(cf_df, retained, n_clusters=3):
    X = cf_df[retained].T.values
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X)
    labels = km.labels_
    # order clusters by mean longitude (west/central/east) using site_stats
    return dict(zip(retained, labels))


def min_variance_weights(cov, target_return, mu, rho_max_pairwise, corr):
    n = len(mu)
    x0 = np.ones(n) / n

    def objective(w):
        return w @ cov @ w

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w @ mu - target_return},
    ]
    bounds = [(0.0, 1.0)] * n
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else None


def sharpe(w, mu, cov):
    ret = w @ mu
    std = np.sqrt(w @ cov @ w)
    return ret / std if std > 0 else 0.0


def main():
    cf_df = pd.read_parquet(os.path.join(DATA_DIR, "capacity_factors.parquet"))
    site_stats = pd.read_csv(os.path.join(DATA_DIR, "site_stats.csv"), index_col=0)

    retained, corr = correlation_prefilter(cf_df, RHO_MAX)
    print(f"Retained {len(retained)}/{len(cf_df.columns)} sites after rho_max={RHO_MAX} filter:", retained)

    regimes = assign_regimes(cf_df, retained)
    print("Cluster regimes:", regimes)

    mu = cf_df[retained].mean().values
    cov = cf_df[retained].cov().values

    # Efficient frontier: sweep target returns
    lo, hi = mu.min(), mu.max()
    targets = np.linspace(lo, hi, 40)
    frontier = []
    for t in targets:
        w = min_variance_weights(cov, t, mu, RHO_MAX, corr)
        if w is not None:
            ret = w @ mu
            std = np.sqrt(w @ cov @ w)
            frontier.append({"target": t, "ret": ret, "std": std, "sharpe": sharpe(w, mu, cov),
                              "weights": dict(zip(retained, w))})
    frontier_df = pd.DataFrame([{k: v for k, v in f.items() if k != "weights"} for f in frontier])
    frontier_df.to_csv(os.path.join(RESULTS_DIR, "efficient_frontier.csv"), index=False)

    # Conservative: min-variance with modest return floor (25th pct of retained means)
    ret_floor_cons = np.percentile(mu, 40)
    w_cons = min_variance_weights(cov, ret_floor_cons, mu, RHO_MAX, corr)

    # Balanced: maximize Sharpe ratio numerically
    def neg_sharpe(w):
        return -sharpe(w, mu, cov)
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res_bal = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    w_bal = res_bal.x

    # Aggressive: full concentration in best site
    best_idx = int(np.argmax(mu))
    w_agg = np.zeros(n)
    w_agg[best_idx] = 1.0

    portfolios = {}
    for name, w in [("conservative", w_cons), ("balanced", w_bal), ("aggressive", w_agg)]:
        portfolios[name] = {
            "E_CF_pct": float(w @ mu * 100),
            "std_pct": float(np.sqrt(w @ cov @ w) * 100),
            "sharpe": float(sharpe(w, mu, cov)),
            "weights": {s: float(wi) for s, wi in zip(retained, w) if wi > 0.005},
        }
        print(name, portfolios[name])

    with open(os.path.join(RESULTS_DIR, "portfolios.json"), "w") as f:
        json.dump({
            "rho_max": RHO_MAX,
            "retained_sites": retained,
            "regimes": {k: int(v) for k, v in regimes.items()},
            "portfolios": portfolios,
        }, f, indent=2)

    print("\nSaved efficient_frontier.csv and portfolios.json to", RESULTS_DIR)


if __name__ == "__main__":
    main()
