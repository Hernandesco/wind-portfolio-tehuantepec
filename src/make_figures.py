import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from wake_model import turbine_layout
from level1_portfolio import correlation_prefilter, min_variance_weights, sharpe

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

cf_df = pd.read_parquet(os.path.join(DATA_DIR, "capacity_factors.parquet"))
dir_df = pd.read_parquet(os.path.join(DATA_DIR, "wind_direction.parquet"))
site_stats = pd.read_csv(os.path.join(DATA_DIR, "site_stats.csv"), index_col=0)
with open(os.path.join(RESULTS_DIR, "portfolios_converged.json")) as f:
    port_data = json.load(f)
with open(os.path.join(RESULTS_DIR, "economics_result.json")) as f:
    econ = json.load(f)
with open(os.path.join(RESULTS_DIR, "level2_layout_result.json")) as f:
    l2 = json.load(f)
frontier = pd.read_csv(os.path.join(RESULTS_DIR, "efficient_frontier.csv"))


def fig_efficient_frontier():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(frontier["std"] * 100, frontier["ret"] * 100, "-", color="#1f77b4", lw=1.5, label="Efficient frontier")
    for name, marker, color in [("conservative", "o", "#2ca02c"), ("balanced", "s", "#ff7f0e"), ("aggressive", "^", "#d62728")]:
        p = port_data["portfolios"][name]
        ax.scatter(p["std_pct"], p["E_CF_pct"], marker=marker, s=90, color=color, edgecolor="k", zorder=5, label=name.capitalize())
    retained = port_data["retained_sites"]
    ax.scatter(site_stats.loc[retained, "std_cf_pct"], site_stats.loc[retained, "mean_cf_pct"],
               marker="x", color="gray", s=40, label="Retained sites")
    ax.set_xlabel("Portfolio std. dev. of capacity factor (%)")
    ax.set_ylabel("Expected capacity factor (%)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_efficient_frontier.png"), dpi=300)
    plt.close(fig)


def fig_wind_roses():
    retained = port_data["retained_sites"]
    n = len(retained)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(3.2 * ncols, 3.2 * nrows))
    for i, site in enumerate(retained):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="polar")
        d = dir_df[site].values
        bins = np.linspace(0, 360, 17)
        hist, _ = np.histogram(d, bins=bins)
        freq = hist / len(d) * 100
        theta = np.radians(0.5 * (bins[:-1] + bins[1:]))
        ax.bar(theta, freq, width=np.radians(20), color="#1f77b4", edgecolor="k", linewidth=0.3)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(site, fontsize=10)
        ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figB1_wind_roses.png"), dpi=250)
    plt.close(fig)


def fig_correlation_heatmap():
    corr = cf_df.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figB2_correlation_heatmap.png"), dpi=300)
    plt.close(fig)


def fig_site_map():
    retained = set(port_data["retained_sites"])
    fig, ax = plt.subplots(figsize=(7, 6))
    for site, row in site_stats.iterrows():
        is_retained = site in retained
        ax.scatter(row["lon"], row["lat"], s=180 if is_retained else 90,
                   c=[row["mean_cf_pct"]], cmap="viridis", vmin=site_stats["mean_cf_pct"].min(),
                   vmax=site_stats["mean_cf_pct"].max(),
                   edgecolor="red" if is_retained else "none", linewidth=2 if is_retained else 0)
        ax.annotate(site, (row["lon"], row["lat"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(site_stats["mean_cf_pct"].min(), site_stats["mean_cf_pct"].max()))
    fig.colorbar(sm, ax=ax, label="Mean capacity factor (%)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figB3_site_map.png"), dpi=300)
    plt.close(fig)


def fig_layout_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, key, label in [(axes[0], "aep_maximizing", "AEP-maximizing"), (axes[1], "risk_adjusted", "Risk-adjusted")]:
        cfg = l2[key]
        pos = turbine_layout(4, 4, along_D=cfg["along_D"], cross_D=cfg["cross_D"],
                              stagger_frac=cfg["stagger"], rotation_deg=cfg["rotation"])
        ax.scatter(pos[:, 0], pos[:, 1], s=120, c="#1f77b4", edgecolor="k", zorder=5)
        ax.set_title(f"{label}\nAEP={cfg['aep_gwh']:.1f} GWh/yr, corr={cfg['mean_corr']:.3f}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_layout_comparison.png"), dpi=300)
    plt.close(fig)


def fig_rhomax_sensitivity():
    rhos = np.linspace(0.5, 0.99, 15)
    n_retained, sh = [], []
    for rho in rhos:
        retained, _ = correlation_prefilter(cf_df, rho)
        n_retained.append(len(retained))
        mu = cf_df[retained].mean().values
        cov = cf_df[retained].cov().values
        w = min_variance_weights(cov, np.percentile(mu, 40), mu)
        sh.append(sharpe(w, mu, cov) if w is not None else np.nan)
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(rhos, n_retained, "o-", color="#1f77b4")
    ax1.set_xlabel(r"Correlation pre-filter threshold $\rho_{\max}$")
    ax1.set_ylabel("Number of retained sites", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(rhos, sh, "s--", color="#d62728")
    ax2.set_ylabel("Conservative portfolio Sharpe-style ratio", color="#d62728")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_rhomax_sensitivity.png"), dpi=300)
    plt.close(fig)


def fig_economics():
    names = ["conservative", "balanced", "aggressive"]
    lcoe = [econ[n]["lcoe_usd_per_mwh"] for n in names]
    npv = [econ[n]["npv_usd_m"] for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar(names, lcoe, color="#1f77b4")
    ax1.set_ylabel("LCOE (USD/MWh)")
    ax2.bar(names, npv, color="#ff7f0e")
    ax2.set_ylabel("NPV (USD M)")
    ax2.axhline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figB5_economics_comparison.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_efficient_frontier()
    fig_wind_roses()
    fig_correlation_heatmap()
    fig_site_map()
    fig_layout_comparison()
    fig_rhomax_sensitivity()
    fig_economics()
    print("Saved all figures to", FIG_DIR)
