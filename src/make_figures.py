import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from wake_model import turbine_layout, D
from level1_portfolio import (correlation_prefilter, min_variance_weights, sharpe,
                              CONSERVATIVE_RETURN_FLOOR)
import level2_layout as l2

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

cf_df = pd.read_parquet(os.path.join(DATA_DIR, "capacity_factors.parquet"))
dir_df = pd.read_parquet(os.path.join(DATA_DIR, "wind_direction.parquet"))
site_stats = pd.read_csv(os.path.join(DATA_DIR, "site_stats.csv"), index_col=0)
port_data = json.load(open(os.path.join(RESULTS_DIR, "portfolios_converged.json")))
econ = json.load(open(os.path.join(RESULTS_DIR, "economics_result.json")))
frontier = pd.read_csv(os.path.join(RESULTS_DIR, "efficient_frontier.csv"))
NAMES = ["conservative", "balanced", "aggressive"]


def fig_efficient_frontier():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(frontier["std"] * 100, frontier["ret"] * 100, "-", color="#1f77b4", lw=1.5,
            label="Efficient frontier")
    retained = port_data["retained_sites"]
    ax.scatter(site_stats.loc[retained, "std_cf_pct"], site_stats.loc[retained, "mean_cf_pct"],
               marker="x", color="gray", s=40, label="Retained sites")
    offsets = {"T06": (6, -12), "T11": (-4, -14), "T15": (-18, 6), "T16": (-26, 6),
               "T20": (6, 6), "T18": (6, -12), "T03": (-24, -14)}
    for s in retained:
        ax.annotate(s, (site_stats.loc[s, "std_cf_pct"], site_stats.loc[s, "mean_cf_pct"]),
                    xytext=offsets.get(s, (5, 5)), textcoords="offset points", fontsize=8, color="#555555")
    for name, marker, color in [("conservative", "o", "#2ca02c"), ("balanced", "s", "#ff7f0e"),
                                ("aggressive", "^", "#d62728")]:
        p = port_data["portfolios"][name]
        ax.scatter(p["std_pct"], p["E_CF_pct"], marker=marker, s=90, color=color, edgecolor="k",
                   zorder=5, label=name.capitalize())
    ax.set_xlabel("Portfolio std. dev. of capacity factor (%)")
    ax.set_ylabel("Expected capacity factor (%)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_efficient_frontier.png"), dpi=300)
    plt.close(fig)


def fig_wind_roses():
    sites = list(cf_df.columns)
    fig = plt.figure(figsize=(9, 7.4))
    for i, site in enumerate(sites):
        ax = fig.add_subplot(4, 5, i + 1, projection="polar")
        d = dir_df[site].values
        bins = np.linspace(0, 360, 17)
        hist, _ = np.histogram(d, bins=bins)
        theta = np.radians(0.5 * (bins[:-1] + bins[1:]))
        ax.bar(theta, hist / len(d) * 100, width=np.radians(20), color="#1f77b4",
               edgecolor="k", linewidth=0.3)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(site, fontsize=11)
        ax.tick_params(labelsize=7)
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
    vmin, vmax = site_stats["mean_cf_pct"].min(), site_stats["mean_cf_pct"].max()
    fig, ax = plt.subplots(figsize=(7, 6))
    for site, row in site_stats.iterrows():
        r = site in retained
        ax.scatter(row["lon"], row["lat"], s=180 if r else 90, c=[row["mean_cf_pct"]],
                   cmap="viridis", vmin=vmin, vmax=vmax,
                   edgecolor="red" if r else "none", linewidth=2 if r else 0)
        ax.annotate(site, (row["lon"], row["lat"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, label="Mean capacity factor (%)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figB3_site_map.png"), dpi=300)
    plt.close(fig)


def fig_layout_comparison(area=30.0):
    """Layouts at the land budget where AEP-maximizing and risk-adjusted layouts differ."""
    bins, _ = l2.build_wind_bins(l2.SITE)
    _, a, r, _ = l2.search_site(l2.SITE, bins, max_area=area)
    panels = [("(a) AEP-maximizing", a), ("(b) Risk-adjusted", r)]
    lays = [turbine_layout(l2.N_ROWS, l2.N_COLS, along_D=row["along_D"], cross_D=row["cross_D"],
                           stagger_frac=row["stagger"], rotation_deg=row["rotation"]) for _, row in panels]
    xmin = min(p[:, 0].min() for p in lays) - 1.5 * D
    xmax = max(p[:, 0].max() for p in lays) + 6.5 * D
    ymin = min(p[:, 1].min() for p in lays) - 1.5 * D
    ymax = max(p[:, 1].max() for p in lays) + 1.5 * D
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), sharex=True, sharey=True)
    for ax, (label, row), pos in zip(axes, panels, lays):
        ax.scatter(pos[:, 0], pos[:, 1], s=70, c="#1f77b4", edgecolor="k", zorder=5)
        ax_x = xmax - 2.0 * D
        ax.annotate("", xy=(ax_x, 3.0 * D), xytext=(ax_x, -3.0 * D),
                    arrowprops=dict(arrowstyle="<|-|>", color="#B22222", lw=2))
        ax.text(ax_x, -3.6 * D, "dominant\nN-S wind", color="#B22222", fontsize=7.5, ha="center", va="top")
        ax.set_title(f"{label}: {row['along_D']:.0f}D x {row['cross_D']:.0f}D, stagger {row['stagger']:.2f}"
                     f"\nAEP = {row['aep_gwh']:.2f} GWh/yr, WR = {row['wake_risk']:.4f} MW$^2$, "
                     f"corr = {row['mean_corr']:.3f}", fontsize=7.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("East-West (m)")
        ax.set_ylabel("North-South (m)")
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
        w = min_variance_weights(cov, CONSERVATIVE_RETURN_FLOOR, mu)
        sh.append(sharpe(w, mu, cov) if w is not None else np.nan)
    pd.DataFrame({"rho_max": rhos, "n_retained": n_retained, "conservative_sharpe": sh}).to_csv(
        os.path.join(RESULTS_DIR, "rhomax_sensitivity.csv"), index=False)
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


def fig_pareto():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "layout_search.csv"))
    best = df.loc[df["aep_gwh"].idxmax()]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    sc = ax.scatter(df["aep_gwh"], df["wake_risk"], c=df["stagger"], cmap="viridis", s=40, edgecolor="none")
    fig.colorbar(sc, ax=ax, label="Row stagger fraction")
    ax.scatter([best["aep_gwh"]], [best["wake_risk"]], s=160, facecolor="none", edgecolor="red", lw=2,
               label="AEP-maximizing = risk-adjusted")
    ax.set_xlabel("Annual energy production (GWh/yr)")
    ax.set_ylabel(r"Wake risk WR (MW$^2$)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figB4_pareto_wake_risk.png"), dpi=300)
    plt.close(fig)


def fig_economics():
    lcoe = [econ[n]["lcoe_usd_per_mwh"] for n in NAMES]
    npv = [econ[n]["npv_usd_m"] for n in NAMES]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar(NAMES, lcoe, color="#1f77b4")
    ax1.axhline(45, color="k", ls="--", lw=1)
    ax1.text(2.4, 47, "USD 45/MWh price", ha="right", fontsize=8)
    ax1.set_ylabel("LCOE (USD/MWh)")
    ax2.bar(NAMES, npv, color="#ff7f0e")
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
    fig_pareto()
    fig_economics()
    print("Saved all figures to", FIG_DIR)
