"""Conceptual diagram of the two-level framework (Figure 1)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BG = "#FFFFFF"      # white figure background
BLOCK = "#E6E6E6"   # light gray blocks
fig, ax = plt.subplots(figsize=(7.9, 5.7), facecolor=BG)
ax.set_facecolor(BG)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, fc, ec, title, lines, title_size=12, line_size=10, start=1.35, step=0.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", linespacing=1.25)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - start - i * step, ln, ha="center", va="top", fontsize=line_size)


def arrow(p0, p1, color="#333333", lw=2.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0))


# Inputs
box(0.1, 5.65, 9.8, 1.45, BLOCK, "#666666", "Input data",
    [r"ERA5 hourly wind (100 m and 10 m, 2000$-$2024), 20 candidate sites; "
     r"AW-77/1500 power curve at 80 m hub height",
     r"$\rightarrow$ capacity factors $CF_i(t)$ and site covariance matrix $\Sigma^{(0)}$"],
    start=0.75, step=0.4, line_size=9.6)

# Level 1 and Level 2
box(0.1, 2.3, 3.9, 2.9, BLOCK, "#1F5C99", "Level 1 (Macro)\nSite Portfolio Optimization",
    [r"Markowitz mean-variance QP (Eq. 4$-$8)",
     r"Correlation pre-filter ($\rho_{\max}$) + k-means",
     r"$\rightarrow$ portfolio weights $\mathbf{w}^{*}$"])
box(6.0, 2.3, 3.9, 2.9, BLOCK, "#C97A1C", "Level 2 (Micro)\nTurbine Layout Optimization",
    [r"Bastankhah-Gaussian wake (Eq. 9$-$10)",
     r"Layout search under land budget (Eq. 15)",
     r"$\rightarrow$ wake-risk covariance $\Sigma^{\mathrm{wake}}$"])

# Inputs -> levels
arrow((2.05, 5.63), (2.05, 5.22))
arrow((7.95, 5.63), (7.95, 5.22))

# Level 1 -> Level 2 (selected sites)
arrow((4.05, 4.3), (5.95, 4.3))
ax.text(5.0, 4.42, "selected sites", ha="center", va="bottom", fontsize=10)

# Level 2 -> Level 1 (risk feedback)
arrow((5.95, 3.15), (4.05, 3.15), color="#B22222")
ax.text(5.0, 3.02, r"$\Sigma^{(1)}=\Sigma^{(0)}+\Sigma^{\mathrm{wake}}$", ha="center", va="top",
        fontsize=10, color="#B22222")
ax.text(5.0, 2.65, "(Eq. 16)", ha="center", va="top", fontsize=9.5, color="#B22222")

# Convergence
box(2.6, 0.2, 4.8, 1.35, BLOCK, "#666666", "Convergence check (Eq. 17)",
    [r"iterate until $\|\mathbf{w}^{(k)}-\mathbf{w}^{(k-1)}\|<\epsilon$"],
    start=0.75, line_size=10.2)
arrow((2.05, 2.28), (3.3, 1.57))
arrow((7.95, 2.28), (6.7, 1.57))

fig.savefig(os.path.join(FIG_DIR, "fig1_conceptual_framework.png"), dpi=300, facecolor=fig.get_facecolor())
print("saved fig1")
