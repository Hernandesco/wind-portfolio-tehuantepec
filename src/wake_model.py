"""
Bastankhah-Gaussian wake model (Bastankhah and Porte-Agel, 2014) and a
parametric turbine-layout generator for the Level 2 wake-risk analysis.
"""
import numpy as np

D = 150.0  # rotor diameter, m (Vestas V150)
CT = 0.8   # representative thrust coefficient near rated
K_WAKE = 0.075  # wake growth rate, typical onshore value


def wake_deficit(x, y, ct=CT, k=K_WAKE, d=D):
    """Velocity deficit (fraction of freestream) at downstream distance x,
    lateral offset y, both in meters. x must be > 0 (downstream)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = k * x + d / np.sqrt(8)
    beta = 0.5 * (1 + np.sqrt(1 - ct)) / np.sqrt(1 - ct)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = 1 - np.sqrt(np.clip(1 - ct / (8 * (sigma / d) ** 2), 0, None))
        deficit = a * np.exp(-0.5 * (y / sigma) ** 2)
    deficit = np.where(x > 0, deficit, 0.0)
    return np.nan_to_num(deficit)


def turbine_layout(n_rows=4, n_cols=4, along_D=10.0, cross_D=5.0,
                    stagger_frac=0.0, rotation_deg=0.0, d=D):
    """Generate an n_rows x n_cols grid of turbine (x, y) positions in meters,
    with spacing in rotor diameters, optional row staggering, and rotation."""
    xs, ys = [], []
    for i in range(n_rows):
        row_offset = (i % 2) * stagger_frac * cross_D * d if stagger_frac else 0.0
        for j in range(n_cols):
            x = i * along_D * d
            y = j * cross_D * d + row_offset
            xs.append(x)
            ys.append(y)
    pos = np.column_stack([xs, ys])
    pos -= pos.mean(axis=0)
    theta = np.radians(rotation_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return pos @ rot.T


def effective_speeds(positions, wind_dir_deg, freestream_speed, ct=CT, k=K_WAKE, d=D):
    """Vectorized effective wind speed at each turbine for a single (direction,
    speed) condition, using the energy-deficit superposition of Katic et al.
    (1986): total deficit^2 = sum of individual deficits^2. Returns
    (effective_speed[n], total_deficit[n], pairwise_deficit[n,n]) where
    pairwise_deficit[m, src] is the deficit imposed on turbine m by src
    (0 if src is not upstream of m)."""
    theta = np.radians(270 - wind_dir_deg)  # meteorological -> math convention
    wind_vec = np.array([np.cos(theta), np.sin(theta)])
    perp_vec = np.array([-wind_vec[1], wind_vec[0]])

    downstream = positions @ wind_vec        # (n,)
    lateral = positions @ perp_vec           # (n,)

    dx = downstream[:, None] - downstream[None, :]   # dx[m, src]
    dy = lateral[:, None] - lateral[None, :]
    upstream_mask = dx > 0
    defi = np.where(upstream_mask, wake_deficit(np.where(upstream_mask, dx, 1.0), dy, ct=ct, k=k, d=d), 0.0)
    np.fill_diagonal(defi, 0.0)

    total_deficit = np.clip(np.sqrt((defi ** 2).sum(axis=1)), 0, 1)
    return freestream_speed * (1 - total_deficit), total_deficit, defi
