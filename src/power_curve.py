"""
Idealized power curve for the Vestas V150-4.2 MW turbine (rotor diameter 150 m,
hub height 100 m), calibrated to manufacturer cut-in / rated / cut-out speeds.
Cubic interpolation between cut-in and rated speed (standard IEC-class
approximation used when a full manufacturer power-curve table is unavailable).
"""
import numpy as np

RATED_POWER_MW = 4.2
CUT_IN = 3.0
RATED_SPEED = 12.0
CUT_OUT = 24.5
ROTOR_DIAMETER_M = 150.0
HUB_HEIGHT_M = 100.0


def capacity_factor(wind_speed_ms):
    """Vectorized capacity factor (fraction of rated power) for wind speed(s) in m/s."""
    v = np.asarray(wind_speed_ms, dtype=float)
    cf = np.zeros_like(v)
    ramp = (v >= CUT_IN) & (v < RATED_SPEED)
    cf[ramp] = ((v[ramp] ** 3 - CUT_IN ** 3) / (RATED_SPEED ** 3 - CUT_IN ** 3))
    rated = (v >= RATED_SPEED) & (v <= CUT_OUT)
    cf[rated] = 1.0
    return np.clip(cf, 0.0, 1.0)
