"""
Idealized power curve for the Acciona AW-77/1500 turbine (1.5 MW, rotor diameter 77 m,
80 m hub height option), built from manufacturer-datasheet cut-in / rated / cut-out
speeds (3.5 / 11.1 / 25.0 m/s) with a cubic ramp between cut-in and rated speed.
A full manufacturer power-curve table was not available, so the cubic ramp is an
approximation; see README for the resulting caveat.
"""
import numpy as np

RATED_POWER_MW = 1.5
CUT_IN = 3.5
RATED_SPEED = 11.1
CUT_OUT = 25.0
ROTOR_DIAMETER_M = 77.0
HUB_HEIGHT_M = 80.0


def capacity_factor(wind_speed_ms):
    """Vectorized capacity factor (fraction of rated power) for wind speed(s) in m/s."""
    v = np.asarray(wind_speed_ms, dtype=float)
    cf = np.zeros_like(v)
    ramp = (v >= CUT_IN) & (v < RATED_SPEED)
    cf[ramp] = (v[ramp] ** 3 - CUT_IN ** 3) / (RATED_SPEED ** 3 - CUT_IN ** 3)
    rated = (v >= RATED_SPEED) & (v <= CUT_OUT)
    cf[rated] = 1.0
    return np.clip(cf, 0.0, 1.0)
