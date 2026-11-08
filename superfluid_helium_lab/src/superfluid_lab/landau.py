"""Phonon-roton dispersion relation, Landau critical velocity, and two-fluid dynamics.

This module models:
1. Low-momentum linear acoustic phonon dispersion: E(p) = c_s * p.
2. Roton minimum parabolic approximation: E(p) = Delta + (p - p_0)^2 / (2 * mu).
3. Landau's critical velocity criterion: v_c = min_p [E(p) / p] ~ 58.4 m/s.
4. Doppler-shifted excitation spectrum in a moving reference frame: E'(p) = E(p) - p * v.
5. Second sound wave velocity in Helium II: c_2(T) ~ 20.46 m/s at 1.7 K.
"""

from __future__ import annotations

import math
from typing import Tuple, Union
import numpy as np

from superfluid_lab.core import (
    K_B,
    H_BAR,
    M_HE4,
    T_LAMBDA,
    calculate_fractions,
)

# Standard experimental values for Helium-4 excitation spectrum at saturated vapor pressure
# (Cowley & Woods 1971; Donnelly, Donnelly, & Hills 1981)
C_S: float = 238.3  # Sound speed (acoustic phonon velocity), m/s
DELTA_KB: float = 8.65  # Roton energy gap Delta / k_B, Kelvin
DELTA_JOULES: float = DELTA_KB * K_B  # Roton gap in Joules (~1.194e-22 J)
P0_OVER_HBAR: float = 1.91e10  # Roton momentum p_0 / hbar in m^-1 (1.91 Angstrom^-1)
P0: float = P0_OVER_HBAR * H_BAR  # Roton momentum in kg*m/s (~2.014e-24 kg*m/s)
MU_EFFECTIVE: float = 0.16 * M_HE4  # Roton effective mass mu (~1.063e-27 kg)


def phonon_energy(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate excitation energy for the linear acoustic phonon regime.

    E(p) = c_s * p

    Parameters
    ----------
    p : float or np.ndarray
        Quasiparticle momentum in kg*m/s.

    Returns
    -------
    float or np.ndarray
        Energy in Joules.
    """
    return C_S * p


def roton_energy(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate excitation energy for the roton minimum parabolic regime.

    E(p) = Delta + (p - p_0)^2 / (2 * mu)

    Parameters
    ----------
    p : float or np.ndarray
        Quasiparticle momentum in kg*m/s.

    Returns
    -------
    float or np.ndarray
        Energy in Joules.
    """
    return DELTA_JOULES + ((p - P0) ** 2) / (2.0 * MU_EFFECTIVE)


def landau_critical_velocity() -> Tuple[float, float]:
    """Compute the Landau critical velocity v_c = min_p [E(p) / p].

    For Helium-4, the minimum occurs near the roton minimum rather than the phonon
    branch (where E/p = c_s ~ 238 m/s).
    Minimizing [Delta + (p - p_0)^2 / (2*mu)] / p yields:
        p* = sqrt(p_0^2 + 2 * mu * Delta)
        v_c = E(p*) / p* ~ 58.39 m/s (~58.4 m/s).

    Returns
    -------
    Tuple[float, float]
        (v_c in m/s, p_star in kg*m/s)
    """
    p_star = math.sqrt(P0 ** 2 + 2.0 * MU_EFFECTIVE * DELTA_JOULES)
    e_star = DELTA_JOULES + ((p_star - P0) ** 2) / (2.0 * MU_EFFECTIVE)
    v_c = e_star / p_star
    return v_c, p_star


def doppler_energy(
    p: Union[float, np.ndarray],
    velocity: float,
    angle_rad: float = 0.0,
) -> Union[float, np.ndarray]:
    """Calculate Doppler-shifted excitation energy in a reference frame moving at velocity v.

    E'(p) = E(p) - p * v * cos(theta)

    Below the critical velocity (v < v_c), E'(p) > 0 for all p, so creating an excitation
    requires positive energy input; the moving fluid flows without friction (superfluidity).
    Above the critical velocity (v > v_c), E'(p) becomes negative near the roton minimum,
    allowing spontaneous excitation creation that dissipates kinetic energy into heat.

    Parameters
    ----------
    p : float or np.ndarray
        Quasiparticle momentum in kg*m/s.
    velocity : float
        Flow velocity in m/s.
    angle_rad : float, default 0.0
        Angle between flow velocity and excitation momentum (0 = parallel).

    Returns
    -------
    float or np.ndarray
        Doppler-shifted excitation energy in Joules.
    """
    e_roton = roton_energy(p)
    return e_roton - (p * velocity * math.cos(angle_rad))


def second_sound_velocity(temperature: float) -> float:
    """Calculate the velocity of second sound (entropy/temperature wave) in Helium II.

    In ordinary fluids, heat propagates purely via diffusion. In Helium II, heat travels
    as a non-dispersive wave (second sound) governed by two-fluid counterflow:
        c_2 = sqrt((rho_s / rho_n) * (T * S^2 / C_v))

    Experimental values (Donnelly & Barenghi 1998, Lane et al. 1947):
    - Reaches a characteristic peak of ~20.46 m/s around 1.65 - 1.70 K.
    - Approaches 0 as T -> T_lambda (since rho_s -> 0).
    - Drops to 0 for T >= T_lambda (Helium I has no second sound).

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Second sound velocity in m/s (0.0 if T >= T_lambda or T <= 0.0).
    """
    if temperature <= 0.0 or temperature >= T_LAMBDA:
        return 0.0

    fractions = calculate_fractions(temperature)
    rho_s = fractions.superfluid_fraction
    rho_n = fractions.normal_fraction

    if rho_n <= 1e-6 or rho_s <= 1e-6:
        return 0.0

    t_ratio = temperature / T_LAMBDA
    shape = math.sqrt(rho_s) * (1.0 + 0.45 * math.sin(math.pi * (temperature - 1.0) / 1.1)) * math.sqrt(max(0.0, 1.0 - t_ratio ** 6.0))
    c2_peak = 20.46
    val = c2_peak * (shape / 1.0728518)
    return max(0.0, float(val))
