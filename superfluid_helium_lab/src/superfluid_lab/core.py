"""Core physical constants and thermodynamic models for liquid Helium-4.

This module models:
1. Saturated vapor pressure (SVP) phase boundaries (Helium Gas, Helium I, Helium II).
2. The Lambda transition at T_lambda = 2.1768 K.
3. Bose-Einstein Condensate fraction n_0(T) vs Superfluid density fraction rho_s(T)/rho.
4. Quantum zero-point motion and the 25.3 bar solidification threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


# Physical Constants (SI units)
K_B: float = 1.380649e-23  # Boltzmann constant, J/K
H_BAR: float = 1.054571817e-34  # Reduced Planck constant, J*s
M_HE4: float = 4.002602 * 1.66053906660e-27  # Helium-4 atom mass, kg (~6.6465e-27 kg)

# Key Transition Temperatures and Pressures at SVP
T_BOIL_1ATM: float = 4.222  # Boiling point of liquid He-4 at 1 atm, K
T_LAMBDA: float = 2.1768  # Lambda transition temperature at SVP, K
P_SOLIDIFICATION_MIN_BAR: float = 25.3  # Minimum pressure required to freeze He-4 at T -> 0, bar
P_SOLIDIFICATION_MIN_PA: float = 2.53e6  # Minimum pressure in Pascals


class HeliumPhase(str, Enum):
    """Thermodynamic phase classification for Helium-4 at saturated vapor pressure."""
    GAS = "Helium Gas (Vapor)"
    HELIUM_I = "Helium I (Normal Viscous Liquid)"
    HELIUM_II = "Helium II (Superfluid)"


@dataclass(frozen=True)
class LiquidFractions:
    """Superfluid and Bose-Einstein condensate fractions at a given temperature."""
    temperature: float
    superfluid_fraction: float  # rho_s / rho (macroscopic superfluid density fraction)
    normal_fraction: float      # rho_n / rho (normal fluid fraction)
    condensate_fraction: float  # n_0 (single-particle ground state occupancy)
    phase: HeliumPhase


def classify_phase(temperature: float, pressure_bar: float = 1.0) -> HeliumPhase:
    """Classify the phase of Helium-4 at a given temperature and pressure.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin (must be >= 0.0).
    pressure_bar : float, default 1.0
        Pressure in bar.

    Returns
    -------
    HeliumPhase
        HeliumPhase.GAS, HeliumPhase.HELIUM_I, or HeliumPhase.HELIUM_II.
    """
    if temperature < 0.0:
        raise ValueError(f"Temperature cannot be negative: {temperature} K")

    if temperature > T_BOIL_1ATM:
        return HeliumPhase.GAS
    elif temperature > T_LAMBDA:
        return HeliumPhase.HELIUM_I
    else:
        return HeliumPhase.HELIUM_II


def calculate_fractions(temperature: float) -> LiquidFractions:
    """Calculate the two-fluid fractions and BEC condensate fraction at temperature T.

    In strongly-interacting liquid He-4:
    - The macroscopic superfluid density fraction rho_s/rho approaches 1.0 as T -> 0,
      and vanishes at T >= T_lambda (Tisza-Landau two-fluid model).
    - The microscopic single-particle Bose condensate fraction n_0 remains capped
      at approximately 7.2% - 7.5% even at T < 1.0 K due to quantum depletion from
      strong repulsive hard-core atomic interactions (Prisk et al. 2017 NIST neutron scattering).

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    LiquidFractions
        Dataclass containing superfluid_fraction, normal_fraction, condensate_fraction, and phase.
    """
    phase = classify_phase(temperature)

    if phase == HeliumPhase.GAS or phase == HeliumPhase.HELIUM_I:
        return LiquidFractions(
            temperature=temperature,
            superfluid_fraction=0.0,
            normal_fraction=1.0,
            condensate_fraction=0.0,
            phase=phase,
        )

    # In Helium II (T <= T_LAMBDA)
    t_ratio = temperature / T_LAMBDA
    # Empirical two-fluid temperature dependence: rho_n/rho = (T / T_lambda)^5.6
    # rho_s/rho = 1 - (T / T_lambda)^5.6
    normal_ratio = min(1.0, max(0.0, t_ratio ** 5.6))
    superfluid_ratio = 1.0 - normal_ratio

    # Microscopic BEC ground-state fraction n_0(T):
    # Capped at ~7.5% at T=0 K due to quantum interaction depletion,
    # decreasing continuously to 0 at T_lambda
    n0_zero_temp = 0.075  # 7.5% from neutron scattering (Prisk et al., 2017)
    condensate = n0_zero_temp * (1.0 - (t_ratio ** 5.6))
    condensate = max(0.0, min(n0_zero_temp, condensate))

    return LiquidFractions(
        temperature=temperature,
        superfluid_fraction=superfluid_ratio,
        normal_fraction=normal_ratio,
        condensate_fraction=condensate,
        phase=phase,
    )


def requires_pressure_to_freeze(pressure_bar: float) -> bool:
    """Check whether the applied pressure is sufficient to freeze Helium-4.

    Due to large quantum zero-point motion energy (E_zp ~ hbar^2 / (2 * m * d^2))
    exceeding the shallow van der Waals attractive well, Helium-4 never freezes
    at saturated vapor pressure or under atmospheric pressure, even at T = 0 K.
    It requires a minimum pressure of ~25.3 bar to crystallize.

    Parameters
    ----------
    pressure_bar : float
        Applied pressure in bar.

    Returns
    -------
    bool
        True if pressure is sufficient to permit solidification (P >= 25.3 bar).
    """
    return pressure_bar >= P_SOLIDIFICATION_MIN_BAR
