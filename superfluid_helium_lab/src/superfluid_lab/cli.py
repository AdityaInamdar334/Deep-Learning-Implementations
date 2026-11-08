"""Command-line interface for the Superfluid Helium Physics Lab.

Usage:
    PYTHONPATH=src python -m superfluid_lab.cli --temperature 1.9 --out results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from superfluid_lab.core import (
    K_B,
    H_BAR,
    M_HE4,
    T_BOIL_1ATM,
    T_LAMBDA,
    P_SOLIDIFICATION_MIN_BAR,
    classify_phase,
    calculate_fractions,
    requires_pressure_to_freeze,
)
from superfluid_lab.landau import (
    C_S,
    DELTA_KB,
    P0_OVER_HBAR,
    landau_critical_velocity,
    second_sound_velocity,
)


def generate_report(temperature: float, pressure_bar: float = 1.0) -> Dict[str, Any]:
    """Generate a comprehensive thermodynamic and quantum state report for Helium-4."""
    phase = classify_phase(temperature, pressure_bar)
    fractions = calculate_fractions(temperature)
    v_c, p_star = landau_critical_velocity()
    c_2 = second_sound_velocity(temperature)
    can_solidify = requires_pressure_to_freeze(pressure_bar)

    report = {
        "query_parameters": {
            "temperature_k": temperature,
            "pressure_bar": pressure_bar,
        },
        "phase_classification": {
            "phase": phase.value,
            "is_superfluid": (temperature <= T_LAMBDA),
            "boiling_point_1atm_k": T_BOIL_1ATM,
            "lambda_point_svp_k": T_LAMBDA,
        },
        "two_fluid_fractions": {
            "superfluid_fraction_rho_s_over_rho": round(fractions.superfluid_fraction, 4),
            "normal_fraction_rho_n_over_rho": round(fractions.normal_fraction, 4),
            "bec_condensate_fraction_n0": round(fractions.condensate_fraction, 4),
            "interpretation": (
                "Superfluidity does not require a 100% single-particle condensate. "
                "Strong repulsive hard-core interactions deplete the ground-state BEC fraction "
                "to ~7.5% at low T, while macroscopic superfluid density fraction rho_s/rho approaches 100%."
            ),
        },
        "excitation_spectrum_and_critical_velocity": {
            "acoustic_sound_speed_c_s_m_per_s": C_S,
            "roton_gap_delta_over_kb_k": DELTA_KB,
            "roton_momentum_p0_over_hbar_inv_angstrom": P0_OVER_HBAR / 1e10,
            "landau_critical_velocity_vc_m_per_s": round(v_c, 2),
            "roton_minimum_momentum_p_star_kg_m_per_s": f"{p_star:.4e}",
            "second_sound_velocity_c2_m_per_s": round(c_2, 2),
        },
        "quantum_zero_point_solidification": {
            "solidification_possible_at_query_pressure": can_solidify,
            "minimum_solidification_pressure_bar": P_SOLIDIFICATION_MIN_BAR,
            "mechanism": (
                "Helium-4 remains liquid down to absolute zero at ambient pressure "
                "because large quantum zero-point kinetic energy prevents lattice crystallization "
                "unless compressed to >= 25.3 bar."
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Superfluid Helium Physics Lab - Thermodynamic & Quantum Hydrodynamics Analyzer."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.9,
        help="Target temperature in Kelvin (default: 1.9 K, operating temperature of the CERN LHC).",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Ambient pressure in bar (default: 1.0 bar).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory to save summary report and figures (default: results).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also generate publication figures.",
    )
    args = parser.parse_args()

    report = generate_report(args.temperature, args.pressure)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "superfluid_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("=" * 72)
    print("         SUPERFLUID HELIUM-4 QUANTUM HYDRODYNAMICS REPORT")
    print("=" * 72)
    print(f"Temperature:            {args.temperature:.4f} K")
    print(f"Pressure:               {args.pressure:.2f} bar")
    print(f"Phase:                  {report['phase_classification']['phase']}")
    print(f"Lambda Transition:      T_lambda = {T_LAMBDA:.4f} K")
    print("-" * 72)
    print(f"Superfluid Fraction:    rho_s / rho = {report['two_fluid_fractions']['superfluid_fraction_rho_s_over_rho'] * 100:.2f}%")
    print(f"Normal Fluid Fraction:  rho_n / rho = {report['two_fluid_fractions']['normal_fraction_rho_n_over_rho'] * 100:.2f}%")
    print(f"Bose Condensate Frac:   n_0 = {report['two_fluid_fractions']['bec_condensate_fraction_n0'] * 100:.2f}% (Prisk et al. 2017)")
    print("-" * 72)
    print(f"Phonon Sound Speed:     c_s = {report['excitation_spectrum_and_critical_velocity']['acoustic_sound_speed_c_s_m_per_s']:.1f} m/s")
    print(f"Roton Minimum Gap:      Delta / k_B = {DELTA_KB:.2f} K")
    print(f"Landau Critical Speed:  v_c = {report['excitation_spectrum_and_critical_velocity']['landau_critical_velocity_vc_m_per_s']:.2f} m/s")
    print(f"Second Sound Speed:     c_2 = {report['excitation_spectrum_and_critical_velocity']['second_sound_velocity_c2_m_per_s']:.2f} m/s")
    print("-" * 72)
    print(f"Can Solidify at P:      {report['quantum_zero_point_solidification']['solidification_possible_at_query_pressure']} (Requires >= {P_SOLIDIFICATION_MIN_BAR} bar)")
    print("=" * 72)
    print(f"Report saved to: {json_path}")

    if args.plot:
        from superfluid_lab.plot import generate_all_figures
        fig_dir = Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        generate_all_figures(fig_dir)
        print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
