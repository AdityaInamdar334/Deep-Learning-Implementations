"""Publication-grade dark-slate figures for the Superfluid Helium Physics Lab.

Generates:
1. figures/01_phase_transitions_cooling.png - Phase boundaries & cooling path (4.22 K -> 2.17 K -> 1.7 K).
2. figures/02_condensate_vs_superfluid_fraction.png - Superfluid fraction (-> 100%) vs BEC fraction (~7.5%).
3. figures/03_phonon_roton_dispersion.png - Linear phonon + parabolic roton excitation spectrum.
4. figures/04_landau_critical_velocity.png - E(p)/p minimization and Doppler-shifted moving frame energies.
"""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from superfluid_lab.core import (
    K_B,
    H_BAR,
    M_HE4,
    T_BOIL_1ATM,
    T_LAMBDA,
    calculate_fractions,
)
from superfluid_lab.landau import (
    C_S,
    DELTA_KB,
    DELTA_JOULES,
    P0_OVER_HBAR,
    P0,
    MU_EFFECTIVE,
    phonon_energy,
    roton_energy,
    landau_critical_velocity,
    doppler_energy,
)

# Dark-slate theme styling constants
BG_COLOR = "#0e1117"
PANEL_COLOR = "#161b22"
BORDER_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"
MUTED_TEXT = "#8b949e"
GRID_COLOR = "#21262d"

ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_AMBER = "#d29922"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#bc8cff"
ACCENT_CYAN = "#39c5bb"


def apply_theme(fig, axs):
    fig.patch.set_facecolor(BG_COLOR)
    if isinstance(axs, np.ndarray):
        ax_list = axs.flat
    elif isinstance(axs, list):
        ax_list = axs
    else:
        ax_list = [axs]
    for ax in ax_list:
        ax.set_facecolor(PANEL_COLOR)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(1.2)
        ax.tick_params(colors=MUTED_TEXT, labelsize=10)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)


def plot_01_phase_transitions(out_path: Path):
    """Plot 1: Phase transitions & cooling trajectory from He gas to He II."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    apply_theme(fig, ax)

    # Temperature axis from 5.0 K down to 1.0 K
    t_vals = np.linspace(5.0, 1.0, 500)

    # Plot colored temperature bands
    ax.axvspan(T_BOIL_1ATM, 5.0, color=ACCENT_AMBER, alpha=0.15, label="Helium Gas (Vapor)")
    ax.axvspan(T_LAMBDA, T_BOIL_1ATM, color=ACCENT_BLUE, alpha=0.15, label="Helium I (Normal Viscous Liquid)")
    ax.axvspan(1.0, T_LAMBDA, color=ACCENT_CYAN, alpha=0.20, label="Helium II (Superfluid)")

    # Transition boundary vertical lines
    ax.axvline(T_BOIL_1ATM, color=ACCENT_AMBER, linestyle="--", linewidth=2.0)
    ax.axvline(T_LAMBDA, color=ACCENT_CYAN, linestyle="--", linewidth=2.5)

    # Illustrative effective heat conductivity / calmness metric
    # Helium II has an apparent thermal conductivity orders of magnitude higher than copper!
    delta_t = np.maximum(0.0, T_LAMBDA - t_vals + 0.1)
    cond = np.where(
        t_vals > T_BOIL_1ATM,
        0.05,
        np.where(t_vals > T_LAMBDA, 0.2, 5.0 * (delta_t ** 0.3) + 2.0),
    )
    ax.plot(t_vals, cond, color=ACCENT_GREEN, linewidth=2.5, label="Effective Heat Transport Index (arb. units)")

    # Key historical and facility markers
    ax.scatter([1.7], [6.5], color=ACCENT_PURPLE, s=120, zorder=5, label="Onnes 1908 Achieved (1.7 K)")
    ax.scatter([1.9], [5.8], color=ACCENT_RED, s=120, zorder=5, label="CERN LHC Magnet Cooling (1.9 K)")
    ax.scatter([4.222], [0.2], color=ACCENT_AMBER, s=100, zorder=5)
    ax.scatter([2.1768], [0.2], color=ACCENT_CYAN, s=120, marker="D", zorder=5)

    ax.annotate(
        "Boiling Point (4.22 K)\nVigorous Bubbling",
        xy=(4.222, 1.0),
        xytext=(4.35, 3.5),
        arrowprops=dict(facecolor=ACCENT_AMBER, shrink=0.08, width=1.5, headwidth=7),
        color=ACCENT_AMBER,
        fontsize=10,
        fontweight="bold",
    )

    ax.annotate(
        "Lambda Point (2.1768 K)\nBubbling Stops Instantly;\nHeat Transport Diverges",
        xy=(2.1768, 2.5),
        xytext=(2.35, 6.2),
        arrowprops=dict(facecolor=ACCENT_CYAN, shrink=0.08, width=1.5, headwidth=7),
        color=ACCENT_CYAN,
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlim(5.0, 1.0)  # Inverted to show cooling from right to left
    ax.set_ylim(-0.5, 9.0)
    ax.set_xlabel("Temperature (Kelvin)  [Cooling Direction \u2190]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Transport / State Signature", fontsize=12, fontweight="bold")
    ax.set_title("Helium-4 Cooling Path: Gas \u2192 Helium I \u2192 Helium II (Superfluid)", fontsize=13, fontweight="bold", pad=12)

    legend = ax.legend(loc="upper left", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def plot_02_condensate_vs_superfluid_fraction(out_path: Path):
    """Plot 2: Superfluid density fraction vs microscopic BEC condensate fraction."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    apply_theme(fig, ax)

    temps = np.linspace(0.01, 2.3, 400)
    rho_s = []
    n_0 = []

    for t in temps:
        fracs = calculate_fractions(t)
        rho_s.append(fracs.superfluid_fraction * 100.0)
        n_0.append(fracs.condensate_fraction * 100.0)

    rho_s = np.array(rho_s)
    n_0 = np.array(n_0)

    # Superfluid fraction curve
    ax.plot(temps, rho_s, color=ACCENT_CYAN, linewidth=3.2, label=r"Superfluid Density Fraction $\rho_s / \rho$ (%)")
    ax.fill_between(temps, 0, rho_s, color=ACCENT_CYAN, alpha=0.15)

    # BEC condensate fraction curve
    ax.plot(temps, n_0, color=ACCENT_AMBER, linewidth=3.0, linestyle="--", label=r"Bose Condensate Fraction $n_0$ (%) [NIST / Prisk 2017]")
    ax.fill_between(temps, 0, n_0, color=ACCENT_AMBER, alpha=0.25)

    # Lambda line
    ax.axvline(T_LAMBDA, color=BORDER_COLOR, linestyle=":", linewidth=2.0)
    ax.text(T_LAMBDA + 0.02, 50, r"$T_\lambda = 2.1768\text{ K}$", color=MUTED_TEXT, rotation=90, fontsize=10)

    # Explanatory callouts
    ax.annotate(
        r"Macroscopic Coherence: $\rho_s / \rho \to 100\%$ at $T \to 0$",
        xy=(0.2, 99.5),
        xytext=(0.4, 85.0),
        arrowprops=dict(facecolor=ACCENT_CYAN, shrink=0.08, width=1.5, headwidth=7),
        color=ACCENT_CYAN,
        fontsize=10,
        fontweight="bold",
    )

    ax.annotate(
        "Strong Atomic Repulsion Depletes\nSingle-Particle BEC to ~7.5%",
        xy=(0.5, 7.4),
        xytext=(0.7, 25.0),
        arrowprops=dict(facecolor=ACCENT_AMBER, shrink=0.08, width=1.5, headwidth=7),
        color=ACCENT_AMBER,
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlim(0.0, 2.3)
    ax.set_ylim(-2.0, 110.0)
    ax.set_xlabel("Temperature (Kelvin)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fractional Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Superfluid Fraction vs. Bose-Einstein Condensate Fraction in Helium-4", fontsize=13, fontweight="bold", pad=12)

    legend = ax.legend(loc="center right", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, fontsize=10)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def plot_03_phonon_roton_dispersion(out_path: Path):
    """Plot 3: Phonon and roton branches with Landau tangent."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    apply_theme(fig, ax)

    # Wavevector k = p / hbar in Angstrom^-1
    k_phonon = np.linspace(0.0, 0.65, 100)
    p_phonon = (k_phonon * 1e10) * H_BAR
    e_phonon_k = phonon_energy(p_phonon) / K_B

    k_roton = np.linspace(1.4, 2.4, 150)
    p_roton = (k_roton * 1e10) * H_BAR
    e_roton_k = roton_energy(p_roton) / K_B

    # Intermediate uncommitted region (schematic dotted connector)
    k_inter = np.linspace(0.65, 1.4, 50)
    e_inter = np.linspace(e_phonon_k[-1], e_roton_k[0], 50) + 3.0 * np.sin(np.pi * (k_inter - 0.65) / 0.75)

    ax.plot(k_phonon, e_phonon_k, color=ACCENT_BLUE, linewidth=3.2, label=r"Phonon Regime: $E(p) = c_s p$ ($c_s \approx 238\text{ m/s}$)")
    ax.plot(k_roton, e_roton_k, color=ACCENT_GREEN, linewidth=3.2, label=r"Roton Minimum: $\Delta + \frac{(p - p_0)^2}{2\mu}$ ($\Delta/k_B = 8.65\text{ K}$)")
    ax.plot(k_inter, e_inter, color=MUTED_TEXT, linestyle=":", linewidth=1.8, label="Intermediate Region (Unspecified Approximation)")

    # Landau critical velocity tangent line: E = v_c * p
    v_c, p_star = landau_critical_velocity()
    k_tangent = np.linspace(0.0, 2.3, 100)
    p_tangent = (k_tangent * 1e10) * H_BAR
    e_tangent_k = (v_c * p_tangent) / K_B
    ax.plot(k_tangent, e_tangent_k, color=ACCENT_RED, linestyle="--", linewidth=2.2, label=rf"Landau Critical Slope $v_c = \min [E(p)/p] \approx {v_c:.1f}\text{{ m/s}}$")

    # Mark roton minimum
    k0 = P0_OVER_HBAR / 1e10
    ax.scatter([k0], [DELTA_KB], color=ACCENT_AMBER, s=130, zorder=5, marker="o")
    ax.annotate(
        rf"Roton Minimum: $k_0 \approx {k0:.2f}\text{{~\AA}}^{{-1}}$, $\Delta/k_B \approx {DELTA_KB:.2f}\text{{ K}}$",
        xy=(k0, DELTA_KB),
        xytext=(k0 - 0.7, DELTA_KB + 3.5),
        arrowprops=dict(facecolor=ACCENT_AMBER, shrink=0.08, width=1.5, headwidth=7),
        color=ACCENT_AMBER,
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(0.0, 20.0)
    ax.set_xlabel(r"Momentum Wavevector $k = p / \hbar$ ($\text{\AA}^{-1}$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Excitation Energy $E(p) / k_B$ (Kelvin)", fontsize=12, fontweight="bold")
    ax.set_title("Helium-4 Elementary Excitation Spectrum: Phonons & Rotons", fontsize=13, fontweight="bold", pad=12)

    legend = ax.legend(loc="upper left", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def plot_04_landau_critical_velocity(out_path: Path):
    """Plot 4: E(p)/p minimization and Doppler-shifted spectrum in moving frame."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    apply_theme(fig, [ax1, ax2])

    v_c, p_star = landau_critical_velocity()

    # Panel 1: E(p) / p ratio vs k around the roton minimum
    k_vals = np.linspace(1.2, 2.4, 200)
    p_vals = (k_vals * 1e10) * H_BAR
    e_roton = roton_energy(p_vals)
    ratio_v = e_roton / p_vals

    ax1.plot(k_vals, ratio_v, color=ACCENT_BLUE, linewidth=3.0, label=r"$E(p) / p$ (Phase Velocity)")
    ax1.axhline(v_c, color=ACCENT_RED, linestyle="--", linewidth=2.0, label=rf"Minimum $v_c = {v_c:.2f}\text{{ m/s}}$")
    k_min = (p_star / H_BAR) / 1e10
    ax1.scatter([k_min], [v_c], color=ACCENT_AMBER, s=120, zorder=5)
    ax1.annotate(
        rf"Critical Minimum: $v_c \approx {v_c:.1f}\text{{ m/s}}$",
        xy=(k_min, v_c),
        xytext=(k_min - 0.4, v_c + 15),
        arrowprops=dict(facecolor=ACCENT_AMBER, shrink=0.08, width=1.5, headwidth=7),
        color=ACCENT_AMBER,
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_xlabel(r"Wavevector $k = p / \hbar$ ($\text{\AA}^{-1}$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Excitation Velocity $E(p)/p$ (m/s)", fontsize=11, fontweight="bold")
    ax1.set_title(r"Minimizing $E(p)/p$ across Roton Branch", fontsize=12, fontweight="bold", pad=10)
    leg1 = ax1.legend(loc="upper right", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, fontsize=9.5)
    for text in leg1.get_texts():
        text.set_color(TEXT_COLOR)

    # Panel 2: Doppler-shifted excitation spectrum E'(p) = E(p) - p * v
    velocities = [0.0, 0.5 * v_c, v_c, 1.35 * v_c]
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED]
    labels = [
        r"$v = 0$ (Static Superfluid)",
        r"$v = 0.50 v_c$ ($29.2\text{ m/s}$, Subcritical)",
        rf"$v = v_c$ (${v_c:.1f}\text{{ m/s}}$, Critical Limit)",
        r"$v = 1.35 v_c$ ($78.8\text{ m/s}$, Supercritical)",
    ]

    for v, c, lbl in zip(velocities, colors, labels):
        e_doppler_k = doppler_energy(p_vals, velocity=v) / K_B
        ax2.plot(k_vals, e_doppler_k, color=c, linewidth=2.5, label=lbl)

    # Zero energy threshold
    ax2.axhline(0.0, color=BORDER_COLOR, linestyle=":", linewidth=2.0)
    ax2.fill_between(k_vals, -10.0, 0.0, color=ACCENT_RED, alpha=0.10)
    ax2.text(1.3, -3.0, "Dissipation Region: E'(p) < 0\n(Spontaneous Quasiparticle Emission)", color=ACCENT_RED, fontsize=9, fontweight="bold")

    ax2.set_ylim(-5.0, 16.0)
    ax2.set_xlabel(r"Wavevector $k = p / \hbar$ ($\text{\AA}^{-1}$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Moving-Frame Energy $E'(p) / k_B$ (Kelvin)", fontsize=11, fontweight="bold")
    ax2.set_title(r"Doppler Shift $E'(p) = E(p) - p \cdot v$", fontsize=12, fontweight="bold", pad=10)
    leg2 = ax2.legend(loc="upper right", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, fontsize=9.5)
    for text in leg2.get_texts():
        text.set_color(TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def generate_all_figures(output_dir: Path):
    """Generate all 4 high-resolution figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_01_phase_transitions(output_dir / "01_phase_transitions_cooling.png")
    plot_02_condensate_vs_superfluid_fraction(output_dir / "02_condensate_vs_superfluid_fraction.png")
    plot_03_phonon_roton_dispersion(output_dir / "03_phonon_roton_dispersion.png")
    plot_04_landau_critical_velocity(output_dir / "04_landau_critical_velocity.png")


if __name__ == "__main__":
    fig_dir = Path("figures")
    generate_all_figures(fig_dir)
    print(f"Generated 4 publication figures in {fig_dir.resolve()}")
