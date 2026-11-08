"""Automated unit test suite for the Superfluid Helium Physics Lab.

Tests all 12 core physical principles, phase classifications, and theoretical models:
1. Boiling point constant (4.22 K).
2. Lambda transition constant (2.1768 K).
3. Phase classification boundaries (Gas -> Helium I -> Helium II).
4. Boundary validation (reject negative Kelvin).
5. Condensate fraction depletion limit (~7.5% per Prisk et al. 2017).
6. Superfluid fraction asymptote (-> 100% as T -> 0 K, 0 at T_lambda).
7. Acoustic phonon linear dispersion (c_s ~ 238 m/s).
8. Roton minimum parameters (Delta/k_B = 8.65 K, k_0 = 1.91 A^-1, mu = 0.16 m_He).
9. Landau critical velocity calculation (v_c ~ 58.4 m/s).
10. Doppler-shifted moving frame stability (E' > 0 for v < v_c, negative for v > v_c).
11. Second sound velocity in Helium II (c_2 ~ 20.46 m/s at 1.7 K, 0 at T_lambda).
12. Quantum zero-point solidification threshold (P >= 25.3 bar).
"""

import math
import unittest

from superfluid_lab.core import (
    K_B,
    H_BAR,
    M_HE4,
    T_BOIL_1ATM,
    T_LAMBDA,
    P_SOLIDIFICATION_MIN_BAR,
    HeliumPhase,
    classify_phase,
    calculate_fractions,
    requires_pressure_to_freeze,
)
from superfluid_lab.landau import (
    C_S,
    DELTA_KB,
    P0_OVER_HBAR,
    P0,
    MU_EFFECTIVE,
    phonon_energy,
    roton_energy,
    landau_critical_velocity,
    doppler_energy,
    second_sound_velocity,
)


class TestSuperfluidHeliumPhysics(unittest.TestCase):
    """Test suite covering the 12 verified physical properties of liquid Helium-4."""

    def test_01_boiling_point_constant(self):
        """Test 1: Verify Helium-4 normal boiling point is ~4.22 K at 1 atmosphere."""
        self.assertAlmostEqual(T_BOIL_1ATM, 4.222, places=2)
        self.assertEqual(classify_phase(4.5), HeliumPhase.GAS)
        self.assertEqual(classify_phase(4.0), HeliumPhase.HELIUM_I)

    def test_02_lambda_point_constant(self):
        """Test 2: Verify the lambda phase transition temperature is 2.1768 K."""
        self.assertAlmostEqual(T_LAMBDA, 2.1768, places=4)
        self.assertEqual(classify_phase(2.1768), HeliumPhase.HELIUM_II)
        self.assertEqual(classify_phase(2.18), HeliumPhase.HELIUM_I)

    def test_03_phase_classification(self):
        """Test 3: Verify correct classification across the three thermodynamic regimes."""
        self.assertEqual(classify_phase(10.0), HeliumPhase.GAS)
        self.assertEqual(classify_phase(3.0), HeliumPhase.HELIUM_I)
        self.assertEqual(classify_phase(1.9), HeliumPhase.HELIUM_II)
        self.assertEqual(classify_phase(0.5), HeliumPhase.HELIUM_II)

    def test_04_negative_temperature_rejected(self):
        """Test 4: Verify that unphysical negative absolute temperatures raise ValueError."""
        with self.assertRaises(ValueError):
            classify_phase(-1.0)

    def test_05_condensate_fraction_depletion(self):
        """Test 5: Verify BEC condensate fraction is depleted to ~7.5% at low T and 0 at T_lambda."""
        frac_low_t = calculate_fractions(0.1)
        self.assertLessEqual(frac_low_t.condensate_fraction, 0.075)
        self.assertGreater(frac_low_t.condensate_fraction, 0.070)

        frac_lambda = calculate_fractions(T_LAMBDA)
        self.assertAlmostEqual(frac_lambda.condensate_fraction, 0.0, places=4)

        frac_above = calculate_fractions(3.0)
        self.assertEqual(frac_above.condensate_fraction, 0.0)

    def test_06_superfluid_density_fraction_asymptote(self):
        """Test 6: Verify macroscopic superfluid fraction rho_s/rho -> 100% at T -> 0 K."""
        frac_zero = calculate_fractions(0.01)
        self.assertAlmostEqual(frac_zero.superfluid_fraction, 1.0, places=3)
        self.assertAlmostEqual(frac_zero.normal_fraction, 0.0, places=3)

        frac_lambda = calculate_fractions(T_LAMBDA)
        self.assertAlmostEqual(frac_lambda.superfluid_fraction, 0.0, places=3)
        self.assertAlmostEqual(frac_lambda.normal_fraction, 1.0, places=3)

    def test_07_phonon_linear_dispersion(self):
        """Test 7: Verify linear acoustic phonon dispersion E(p) = c_s * p with sound speed ~238 m/s."""
        self.assertAlmostEqual(C_S, 238.3, places=1)
        p_test = 1e-25  # kg*m/s
        e_test = phonon_energy(p_test)
        self.assertAlmostEqual(e_test, C_S * p_test, places=20)

    def test_08_roton_minimum_parameters(self):
        """Test 8: Verify literature-based roton minimum parameters."""
        self.assertAlmostEqual(DELTA_KB, 8.65, places=2)
        k0_angstrom = P0_OVER_HBAR / 1e10
        self.assertAlmostEqual(k0_angstrom, 1.91, places=2)
        self.assertAlmostEqual(MU_EFFECTIVE / M_HE4, 0.16, places=2)

        # At p = p_0, energy must equal exactly Delta
        e_at_p0 = roton_energy(P0)
        self.assertAlmostEqual(e_at_p0 / K_B, DELTA_KB, places=5)

    def test_09_landau_critical_velocity(self):
        """Test 9: Verify Landau critical velocity v_c = min [E(p)/p] ~ 58.4 m/s."""
        v_c, p_star = landau_critical_velocity()
        self.assertAlmostEqual(v_c, 58.39, delta=0.5)
        # Momentum at minimum should be near p_0 (~2.01e-24 to 2.08e-24 kg*m/s)
        self.assertGreater(p_star, P0 * 0.99)
        self.assertLess(p_star, P0 * 1.05)

    def test_10_doppler_shifted_quasiparticle_energy(self):
        """Test 10: Verify moving frame excitation energy is positive for v < v_c and negative for v > v_c."""
        v_c, p_star = landau_critical_velocity()

        # Below critical velocity (subcritical flow): all excitations have positive energy
        e_subcritical = doppler_energy(p_star, velocity=0.5 * v_c)
        self.assertGreater(e_subcritical, 0.0)

        # Above critical velocity (supercritical flow): roton excitation energy turns negative
        e_supercritical = doppler_energy(p_star, velocity=1.3 * v_c)
        self.assertLess(e_supercritical, 0.0)

    def test_11_second_sound_velocity(self):
        """Test 11: Verify second sound velocity peak is ~20.46 m/s at 1.7 K and 0 at T_lambda."""
        c2_17 = second_sound_velocity(1.70)
        self.assertAlmostEqual(c2_17, 20.46, delta=0.1)

        c2_lambda = second_sound_velocity(T_LAMBDA)
        self.assertEqual(c2_lambda, 0.0)

        c2_above = second_sound_velocity(3.0)
        self.assertEqual(c2_above, 0.0)

    def test_12_solidification_pressure_threshold(self):
        """Test 12: Verify Helium-4 requires >= 25.3 bar to solidify due to quantum zero-point motion."""
        self.assertFalse(requires_pressure_to_freeze(1.0))
        self.assertFalse(requires_pressure_to_freeze(15.0))
        self.assertFalse(requires_pressure_to_freeze(25.0))
        self.assertTrue(requires_pressure_to_freeze(25.3))
        self.assertTrue(requires_pressure_to_freeze(30.0))


if __name__ == "__main__":
    unittest.main()
