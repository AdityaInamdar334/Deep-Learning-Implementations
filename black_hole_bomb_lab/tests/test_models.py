import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from black_hole_bomb_lab.models import (GAMMA_PER_S,PLATEAU_M,A0_M,linear_amplitude,
 saturated_amplitude,mode_amplitudes,coupled_energy,circulation_quanta,bicoherence_experiment)

class ToyModelTests(unittest.TestCase):
    def test_initial_value(self):self.assertAlmostEqual(float(saturated_amplitude(0)),A0_M)
    def test_plateau(self):self.assertAlmostEqual(float(saturated_amplitude(400)),PLATEAU_M,places=10)
    def test_linear_exponential(self):
        self.assertAlmostEqual(float(linear_amplitude(10)/linear_amplitude(0)),np.exp(GAMMA_PER_S*10))
    def test_saturation_below_linear(self):
        self.assertLess(float(saturated_amplitude(80)),float(linear_amplitude(80)))
    def test_around_80_seconds(self):
        self.assertGreater(float(saturated_amplitude(80)/PLATEAU_M),0.95)
        self.assertLessEqual(float(saturated_amplitude(80)/PLATEAU_M),1)
    def test_growth_rate_early(self):
        t=np.linspace(0,12,200)
        slope=np.polyfit(t,np.log(saturated_amplitude(t)),1)[0]
        self.assertAlmostEqual(slope,GAMMA_PER_S,places=3)
    def test_harmonic_twice_rate_early(self):
        t=np.linspace(0,12,200);a,h,dc=mode_amplitudes(t)
        self.assertAlmostEqual(np.polyfit(t,np.log(h),1)[0],2*np.polyfit(t,np.log(a),1)[0],places=9)
    def test_static_mode_quadratic(self):
        a,h,dc=mode_amplitudes(20)
        self.assertAlmostEqual(float(dc/(a*a)),.07/PLATEAU_M)
    def test_energy_nonnegative(self):
        e=coupled_energy(np.linspace(0,110,1101))
        self.assertTrue(np.all(e['primary']>=0));self.assertTrue(np.all(e['secondary']>=0))
    def test_energy_model_steady(self):
        e=coupled_energy(np.linspace(0,400,4001))
        self.assertAlmostEqual(float(e['injection'][-1]/e['transfer'][-1]),1,places=6)
        self.assertAlmostEqual(float(e['dissipation'][-1]/e['transfer'][-1]),1,places=3)
    def test_circulation_quanta(self):self.assertAlmostEqual(circulation_quanta(),121000,delta=2000)
    def test_bicoherence_locked(self):self.assertGreater(bicoherence_experiment()['coherent'],.99)
    def test_bicoherence_random_control(self):self.assertLess(bicoherence_experiment()['random_phase'],.3)
    def test_bad_input(self):
        with self.assertRaises(ValueError):saturated_amplitude(1,initial=PLATEAU_M)
    def test_bad_times(self):
        with self.assertRaises(ValueError):coupled_energy(np.array([0,2,1]))
    def test_consistent_seed(self):self.assertEqual(bicoherence_experiment(seed=12),bicoherence_experiment(seed=12))
if __name__=='__main__':unittest.main()
