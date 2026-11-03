import unittest
import numpy as np
from spikeguard_lab.data import make_synthetic_recording, Recording
from spikeguard_lab.methods import sttc, bh_adjust, population_coupling, pca_variance, bin_spikes
from spikeguard_lab.agent import ToolSession, ClarificationRequired, run_policy
from spikeguard_lab.evaluator import evaluate


class TestRecording(unittest.TestCase):
    def setUp(self): self.r=make_synthetic_recording(2)
    def test_determinism(self): self.assertEqual(self.r.digest(),make_synthetic_recording(2).digest())
    def test_seed_affects_data(self): self.assertNotEqual(self.r.digest(),make_synthetic_recording(1).digest())
    def test_data_unit(self):self.assertEqual(self.r.metadata['time_unit'],'ms')
    def test_window(self):
        q=self.r.window(1000,2000)
        self.assertEqual(q.duration_ms,1000)
        self.assertTrue(all(np.all((s>=0)&(s<1000)) for s in q.spikes))
    def test_invalid_unsorted(self):
        with self.assertRaises(ValueError):Recording((np.array([4.,2.]),),10,{})
    def test_hash_not_mutated(self):
        h=self.r.digest();run_policy(self.r,'bounded',surrogates=3);self.assertEqual(h,self.r.digest())

class TestStatistics(unittest.TestCase):
    def test_sttc_identical(self):
        a=np.array([100.,300.,700.,900.])
        self.assertAlmostEqual(sttc(a,a,1000,5),1,places=8)
    def test_sttc_symmetric(self):
        a=np.array([10.,20.,200.]);b=np.array([40.,150.,240.])
        self.assertAlmostEqual(sttc(a,b,300,8),sttc(b,a,300,8))
    def test_sttc_empty_nan(self):
        self.assertTrue(np.isnan(sttc(np.array([]),np.array([2.]),1000,10)))
    def test_sttc_reject_window(self):
        with self.assertRaises(ValueError):sttc(np.array([3.]),np.array([8.]),100,90)
    def test_bh(self):
        q=bh_adjust([.01,.04,.5,np.nan])
        self.assertTrue(np.allclose(q[:3],[.03,.06,.5]));self.assertTrue(np.isnan(q[3]))
    def test_coupling_excludes_self(self):
        r=make_synthetic_recording(3,n_units=8)
        counts=bin_spikes(r,100)
        self.assertAlmostEqual(population_coupling(r,100)[0],np.corrcoef(counts[0],counts[1:].sum(axis=0))[0,1])
    def test_pca_sum(self):
        self.assertLessEqual(sum(pca_variance(make_synthetic_recording(1),50,5)),1.0001)
    def test_surrogate_min_p(self):
        from spikeguard_lab.methods import circular_shift_surrogates
        a=np.array([100.,500.,900.]);v=circular_shift_surrogates(a,a,1000,10,9,2)
        self.assertGreaterEqual(v['p_two_sided'],.1)

class TestPolicies(unittest.TestCase):
    def setUp(self):self.r=make_synthetic_recording(0,n_units=8)
    def test_tool_allowlist(self):
        with self.assertRaises(PermissionError):ToolSession(self.r).call('shell',command='rm -rf /')
    def test_ambiguity_requires_clarification(self):
        with self.assertRaises(ClarificationRequired):ToolSession(self.r).call('slice_conditions',windows=None)
    def test_bounded_succeeds(self):
        self.assertEqual(evaluate(self.r,run_policy(self.r,'bounded',surrogates=3))['passed'],14)
    def test_improvised_caught(self):
        self.assertLess(evaluate(self.r,run_policy(self.r,'improvised'))['passed'],8)
    def test_no_verify_caught(self):
        result=evaluate(self.r,run_policy(self.r,'no_verification',surrogates=3))
        self.assertFalse(result['checks']['multiple_testing_q_values_present'])
        self.assertFalse(result['checks']['surrogate_p_values_present'])
    def test_independent_evaluator_detects_tamper(self):
        v=run_policy(self.r,'bounded',surrogates=3);v['analysis']['cue']['pair_results'][0]['observed']=0.123456
        self.assertFalse(evaluate(self.r,v)['checks']['sttc_matches_reference'])
    def test_downsample_ablation_isolated(self):
        out=run_policy(self.r,'silent_downsample',surrogates=3)
        ev=evaluate(self.r,out)
        self.assertEqual(ev['passed'],11)
        self.assertTrue(ev['checks']['sttc_matches_reference'])
        self.assertFalse(ev['checks']['no_undocumented_rebin'])
        self.assertFalse(ev['checks']['leave_one_out_coupling'])
        self.assertFalse(ev['checks']['pca_matches_reference'])
    def test_repeatability(self):
        a=run_policy(self.r,'bounded',seed=2,surrogates=3)
        b=run_policy(self.r,'bounded',seed=2,surrogates=3)
        self.assertEqual(a,b)

if __name__=='__main__':unittest.main()
