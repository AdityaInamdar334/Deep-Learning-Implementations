import unittest
import numpy as np
from science_lab.data import make_study
from science_lab.stats import benjamini_hochberg,inspect_feature
from science_lab.orchestrator import execute,prioritize_features

class TestLab(unittest.TestCase):
    def setUp(self):
        self.st=make_study(seed=0,null_count=6)
    def test_study_dimensions(self):
        self.assertEqual(self.st.x_train.shape[1],19)
    def test_no_row_leakage(self):
        a={tuple(row) for row in self.st.x_train}
        b={tuple(row) for row in self.st.x_holdout}
        self.assertFalse(a & b)
    def test_study_reproducible(self):
        st=make_study(seed=0,null_count=6)
        np.testing.assert_array_equal(self.st.x_train,st.x_train)
    def test_stratified(self):
        self.assertEqual(set(self.st.y_train),{0,1})
        self.assertEqual(set(self.st.y_holdout),{0,1})
    def test_nulls_independent_generated(self):
        self.assertEqual(self.st.observed_count,13)
        self.assertTrue(self.st.names[-1].startswith('synthetic_null'))
    def test_bh_known_values(self):
        np.testing.assert_allclose(benjamini_hochberg([0.01,0.04,0.03,0.2]),[0.04,0.053333333333,0.053333333333,0.2],atol=1e-9)
    def test_bh_monotone(self):
        p=np.array([0.004,0.002,0.2,0.7])
        q=benjamini_hochberg(p)
        self.assertTrue(np.all(np.diff(q[np.argsort(p)])>=-1e-12))
    def test_bh_empty(self):
        self.assertEqual(len(benjamini_hochberg([])),0)
    def test_bh_reject_invalid(self):
        with self.assertRaises(ValueError): benjamini_hochberg([1.2])
    def test_inspect_provenance(self):
        f=inspect_feature(self.st,0)
        self.assertIn('seed_0',f.provenance)
        self.assertLessEqual(f.discovery_p,1.0)
    def test_rank_discovery_only(self):
        first=prioritize_features(self.st)
        # Mutating holdout cannot change ranking.
        mutable=self.st.x_holdout.copy(); mutable *= 999
        from dataclasses import replace
        second=prioritize_features(replace(self.st,x_holdout=mutable))
        self.assertEqual(first,second)
    def test_parallel_matches_serial_claims(self):
        serial=execute(self.st,mode='distributed',workers=1,max_features=12)
        parallel=execute(self.st,mode='distributed',workers=3,max_features=12)
        for key in ('accepted_claims','independently_replicated_claims','known_null_claims','claims'):
            self.assertEqual(serial[key],parallel[key])
    def test_holdout_cannot_change_discovery_p(self):
        from dataclasses import replace
        changed=replace(self.st,y_holdout=1-self.st.y_holdout)
        a=inspect_feature(self.st,3);b=inspect_feature(changed,3)
        self.assertEqual(a.discovery_p,b.discovery_p)
    def test_verification_can_only_reject(self):
        checked=execute(self.st,mode='distributed',max_features=19,verification=True)
        unchecked=execute(self.st,mode='distributed',max_features=19,verification=False)
        self.assertLessEqual(checked['accepted_claims'],unchecked['accepted_claims'])
    def test_feature_budget(self):
        r=execute(self.st,mode='generalist',max_features=5)
        self.assertEqual(r['features_examined'],5)
    def test_invalid_workers(self):
        with self.assertRaises(ValueError): execute(self.st,workers=0)
    def test_synthetic_controls_not_medical(self):
        self.assertEqual(len(self.st.names),19)

if __name__=='__main__':unittest.main()
