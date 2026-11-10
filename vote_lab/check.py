import unittest
from math import isclose,log
from votes import majority_correct,pair_objective,select,replay_decision

class Checks(unittest.TestCase):
    def test_binary_symmetry(self):
        for n in (1,9,33,65):
            self.assertAlmostEqual(majority_correct(.6,n)+majority_correct(.4,n),1)
    def test_more_votes_can_hurt(self):
        self.assertLess(majority_correct(.4,33),majority_correct(.4,9))
        self.assertGreater(majority_correct(.6,33),majority_correct(.6,9))
    def test_differentiation(self):
        for kappa in (18/32,28/32):
            m=[-3.,1.,7.]
            _,weights,grad=pair_objective(m,kappa)
            self.assertAlmostEqual(sum(weights),1)
            for i in range(len(m)):
                plus=m.copy();minus=m.copy()
                plus[i]+=1e-5;minus[i]-=1e-5
                estimate=(pair_objective(plus,kappa)[0]-pair_objective(minus,kappa)[0])/2e-5
                self.assertAlmostEqual(estimate,grad[i],places=8)
    def test_equal_at_reference(self):
        a=pair_objective([0,0,0],.55)
        b=pair_objective([0,0,0],.9)
        self.assertTrue(isclose(a[0],log(2)))
        self.assertEqual(a[1:],b[1:])
    def test_score_selection(self):
        xs=["A","A","B"];self.assertEqual(select(xs),"A")
        self.assertEqual(select(xs,[-.3,-.3,-.1]),"B")
    def test_conflict(self):
        c={};replay_decision(c,"x",["A","A"])
        self.assertEqual(replay_decision(c,"x",["B","A","B"]),"replay cached completions")
        self.assertEqual(replay_decision(c,"y",["B","A","B"]),"use fresh pairs")
if __name__=="__main__":unittest.main(verbosity=2)
