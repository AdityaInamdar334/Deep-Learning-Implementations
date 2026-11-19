import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from toy_world import ACTIONS, make_data, fit_transition, predict, evaluate, plan
class TestWorld(unittest.TestCase):
    def test_deterministic(self):
        x,a,y,f,t=make_data(8,11)
        xx,aa,yy,ff,tt=make_data(8,11)
        self.assertTrue(np.array_equal(f,ff));self.assertTrue(np.array_equal(y,yy))
    def test_action_rule(self):
        x,a,y,_,_=make_data(20)
        self.assertTrue(np.array_equal(y-x,ACTIONS[a]))
    def test_learned_transition(self):
        x,a,y,_,_=make_data(1000)
        w=fit_transition(x,a,y)
        self.assertLess(np.max(abs(predict(x,a,w)-y)),1e-8)
    def test_unpredictable_pixels(self):
        r=evaluate(100,20)
        self.assertGreater(r['pixel_persistence_mse'],r['structured_pixel_prediction_mse'])
    def test_plan(self):
        x,a,y,_,_=make_data(1000)
        p=plan([4,4],[9,7],fit_transition(x,a,y),10)
        self.assertLess(p[-1]['distance'],0.5)
if __name__=='__main__':unittest.main()
