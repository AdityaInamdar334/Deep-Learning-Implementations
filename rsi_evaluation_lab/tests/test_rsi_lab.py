import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi_lab import DOMAINS, trajectory, apparent_score, first_crossing, judge_example, run

class TestLab(unittest.TestCase):
 def test_all_dimensions(self):
  self.assertEqual(set(trajectory(1)[0]),set(DOMAINS))
 def test_covered_score(self):
  scores={'pretraining':.9,'posttraining':.9,'harness':.9,'judgment':.1,'systems':.1}
  self.assertGreater(apparent_score(scores), apparent_score(scores,DOMAINS))
 def test_monotonic(self):
  points=trajectory(30)
  self.assertTrue(all(points[i]['judgment']<=points[i+1]['judgment'] for i in range(30)))
 def test_no_extrapolation_from_short_horizon(self):
  self.assertIsNone(first_crossing(trajectory(5),.8))
 def test_judge_bias(self):
  d=judge_example(.04,.07)
  self.assertAlmostEqual(d['estimated_improvement'],.11)
 def test_provenance(self):
  self.assertIn('SYNTHETIC',run()['disclaimer'])

if __name__=='__main__': unittest.main()
