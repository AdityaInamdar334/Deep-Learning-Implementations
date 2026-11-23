import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_compiler import dot_ref, scheduled_dot, render_verilog
class Tests(unittest.TestCase):
 def test_all_parallelism_factors(self):
  a=[1,-2,3,-4,5,-6,7,-8];b=[8,7,-6,-5,4,3,-2,-1]
  for lanes in (1,2,4,8):
   value,trace=scheduled_dot(a,b,lanes)
   self.assertEqual(value,dot_ref(a,b))
   self.assertEqual(len(trace),8//lanes)
 def test_generates_modules(self):
  for lanes in (1,2,4,8):
   self.assertIn('module dot8_lanes_',render_verilog(lanes))
 def test_invalid(self):
  with self.assertRaises(ValueError):scheduled_dot([1]*8,[1]*8,3)
if __name__=='__main__':unittest.main()
