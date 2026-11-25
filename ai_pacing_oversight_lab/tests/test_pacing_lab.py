import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacing_lab import Scenario, examples, simulate

class TestPacingLab(unittest.TestCase):
    def test_zero_month(self):
        self.assertEqual(len(simulate(examples()[0], 0)), 1)
    def test_all_scores_bounded(self):
        for scenario in examples():
            for row in simulate(scenario):
                self.assertTrue(all(0 <= row[key] <= 1 for key in ['capability','oversight','uncovered_gap']))
    def test_more_oversight_reduces_gap(self):
        self.assertLess(simulate(examples()[2])[-1]['uncovered_gap'], simulate(examples()[1])[-1]['uncovered_gap'])
    def test_pacing_reduces_gap(self):
        self.assertLess(simulate(examples()[1])[-1]['uncovered_gap'], simulate(examples()[0])[-1]['uncovered_gap'])
    def test_invalid_months(self):
        with self.assertRaises(ValueError): simulate(examples()[0], -1)
    def test_invalid_growth(self):
        with self.assertRaises(ValueError): simulate(Scenario('invalid', -0.1, .01))

if __name__ == '__main__': unittest.main()
