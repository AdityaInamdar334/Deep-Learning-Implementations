import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_detours import *

class DetourTests(unittest.TestCase):
    def test_original_route(self):
        path = shortest_path(START, GOAL, set(INITIAL_WALLS))
        self.assertIn(NEW_WALL, path)
        self.assertEqual(path[0],START)
        self.assertEqual(path[-1],GOAL)
    def test_fixed_fails_when_road_closes(self):
        ok, _ = fixed_plan_agent(set(INITIAL_WALLS),set(INITIAL_WALLS)|{NEW_WALL})
        self.assertFalse(ok)
    def test_adaptive_replans(self):
        ok, path, replans = adaptive_agent(set(INITIAL_WALLS),set(INITIAL_WALLS)|{NEW_WALL})
        self.assertTrue(ok)
        self.assertGreaterEqual(replans,1)
        self.assertNotIn(NEW_WALL,path)
    def test_impossible_goal(self):
        self.assertIsNone(shortest_path(START,GOAL,{(5,1),(5,2),(5,3),(6,1),(6,3),(5,0),(5,4)}))
    def test_option_count(self):
        walls = set(INITIAL_WALLS)|{NEW_WALL}
        self.assertGreater(reachable_state_count((2,2),walls,3),0)
        self.assertLessEqual(reachable_state_count((2,2),walls,3),GRID[0]*GRID[1])

if __name__=='__main__':unittest.main()
