import unittest,json
from collections import Counter
from pathlib import Path
from beliefs import worlds,after_survival,World,categorical_brier
from vendor.ratings import fit
ROOT=Path(__file__).resolve().parent
class BeliefTests(unittest.TestCase):
    def test_working_slayer_rules_out_demon(self):self.assertEqual(after_survival(worlds(.3,0,0)),0)
    def test_poison_keeps_demon_possible(self):self.assertAlmostEqual(after_survival(worlds(.3,.1,.1)),.03/.73)
    def test_targeted_poison_differs(self):self.assertAlmostEqual(after_survival(worlds(.3,.8,.1)),.24/.94)
    def test_impossible_evidence(self):
        with self.assertRaises(ValueError):after_survival([World(True,False,1)])
    def test_bad_probability(self):
        with self.assertRaises(ValueError):worlds(.3,2,.1)
    def test_current_demon_matters(self):
        p={'old':.9,'new':.1};self.assertGreater(categorical_brier(p,'new'),categorical_brier(p,'old'))
    def test_brier_perfect(self):self.assertEqual(categorical_brier({'a':1,'b':0},'a'),0)
    def test_normalization(self):
        with self.assertRaises(ValueError):categorical_brier({'a':.8,'b':.8},'a')
class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games=json.loads((ROOT/'data/games.json').read_text());cls.p=json.loads((ROOT/'data/published_ratings.json').read_text())
    def test_pairs(self):
        counts=Counter(g['pair'] for g in self.games);self.assertEqual(len(counts),193);self.assertEqual(set(counts.values()),{2})
        for pair in counts:
            a,b=[g for g in self.games if g['pair']==pair]
            self.assertEqual((a['good'],a['evil']),(b['evil'],b['good']))
            self.assertEqual(a['minions'],b['minions']);self.assertEqual(a['players'],b['players'])
    def test_counts(self):self.assertEqual(len(self.games),386);self.assertEqual(sum(not g['won'] for g in self.games),129)
    def test_published_point_estimates(self):
        names=[m['id'] for m in self.p['models']];c,_,_=fit(self.games,names);d,v,_=fit(self.games,names,split=True)
        for m in self.p['models']:
            self.assertEqual([round(c[m['id']]),round(d[m['id']]),round(v[m['id']])],[m['combined'],m['detection'],m['deception']])
if __name__=='__main__':unittest.main()
