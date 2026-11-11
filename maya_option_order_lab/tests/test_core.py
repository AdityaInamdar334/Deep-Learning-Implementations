import math,unittest
from option_audit.core import (cyclic_rotations,sampled_orderings,softmax,predict,total_variation,
                               audit_records,make_toy_questions,synthetic_scores,toy_audit,identical_options_slot_prior)
from option_audit.train import make_rows,train,eval_model

class TestAudit(unittest.TestCase):
    def test_rotations_count(self):
        self.assertEqual(len(cyclic_rotations(['a','b','c'])),3)
    def test_rotations_cover_slots(self):
        order=cyclic_rotations(['a','b','c','d','e'])
        self.assertEqual({row.index('a') for row in order},set(range(5)))
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):cyclic_rotations(['x','x'])
    def test_sampled_unique(self):
        p=sampled_orderings(list('abcd'),6)
        self.assertEqual(len({tuple(x) for x in p}),6)
    def test_sampling_all_possible(self):
        self.assertEqual(len(sampled_orderings(['a','b'],7)),2)
    def test_softmax_valid(self):
        self.assertAlmostEqual(sum(softmax([1000,2000,3000])),1)
    def test_predict_stable_tie(self):
        self.assertEqual(predict({'a':1,'b':1},['b','a']),'a')
    def test_tv_uniform(self):
        self.assertAlmostEqual(total_variation([.25]*4),0)
    def test_tv_invalid(self):
        with self.assertRaises(ValueError):total_variation([.3,.3])
    def test_item_flip(self):
        rows=[{'id':'x','gold':'a','presentations':[{'order':['a','b'],'scores':{'a':2,'b':1}},
                            {'order':['b','a'],'scores':{'a':0,'b':1}}]}]
        self.assertEqual(audit_records(rows,bootstrap=50)['flip_rate'],1)
    def test_item_nonflip(self):
        rows=[{'id':'x','gold':'a','presentations':[{'order':['a','b'],'scores':{'a':2,'b':1}},
                            {'order':['b','a'],'scores':{'a':2,'b':1}}]}]
        self.assertEqual(audit_records(rows,bootstrap=50)['flip_rate'],0)
    def test_mismatched_permutations_rejected(self):
        rows=[{'id':'x','gold':'a','presentations':[{'order':['a','b'],'scores':{'a':2,'b':1}},
                            {'order':['a','c'],'scores':{'a':2,'c':1}}]}]
        with self.assertRaises(ValueError):audit_records(rows)
    def test_parallel_order_invariance(self):
        q=make_toy_questions(18)
        result=toy_audit(q,0)
        self.assertEqual(result['flip_rate'],0)
    def test_bias_reveals_flips(self):
        q=make_toy_questions(90)
        self.assertGreater(toy_audit(q,.85)['flip_rate'],0.05)
    def test_identical_options_parallel_uniform(self):
        p=identical_options_slot_prior(12,0)
        self.assertAlmostEqual(p['tv_distance'],0)
    def test_identical_options_biased(self):
        p=identical_options_slot_prior(12,.85)
        self.assertGreater(p['tv_distance'],0.1)
    def test_train_same_rows_differ(self):
        rows=make_rows(200)
        fixed=train(rows,False,epochs=5)
        shuffled=train(rows,True,epochs=5)
        self.assertNotEqual(fixed['slot_bias'],shuffled['slot_bias'])
    def test_training_shuffling_improves_flip_rate(self):
        rows=make_rows(500)
        fixed=train(rows,False)
        shuffled=train(rows,True)
        self.assertLessEqual(eval_model(shuffled)['flip_rate'],eval_model(fixed)['flip_rate'])

if __name__=='__main__':unittest.main()
