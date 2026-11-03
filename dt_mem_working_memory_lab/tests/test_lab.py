import unittest
import torch
from dtmem_lab.memory import SoftWorkingMemory
from dtmem_lab.toy_tasks import sample_task, draw_support, train_context_baseline, baseline_predict
from dtmem_lab.tiny_dt import TinyDecisionTransformer
from dtmem_lab.benchmark import evaluate, run
from dtmem_lab.lora_demo import LoRALinear, run as run_lora
from dtmem_lab.dt_train import train as train_tiny_dt


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_slot_shapes(self):
        m = SoftWorkingMemory(slots=4)
        self.assertEqual(tuple(m.keys.shape), (4, 8))
        self.assertEqual(tuple(m.values.shape), (4, 4))

    def test_address_weights_normalized(self):
        w = SoftWorkingMemory().addressing(torch.arange(8))
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(8)))

    def test_memory_exact_known_queries(self):
        m = SoftWorkingMemory()
        contexts = torch.arange(8)
        labels = torch.tensor([2, 1, 3, 0, 3, 2, 1, 0])
        m.write(contexts, labels)
        predicted = m.predict(contexts, torch.zeros_like(contexts))
        self.assertTrue(torch.equal(predicted, labels))

    def test_reset_clears_all(self):
        m = SoftWorkingMemory()
        m.write([2], [3])
        self.assertGreater(float(m.mass.sum()), 0)
        m.reset()
        self.assertEqual(float(m.mass.sum()), 0)

    def test_absent_query_falls_back(self):
        m = SoftWorkingMemory()
        m.write([0], [3])
        pred = m.predict([7], [2])
        self.assertEqual(int(pred[0]), 2)

    def test_task_uses_balanced_labels(self):
        g = torch.Generator().manual_seed(17)
        labels = sample_task(g)
        self.assertTrue(torch.equal(torch.bincount(labels, minlength=4), torch.tensor([2, 2, 2, 2])))

    def test_corruption_changes_all_at_rate_one(self):
        g = torch.Generator().manual_seed(17)
        mapping = sample_task(g)
        c,a=draw_support(mapping, 32, g, corruption=1.0)
        self.assertTrue(bool((a != mapping[c]).all()))

    def test_capacity_aliases_some_contexts(self):
        m = SoftWorkingMemory(slots=2)
        w = m.addressing(torch.arange(8))
        self.assertTrue(torch.allclose(w[0], w[2]))

    def test_memory_write_is_independent_of_batching(self):
        m1, m2 = SoftWorkingMemory(), SoftWorkingMemory()
        m1.write([0, 1, 2], [2, 3, 1])
        for c,a in [(0,2),(1,3),(2,1)]: m2.write([c],[a])
        self.assertTrue(torch.allclose(m1.values, m2.values))

    def test_baseline_output_valid(self):
        m = train_context_baseline(0, steps=4)
        p = baseline_predict(m, torch.arange(8))
        self.assertTrue(bool(((p>=0)&(p<4)).all()))

    def test_tiny_dt_shapes(self):
        m = TinyDecisionTransformer()
        x = torch.tensor([[1,2,3],[2,3,4]])
        self.assertEqual(tuple(m(torch.ones_like(x,dtype=torch.float),x,torch.zeros_like(x)).shape), (2,3,4))

    def test_tiny_dt_no_current_action_leakage(self):
        torch.manual_seed(2)
        m=TinyDecisionTransformer().eval()
        states=torch.tensor([[1,2,3]])
        r=torch.ones(1,3)
        a=torch.tensor([[0,0,0]])
        b=torch.tensor([[3,3,3]])
        with torch.no_grad():
            logits_a=m(r,states,a)
            logits_b=m(r,states,b)
        self.assertTrue(torch.allclose(logits_a[:,0],logits_b[:,0],atol=1e-6))
        # Context at t=1 may use action at t=0, not action at t=1.
        b2=torch.tensor([[0,3,3]])
        with torch.no_grad():
            logits_b2=m(r,states,b2)
        self.assertTrue(torch.allclose(logits_a[:,:2],logits_b2[:,:2],atol=1e-6))

    def test_lora_initially_preserves_base(self):
        torch.manual_seed(7)
        w=torch.randn(8,12)
        layer=LoRALinear(w,rank=2)
        x=torch.randn(3,12)
        self.assertTrue(torch.allclose(layer(x),x@w.T,atol=1e-6))
        self.assertFalse(layer.weight.requires_grad)

    def test_lora_can_fit_low_rank_delta(self):
        result=run_lora(steps=200)
        self.assertLess(result['final_mse'],result['initial_mse']*0.1)
        self.assertEqual(result['trainable_adapter_parameters'],40)

    def test_tiny_dt_training_smoke(self):
        metrics=train_tiny_dt(seed=2,steps=12,batch=16,horizon=3)
        self.assertTrue(0 <= metrics['heldout_sequence_accuracy'] <= 1)
        self.assertGreater(metrics['params'], 0)

    def test_benchmark_deterministic(self):
        a=evaluate(0,n_tasks=3,query_per_task=8,baseline_steps=4)
        b=evaluate(0,n_tasks=3,query_per_task=8,baseline_steps=4)
        self.assertEqual(a,b)

    def test_benchmark_summary(self):
        x=run(2,n_tasks=2,query_per_task=8,baseline_steps=4)
        self.assertEqual(x['metadata']['seeds'], 2)
        self.assertEqual(len(x['configurations']),8)


if __name__=='__main__': unittest.main()
