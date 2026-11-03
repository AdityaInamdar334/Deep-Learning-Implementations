import unittest
import torch
from weightlab.core import (make_policy,permute_hidden,rescale_hidden,compile_xor,inspect_axis_naive,inspect_axis_invariant)
from weightlab.editing import propose_edit,diff_audit,hidden_circuit
from weightlab.evaluate import classification_accuracy,navigation_success
from weightlab.benchmark import trial,run

class WeightTest(unittest.TestCase):
    def test_xor_truth_table(self):
        x=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        y=compile_xor()(x).flatten()
        self.assertTrue(torch.equal(y,torch.tensor([0.,1.,1.,0.])))
    def test_xor_has_zero_trainable_steps(self):
        self.assertEqual(sum(p.numel() for p in compile_xor().parameters()),6)
    def test_permutation_preserves_function(self):
        m=make_policy(axis=1,broken=True);g=torch.Generator().manual_seed(9)
        other=permute_hidden(m,torch.randperm(16,generator=g))
        x=torch.randn(30,4,generator=g)
        self.assertTrue(torch.allclose(m(x),other(x),atol=1e-6))
    def test_scaling_preserves_function(self):
        m=make_policy(axis=1,broken=True)
        x=torch.randn(400,4)
        self.assertTrue(torch.allclose(m(x),rescale_hidden(m,1,3)(x),atol=1e-6))
    def test_negative_scale_rejected(self):
        with self.assertRaises(ValueError):rescale_hidden(make_policy(),0,-1)
    def test_invalid_permutation_rejected(self):
        with self.assertRaises(ValueError):permute_hidden(make_policy(),torch.zeros(16,dtype=torch.long))
    def test_invalid_axis_rejected(self):
        with self.assertRaises(ValueError):make_policy(3,3)
    def test_permutation_invariant_reader(self):
        m=make_policy(axis=2,nuisance=3)
        g=torch.Generator().manual_seed(7)
        for _ in range(30): self.assertEqual(inspect_axis_invariant(permute_hidden(m,torch.randperm(16,generator=g))),2)
    def test_semantic_changes_exactly_two_weights(self):
        m=make_policy(broken=True);patch=propose_edit(m,'semantic')
        self.assertEqual(diff_audit(m,patch)['changed'],2)
        self.assertEqual(diff_audit(m,patch)['by_tensor']['fc2.weight'],2)
    def test_input_untouched(self):
        m=make_policy(broken=True);_ = propose_edit(m,'semantic')
        self.assertEqual(float(m.fc2.weight[2,2].detach()),4.)
    def test_patched_matches_good_model(self):
        x=torch.randn(300,4)
        good=make_policy(broken=False);bad=make_policy(broken=True)
        self.assertTrue(torch.allclose(propose_edit(bad,'semantic')(x),good(x)))
    def test_top_magnitude_baseline_succeeds(self):
        x=torch.randn(300,4)
        good=make_policy(broken=False);bad=make_policy(broken=True)
        self.assertTrue(torch.allclose(propose_edit(bad,'top_magnitude')(x),good(x)))
    def test_half_patch_incomplete(self):
        b=make_policy(broken=True);p=propose_edit(b,'half_semantic')
        self.assertEqual(diff_audit(b,p)['changed'],1)
        self.assertLess(classification_accuracy(p,n=3000)['accuracy'],0.9)
    def test_wrong_patch_worse(self):
        b=make_policy(broken=True)
        self.assertLess(classification_accuracy(propose_edit(b,'wrong_path'))['accuracy'],0.5)
    def test_original_identical(self):
        b=make_policy(broken=True)
        self.assertEqual(diff_audit(b,propose_edit(b,'original'))['changed'],0)
    def test_unknown_method(self):
        with self.assertRaises(ValueError):propose_edit(make_policy(),'oops')
    def test_shift_produces_more_stall(self):
        b=make_policy(broken=True)
        self.assertGreater(classification_accuracy(b,n=10000,shift=2.5)['stall_rate'],classification_accuracy(b,n=10000,shift=1)['stall_rate'])
    def test_semantic_nearly_perfect_navigation(self):
        m=propose_edit(make_policy(broken=True),'semantic')
        self.assertGreater(navigation_success(m,episodes=500)['success_rate'],0.98)
    def test_hidden_circuit_has_four_nonzero_units(self):
        self.assertEqual(len(hidden_circuit(make_policy(broken=True))),4)
    def test_invalid_structure_diff(self):
        with self.assertRaises(ValueError):diff_audit(make_policy(),compile_xor())
    def test_reproducible_trial(self):
        self.assertEqual(trial(2,readings=3,samples=100,episodes=40),trial(2,readings=3,samples=100,episodes=40))
    def test_summary_has_no_raw_model_claims(self):
        r=run(seeds=2,readings=2,samples=40,episodes=20)
        self.assertTrue(r['metadata']['not_an_llm_experiment'])
        self.assertEqual(len(r['trials']),2)
    def test_holdout_eval_does_not_mutate_weights(self):
        m=make_policy(broken=True);before={k:v.clone() for k,v in m.state_dict().items()}
        classification_accuracy(m);navigation_success(m)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in m.state_dict().items()))

if __name__=='__main__':unittest.main()

class ProjectionTest(unittest.TestCase):
    def test_projection_removed(self):
        from weightlab.projection import suppress_output_direction
        w=torch.randn(8,10)
        u=torch.randn(8)
        out=suppress_output_direction(w,u,columns=[1,4],scale=1.0)
        self.assertLess(float((u @ out[:,[1,4]]).abs().max()),1e-5)
        self.assertTrue(torch.equal(w[:,0],out[:,0]))
        self.assertTrue(torch.equal(w[:,2],out[:,2]))
        self.assertTrue(torch.equal(w[:,3],out[:,3]))
    def test_projection_does_not_mutate_original(self):
        from weightlab.projection import suppress_output_direction
        w=torch.randn(8,10);orig=w.clone()
        _=suppress_output_direction(w,torch.randn(8))
        self.assertTrue(torch.equal(w,orig))
    def test_reverse_component_at_scale_six(self):
        from weightlab.projection import suppress_output_direction
        w=torch.randn(8,10); u=torch.randn(8)
        out=suppress_output_direction(w,u,columns=[3],scale=6.0)
        before=float(u @ w[:,3]); after=float(u @ out[:,3]);
        self.assertAlmostEqual(after,-5*before,delta=1e-4)
    def test_invalid_direction_rejected(self):
        from weightlab.projection import suppress_output_direction
        with self.assertRaises(ValueError):suppress_output_direction(torch.randn(5,7),torch.zeros(5))
    def test_invalid_columns_rejected(self):
        from weightlab.projection import suppress_output_direction
        with self.assertRaises(ValueError):suppress_output_direction(torch.randn(5,7),torch.ones(5),columns=[1,1])
