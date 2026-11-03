import unittest
from dataclasses import replace
import torch
from monitor_lab.data import generate_dataset, SCENARIOS
from monitor_lab.models import LinearProbe, TextReviewer, features_text, build_vocab, features_probe, train_model, recall_threshold
from monitor_lab.pipeline import stable_audit, Config, evaluate, tool_policy_violation, telemetry_discrepancy
from monitor_lab.metrics import confusion, summarize
from monitor_lab.experiment import run_seed


class DatasetTests(unittest.TestCase):
    def test_reproducible_and_disjoint(self):
        a = generate_dataset(12, 4, 'train')
        b = generate_dataset(12, 4, 'train')
        c = generate_dataset(12, 4, 'test')
        self.assertEqual(a, b)
        self.assertTrue(set(x.case_id for x in a).isdisjoint(x.case_id for x in c))

    def test_scenario_labels(self):
        a = generate_dataset(500, 4)
        self.assertTrue(all(case.label == SCENARIOS[case.scenario][0] for case in a))
        self.assertTrue(any(case.scenario == 'hidden_egress' for case in a))

    def test_proxy_features_are_only_synthetic(self):
        a = generate_dataset(10, 4)
        self.assertEqual(len(a[0].proxy_activation), 10)
        self.assertEqual(tuple(features_probe(a).shape), (10, 10))

    def test_shift_changes_wording(self):
        first = generate_dataset(50, 4, 'test', shift=False)
        second = generate_dataset(50, 4, 'test', shift=True)
        self.assertTrue(any(x.narrative != y.narrative for x, y in zip(first, second)))


class ModelTests(unittest.TestCase):
    def test_text_features_nonnegative(self):
        cases = generate_dataset(22, 1)
        vocab = build_vocab(cases)
        matrix = features_text(cases, vocab)
        self.assertEqual(tuple(matrix.shape), (22, len(vocab)))
        self.assertTrue(((matrix == 0) | (matrix == 1)).all().item())

    def test_models_can_train(self):
        data = generate_dataset(60, 3)
        x = features_probe(data)
        m = train_model(LinearProbe(10), x, [r.label for r in data], seed=3, epochs=5)
        self.assertEqual(tuple(m(x).shape), (60,))
        vocab = build_vocab(data)
        text_model = train_model(TextReviewer(len(vocab)), features_text(data, vocab),
                                 [r.label for r in data], seed=3, epochs=5)
        self.assertEqual(tuple(text_model(features_text(data, vocab)).shape), (60,))

    def test_high_recall_threshold(self):
        scores = [.1, .8, .7, .6, .05]
        labels = [0, 1, 1, 1, 0]
        threshold = recall_threshold(scores, labels, 2/3)
        self.assertGreater(threshold, .6)
        self.assertLess(threshold, .7)
        self.assertGreaterEqual(sum(s >= threshold for s,l in zip(scores, labels) if l), 2)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = generate_dataset(200, 4, 'test')

    def test_no_silent_score_truncation(self):
        self.assertRaises(ValueError, evaluate, self.data[:2], [1], [1, 1], Config())

    def test_stale_approval_false_alarm_when_everything_investigated(self):
        cases = generate_dataset(2000, 3, 'test')
        stale = [r for r in cases if r.approval_record_missing]
        self.assertGreater(len(stale), 0)
        flags = evaluate(stale, [0] * len(stale), [0] * len(stale),
                         Config(use_probe=False, l2_threshold=0))
        self.assertTrue(all(r['security_queue'] and not r['label'] for r in flags))

    def test_stable_audit(self):
        self.assertEqual(stable_audit('abc', 4, .12), stable_audit('abc', 4, .12))
        self.assertTrue(stable_audit('abc', 4, 1))
        self.assertFalse(stable_audit('abc', 4, 0))
        self.assertRaises(ValueError, stable_audit, 'abc', 4, -1)

    def test_permission_check_contextual(self):
        unauthorized = next(r for r in self.data if r.scenario == 'unauthorized_egress')
        authorized = next(r for r in self.data if r.scenario == 'approved_network')
        self.assertTrue(tool_policy_violation(unauthorized))
        self.assertFalse(tool_policy_violation(authorized))

    def test_telemetry_exposes_hidden_network(self):
        case = next(r for r in self.data if r.scenario == 'hidden_egress')
        self.assertFalse(tool_policy_violation(case))
        self.assertTrue(telemetry_discrepancy(case))
        cfg = Config(use_probe=False, l2_threshold=0)
        with_telemetry = evaluate([case], [0], [1], cfg)[0]
        without = evaluate([case], [0], [1], Config(use_probe=False, l2_threshold=0,
                                                     use_telemetry=False))[0]
        self.assertTrue(with_telemetry['urgent_alert'])
        self.assertFalse(without['urgent_alert'])

    def test_no_ground_truth_leak(self):
        case = next(r for r in self.data if r.scenario == 'unauthorized_egress')
        original = evaluate([case], [1], [1], Config(l2_threshold=.5))[0]
        flipped = evaluate([replace(case, label=0, urgent=0)], [1], [1], Config(l2_threshold=.5))[0]
        self.assertEqual(original['urgent_alert'], flipped['urgent_alert'])
        self.assertEqual(original['security_queue'], flipped['security_queue'])

    def test_compaction_hole(self):
        case = replace(self.data[0], compaction=True)
        normal = evaluate([case], [1], [1], Config(l2_threshold=.5))[0]
        hole = evaluate([case], [1], [1], Config(l2_threshold=.5, skip_compaction=True))[0]
        self.assertTrue(normal['screened'])
        self.assertFalse(hole['screened'])

    def test_no_probe_all_reach_l1(self):
        scores = [0]*len(self.data)
        rows = evaluate(self.data, scores, scores, Config(use_probe=False, audit_rate=0))
        self.assertEqual(sum(r['screened'] for r in rows), len(self.data))

    def test_cost_ablations(self):
        case = self.data[0]
        base = evaluate([case], [1], [1], Config(l2_threshold=.5))[0]
        cheap = evaluate([case], [1], [1], Config(l2_threshold=.5, use_telemetry=False))[0]
        self.assertGreater(base['cost_units'], cheap['cost_units'])

    def test_policy_not_sufficient_for_hidden_case(self):
        hidden = next(r for r in self.data if r.scenario == 'hidden_egress')
        self.assertFalse(tool_policy_violation(hidden))

    def test_metrics(self):
        rows = [dict(label=1, urgent_truth=1, screened=True, security_queue=True,
                     urgent_alert=True, investigated=True, l3=True, covered=True, cost_units=10),
                dict(label=0, urgent_truth=0, screened=False, security_queue=False,
                     urgent_alert=False, investigated=False, l3=False, covered=True, cost_units=2)]
        self.assertEqual(confusion(rows)['tp'], 1)
        self.assertEqual(summarize(rows)['l2_recall'], 1)
        self.assertEqual(summarize(rows)['mean_cost_units'], 6)

    def test_small_benchmark_reproducible(self):
        a = run_seed(seed=1, train_n=100, val_n=80, test_n=70, epochs=12)
        b = run_seed(seed=1, train_n=100, val_n=80, test_n=70, epochs=12)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
