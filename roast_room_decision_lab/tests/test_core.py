import random
import unittest
from decision_lab.core import (Case, SeatCard, OPTIONS, aggregate, band, make_case,
                               make_cards, selected_seats, simulate, POLICIES)
from decision_lab.benchmark import run, single_seed

class CouncilTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case(random.Random(7))

    def test_case_shapes(self):
        self.assertEqual(len(self.case.values), 3)
        self.assertTrue(all(len(x) == 7 for x in self.case.values))
        self.assertEqual(len(self.case.weights), 7)

    def test_true_scores_bound(self):
        self.assertTrue(all(0 <= x <= 10 for x in self.case.true_scores()))

    def test_best_has_option(self):
        self.assertIn(self.case.best(), OPTIONS)

    def test_seven_weights(self):
        self.assertEqual(sorted(self.case.weights).count(3), 2)

    def test_quick_four(self):
        ids = selected_seats(self.case.weights, True)
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(set(ids)), 4)

    def test_full_seven(self):
        self.assertEqual(len(selected_seats(self.case.weights, False)), 7)

    def test_cards_four(self):
        self.assertEqual(len(make_cards(self.case, random.Random(8), quick=True)), 4)

    def test_cards_seven(self):
        self.assertEqual(len(make_cards(self.case, random.Random(8))), 7)

    def test_weighted_aggregation(self):
        cards = (SeatCard('x', (10, 0, 0), 3, (False,)*3),
                 SeatCard('y', (0, 10, 0), 1, (False,)*3))
        v = aggregate(cards, ruin_cap=False, calls=2)
        self.assertEqual(v.recommended, 'A')
        self.assertAlmostEqual(v.score, 7.5)

    def test_ruin_cap_changes_label_not_choice(self):
        cards = (SeatCard('x', (9, 0, 0), 3, (False,)*3),
                 SeatCard('y', (9, 0, 0), 1, (True, False, False)))
        a = aggregate(cards, ruin_cap=False, calls=2)
        b = aggregate(cards, ruin_cap=True, calls=2)
        self.assertEqual(a.recommended, b.recommended)
        self.assertEqual(a.label, 'GO')
        self.assertEqual(b.label, 'RESHAPE')

    def test_no_ruin_cap_on_unflagged_option(self):
        cards = (SeatCard('x', (9, 2, 2), 1, (False, True, True)),)
        self.assertEqual(aggregate(cards, ruin_cap=True, calls=1).label, 'GO')

    def test_bands(self):
        self.assertEqual([band(x) for x in (8, 6.1, 5.0, 4.0)],
                         ['GO', 'GO WITH CONDITIONS', 'RESHAPE', "DON'T DO IT"])

    def test_spread(self):
        cards = (SeatCard('a', (10, 1, 1), 1, (False,)*3),
                 SeatCard('b', (6, 1, 1), 1, (False,)*3))
        self.assertEqual(aggregate(cards, ruin_cap=False, calls=2).spread, 4)

    def test_all_policies(self):
        for i, name in enumerate(POLICIES):
            v = simulate(self.case, random.Random(10), name)
            self.assertIn(v.recommended, OPTIONS)
            self.assertGreater(v.proxy_tokens, 0)

    def test_unknown_policy_fails(self):
        with self.assertRaises(ValueError): simulate(self.case, random.Random(0), 'unknown')

    def test_seed_reproducibility(self):
        self.assertEqual(single_seed(0, 12), single_seed(0, 12))

    def test_run_metadata(self):
        r = run(2, 12)
        self.assertEqual(r['metadata']['seeds'], 2)
        self.assertEqual(len(r['trials']), 2)

    def test_unsupported_inputs(self):
        for kwargs in ({'seeds':0}, {'cases':0}):
            with self.assertRaises(ValueError): run(**kwargs)

    def test_no_oracle_leakage_in_seat_cards(self):
        # Cards expose only role, scores, weight, risk flags, not true utility.
        card = make_cards(self.case, random.Random(1))[0]
        self.assertNotIn('true_scores', vars(card))
        self.assertNotIn('values', vars(card))

    def test_audited_cap_is_paired(self):
        a = simulate(self.case, random.Random(33), 'audited_specialists')
        b = simulate(self.case, random.Random(33), 'audited_with_ruin_cap')
        self.assertEqual(a.cards, b.cards)
        self.assertEqual(a.recommended, b.recommended)
        self.assertEqual(a.score, b.score)

    def test_conformity_reduces_spread_without_new_evidence(self):
        a = simulate(self.case, random.Random(33), 'seven_correlated')
        b = simulate(self.case, random.Random(33), 'debate_no_evidence')
        self.assertEqual(a.recommended, b.recommended)
        self.assertLessEqual(b.spread, a.spread)

    def test_empty_cards_fails(self):
        with self.assertRaises(ValueError): aggregate((), ruin_cap=False, calls=1)

    def test_random_seed_changes_observations(self):
        a = make_cards(self.case, random.Random(1))
        b = make_cards(self.case, random.Random(2))
        self.assertNotEqual(a[0].scores, b[0].scores)

if __name__ == '__main__': unittest.main()
