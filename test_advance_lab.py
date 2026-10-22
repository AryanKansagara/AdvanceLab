import math
import sqlite3
import unittest
from advance_lab import Ledger, RiskModel, chronological_split, evaluate, features, generate, psi, choose_threshold


class AdvanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate(200)
        cls.train, cls.validation, cls.test = chronological_split(cls.rows)
        cls.model = RiskModel().fit(cls.train)

    def test_reproducible_data(self):
        self.assertEqual(generate(30), generate(30))

    def test_disjoint_ordered_splits(self):
        self.assertLess(self.train[-1].sequence, self.validation[0].sequence)
        self.assertLess(self.validation[-1].sequence, self.test[0].sequence)
        with self.assertRaises(ValueError):
            chronological_split(list(reversed(self.rows)))

    def test_future_outcome_rejected(self):
        with self.assertRaises(ValueError):
            features(dict(self.rows[0].application, late=1))

    def test_bad_schema_and_numeric_inputs(self):
        for change in ({"monthly_income": 0}, {"monthly_income": float('nan')},
                       {"requested_amount": 0}, {"requested_amount": 501},
                       {"overdrafts_90d": 1.5}, {"income_variability": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                features(dict(self.rows[0].application, **change))
        with self.assertRaises(ValueError):
            features({})

    def test_train_only_scaling(self):
        expected = sum(features(r.application)[0] for r in self.train)/len(self.train)
        self.assertAlmostEqual(self.model.means[0], expected)

    def test_finite_predictions_and_metrics(self):
        for row in self.test:
            self.assertTrue(0 <= self.model.predict(row.application) <= 1)
        metrics = evaluate(self.model, self.test, choose_threshold(self.model, self.validation))
        self.assertTrue(0 <= metrics['brier_score'] <= 1)
        self.assertTrue(0 <= metrics['approval_rate'] <= 1)

    def test_drift_statistic(self):
        values = list(range(100))
        self.assertAlmostEqual(psi(values, values), 0)
        self.assertGreater(psi(values, [v+100 for v in values]), 0.2)
        with self.assertRaises(ValueError):
            psi([], values)

    def test_ledger_duplicate_and_transition_rules(self):
        ledger = Ledger()
        try:
            ledger.decide('a', self.rows[0].application, self.model, 1)
            with self.assertRaises(sqlite3.IntegrityError):
                ledger.decide('a', self.rows[0].application, self.model, 1)
            ledger.repay('a')
            with self.assertRaises(ValueError):
                ledger.repay('a')
            ledger.decide('b', self.rows[0].application, self.model, 0)
            with self.assertRaises(ValueError):
                ledger.repay('b')
        finally:
            ledger.close()


if __name__ == '__main__':
    unittest.main()
