import unittest
import numpy as np
import pandas as pd

from audit_stable_weights import candidate_grids
from mlb_model.weight_audit import DEFAULT_WEIGHTS, choose_weights, weight_grid


class StableWeightTests(unittest.TestCase):
    def test_constraints_and_normalization(self):
        grids = candidate_grids()
        for name, grid in grids.items():
            self.assertTrue(grid)
            for w in grid:
                self.assertAlmostEqual(sum(w.values()), 1)
                self.assertTrue(all(v >= 0 for v in w.values()))
                if name == 'hit_cap15':
                    self.assertTrue(all(abs(w[k]-DEFAULT_WEIGHTS[k]) <= .15+1e-12 for k in w))
        for original, shrunk in zip(weight_grid(), grids['hit_shrink50']):
            for key in original:
                self.assertAlmostEqual(shrunk[key]-DEFAULT_WEIGHTS[key], .5*(original[key]-DEFAULT_WEIGHTS[key]))

    def test_sparse_evidence_uses_fixed_prior(self):
        frame = pd.DataFrame(dict(y=np.arange(60)%2, linear=.7, tree=.6, run=.8,
                                  inner_fold=np.repeat([1, 2, 3], 20)))
        for name in ('hit_shrink50', 'hit_cap15'):
            chosen, info, _ = choose_weights(frame, 'hit_rate', candidate_grids()[name], DEFAULT_WEIGHTS)
            self.assertEqual(chosen, DEFAULT_WEIGHTS)
            self.assertEqual(info['eligible_grid'], 0)
        self.assertEqual(choose_weights(pd.DataFrame(), fallback_weights=DEFAULT_WEIGHTS)[0], DEFAULT_WEIGHTS)

    def test_log_loss_cannot_escape_restricted_candidates(self):
        frame = pd.DataFrame(dict(y=[0, 1]*30, linear=.7, tree=.5, run=.9))
        only = dict(linear=.44, tree=.36, run=.20)
        self.assertEqual(choose_weights(frame, candidates=[only])[0], only)
        with self.assertRaises(ValueError):
            choose_weights(frame, candidates=[])
        with self.assertRaises(ValueError):
            choose_weights(frame, candidates=[dict(linear=1, tree=1, run=1)])


if __name__ == '__main__':
    unittest.main()
