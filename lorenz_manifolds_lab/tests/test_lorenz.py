import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lorenz_lab import (
    SIGMA,
    RHO,
    BETA,
    lorenz,
    trajectory,
    plot_attractor,
    plot_sensitivity,
    plot_saddle,
    plot_local_eigenvectors,
)


class TestLorenzLab(unittest.TestCase):
    def test_equilibria_fixed_points(self):
        # Origin is a fixed point
        np.testing.assert_allclose(lorenz(0.0, [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], atol=1e-12)
        # Symmetric convective fixed points C+ and C-
        c = math.sqrt(BETA * (RHO - 1.0))
        np.testing.assert_allclose(lorenz(0.0, [c, c, RHO - 1.0]), [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(lorenz(0.0, [-c, -c, RHO - 1.0]), [0.0, 0.0, 0.0], atol=1e-12)

    def test_trajectory_shape_and_boundedness(self):
        traj = trajectory([1.0, 1.0, 1.0], duration=10.0, samples=500)
        self.assertEqual(traj.shape, (500, 3))
        self.assertTrue(np.all(np.isfinite(traj)))
        self.assertLess(np.max(np.abs(traj)), 100.0)

    def test_sensitive_dependence(self):
        a = trajectory([1.0, 1.0, 1.0], duration=35.0, samples=1500)
        b = trajectory([1.0 + 1e-8, 1.0, 1.0], duration=35.0, samples=1500)
        sep = np.linalg.norm(a - b, axis=1)
        self.assertAlmostEqual(sep[0], 1e-8, places=12)
        self.assertGreater(np.max(sep), 1.0)

    def test_origin_jacobian_eigenvalues_and_dissipation(self):
        with tempfile.TemporaryDirectory() as tmp:
            eigs = np.sort(plot_local_eigenvectors(Path(tmp)).real)
        # 2 stable (negative) and 1 unstable (positive) eigenvalue
        self.assertEqual(int(np.sum(eigs < 0)), 2)
        self.assertEqual(int(np.sum(eigs > 0)), 1)
        # Exact z-direction eigenvalue is -BETA
        self.assertTrue(np.any(np.isclose(eigs, -BETA, atol=1e-10)))
        # Trace of Jacobian equals divergence -(SIGMA + 1 + BETA)
        self.assertAlmostEqual(float(np.sum(eigs)), -(SIGMA + 1.0 + BETA), places=10)

    def test_figure_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_attractor(out)
            plot_sensitivity(out)
            plot_saddle(out)
            plot_local_eigenvectors(out)
            expected = [
                'lorenz_attractor.png',
                'butterfly_effect.png',
                'stable_unstable_linear.png',
                'lorenz_local_directions.png',
            ]
            for name in expected:
                p = out / name
                self.assertTrue(p.exists(), f'Missing figure: {name}')
                self.assertGreater(p.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
