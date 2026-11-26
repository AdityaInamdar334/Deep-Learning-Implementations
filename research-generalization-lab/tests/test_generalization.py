import unittest
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


class TestGeneralizationLab(unittest.TestCase):
    def test_memorization_vs_generalization(self):
        X, y = load_digits(return_X_y=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        clean = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)).fit(X_tr, y_tr)
        self.assertEqual(clean.score(X_tr, y_tr), 1.0)
        self.assertGreater(clean.score(X_te, y_te), 0.95)

        y_shuf = np.random.default_rng(42).permutation(y_tr)
        shuf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)).fit(X_tr, y_shuf)
        self.assertEqual(shuf.score(X_tr, y_shuf), 1.0)
        self.assertLess(shuf.score(X_te, y_te), 0.20)

    def test_polynomial_capacity_and_ridge(self):
        rng = np.random.default_rng(42)
        x = np.sort(rng.uniform(0, 1, 25)).reshape(-1, 1)
        y = np.sin(2 * np.pi * x.ravel()) + rng.normal(0, 0.15, 25)
        grid = np.linspace(0, 1, 500).reshape(-1, 1)
        truth = np.sin(2 * np.pi * grid.ravel())

        def mse_for(model):
            return mean_squared_error(truth, model.fit(x, y).predict(grid))

        mse_d1 = mse_for(make_pipeline(PolynomialFeatures(1, include_bias=False), LinearRegression()))
        mse_d5 = mse_for(make_pipeline(PolynomialFeatures(5, include_bias=False), LinearRegression()))
        mse_d15 = mse_for(make_pipeline(PolynomialFeatures(15, include_bias=False), LinearRegression()))
        mse_d15_ridge = mse_for(make_pipeline(PolynomialFeatures(15, include_bias=False), Ridge(alpha=1e-3)))

        self.assertLess(mse_d5, mse_d1)
        self.assertLess(mse_d5, mse_d15)
        self.assertLess(mse_d15_ridge, mse_d15)

    def test_artifacts_present(self):
        self.assertTrue((ROOT / 'article.md').exists())
        self.assertTrue((ROOT / 'results.txt').exists())
        expected_figs = [
            'capacity.png',
            'confusion.png',
            'digits.png',
            'gradient_descent.png',
            'label_corruption.png',
            'memorization.png',
            'noise_features.png',
            'pca_digits.png',
            'polynomial_capacity.png',
            'regularization.png',
        ]
        for name in expected_figs:
            p = ROOT / 'figures' / name
            self.assertTrue(p.exists(), f'Missing figure: {name}')
            self.assertGreater(p.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
