import unittest
import numpy as np
from lab import fixture, project, quantize_rows


class Int8Tests(unittest.TestCase):
    def test_hand_example(self):
        x = np.array([[64., .24, .24, .24]])
        w = np.array([[0., 1., 1., 1.]])
        self.assertEqual(project(x, w, 0, 0)[0].item(), 0)
        self.assertAlmostEqual(project(x, w, 0, 6)[0].item(), .72)

    def test_all_columns_bias_once(self):
        x, w, b = fixture()
        out, mask = project(x, w, b, .01)
        self.assertTrue(mask.all())
        np.testing.assert_allclose(out, x @ w.T + b)

    def test_zero_and_union(self):
        x = np.array([[0.,0.,0.],[-9.,1.,1.]])
        out, mask = project(x, np.eye(3), .5, 6)
        np.testing.assert_array_equal(mask, [True,False,False])
        np.testing.assert_allclose(out, x + .5)

    def test_packed_correction(self):
        x,w,b = fixture()
        q,s = quantize_rows(w)
        out,_ = project(x,w,b,.01,True)
        np.testing.assert_allclose(out, x @ (q*s).T + b)

    def test_accumulation_width(self):
        x = np.ones((1, 256))
        self.assertAlmostEqual(project(x,x,0,0)[0].item(), 256)


if __name__ == "__main__":
    unittest.main()
