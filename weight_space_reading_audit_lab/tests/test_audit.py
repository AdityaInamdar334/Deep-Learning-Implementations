import unittest
import numpy as np
from weight_audit.core import (
    CLASS_NAMES, dataset, train_classifier, map_labels_to_new_indices,
    tiny_forward_exercise, hidden_unit_symmetry
)

class WeightReadingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train_classifier(7)
        cls.x, cls.y = dataset(60, 107)
        cls.order = np.array([2,0,3,1])
        cls.permuted = cls.model.permute_outputs(cls.order)

    def test_original_accuracy(self):
        self.assertGreater(self.model.accuracy(self.x,self.y),0.97)

    def test_permuted_logits(self):
        np.testing.assert_allclose(self.permuted.logits(self.x),self.model.logits(self.x)[:,self.order])

    def test_permuted_accuracy(self):
        y_new = map_labels_to_new_indices(self.y,self.order)
        self.assertAlmostEqual(self.permuted.accuracy(self.x,y_new),self.model.accuracy(self.x,self.y))

    def test_all_label_names_move(self):
        self.assertTrue(all(self.permuted.class_names[i] != CLASS_NAMES[i] for i in range(4)))

    def test_shuffled_bad(self):
        self.assertLess(self.model.shuffle_weights(38).accuracy(self.x,self.y), 0.80)

    def test_hidden_neuron_symmetry(self):
        e=tiny_forward_exercise()
        h=hidden_unit_symmetry()
        self.assertTrue(np.allclose(e['output_logits'],h['unchanged_output_logits']))

    def test_tiny_forward_arithmetic(self):
        ex=tiny_forward_exercise()
        x=np.array(ex['x'])
        z=x@np.array(ex['w1'])+np.array(ex['b1'])
        np.testing.assert_allclose(z,np.array(ex['hidden_preactivation']))
        h=np.maximum(z,0)
        np.testing.assert_allclose(h,np.array(ex['hidden_activation']))
        np.testing.assert_allclose(h@np.array(ex['w2'])+np.array(ex['b2']),np.array(ex['output_logits']))

if __name__ == '__main__':
    unittest.main()
