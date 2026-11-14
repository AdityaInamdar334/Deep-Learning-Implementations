import unittest

import numpy as np
from retrieval import inputs, search


class RetrievalTests(unittest.TestCase):
    def test_known_ids_and_partial_batch(self):
        corpus = np.eye(4, dtype=np.float32)
        queries = corpus[[2, 0, 3]]
        for batch in (1, 2, 8):
            result = search(queries, corpus, batch, lambda x: x.argmax(axis=1), np.concatenate)
            np.testing.assert_array_equal(result, [2, 0, 3])

    def test_seed_replays(self):
        first = inputs(21, 5, 7, 17)
        second = inputs(21, 5, 7, 17)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
