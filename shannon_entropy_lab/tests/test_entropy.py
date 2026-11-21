import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy_lab import (
    PROBS,
    entropy,
    huffman,
    encode,
    decode,
    cross_entropy,
    text_prediction_example,
)

class TestEntropyLab(unittest.TestCase):
    def test_shannon_entropy_calculation(self):
        # H(X) for [0.5, 0.25, 0.125, 0.125] = 1.75
        self.assertAlmostEqual(entropy(PROBS), 1.75, places=5)
        # Uniform binary source: H = 1.0
        self.assertAlmostEqual(entropy({'A': 0.5, 'B': 0.5}), 1.0, places=5)
        # Deterministic source: H = 0.0
        self.assertAlmostEqual(entropy({'A': 1.0, 'B': 0.0}), 0.0, places=5)

    def test_huffman_prefix_free_property(self):
        codes = huffman(PROBS)
        # Verify no codeword is a prefix of any other codeword
        codewords = list(codes.values())
        for i, c1 in enumerate(codewords):
            for j, c2 in enumerate(codewords):
                if i != j:
                    self.assertFalse(
                        c2.startswith(c1),
                        f"Codeword '{c1}' is a prefix of '{c2}'"
                    )

    def test_lossless_roundtrip_encode_decode(self):
        codes = huffman(PROBS)
        message = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UP', 'UP', 'DOWN']
        bits = encode(message, codes)
        recovered = decode(bits, codes)
        self.assertEqual(message, recovered)

    def test_truncated_bitstream_raises_error(self):
        codes = huffman(PROBS)
        message = ['UP', 'DOWN', 'LEFT']
        bits = encode(message, codes)
        # Truncate by dropping the last bit
        with self.assertRaises(ValueError):
            decode(bits[:-1], codes)

    def test_cross_entropy_and_gibbs_inequality(self):
        # Gibbs inequality: H(P, Q) >= H(P) with equality iff P == Q
        p = PROBS
        q_uniform = {'UP': 0.25, 'DOWN': 0.25, 'LEFT': 0.25, 'RIGHT': 0.25}
        h_p = entropy(p)
        ce_wrong = cross_entropy(p, q_uniform)
        ce_true = cross_entropy(p, p)
        self.assertAlmostEqual(ce_true, h_p, places=5)
        self.assertGreater(ce_wrong, h_p)

    def test_conditioning_reduces_entropy(self):
        uni_bits, bi_bits = text_prediction_example()
        # Conditioning on previous character reduces per-character entropy
        self.assertLess(bi_bits, uni_bits)
        self.assertGreater(bi_bits, 0.0)

if __name__ == '__main__':
    unittest.main()
