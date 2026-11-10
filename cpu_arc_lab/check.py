import unittest
from cache import Request, execute, answer
class CacheTests(unittest.TestCase):
    def test_phrasing(self):
        self.assertEqual(answer("Batch 103 examples at 32 per batch, keep the last partial batch"),
                         answer("With batches of 32, group 103 examples and keep the remainder."))
    def test_new_parameters(self):
        self.assertEqual(answer("Batch 129 examples at 32 per batch, keep the last partial batch")["batches"],5)
    def test_combined(self):
        self.assertEqual(answer("With batches of 64, group 129 examples and keep the remainder")["batches"],3)
    def test_semantic_boundary(self):
        self.assertEqual(answer("With batches of 32, group 103 examples and drop the remainder")["batches"],3)
        self.assertIsNone(answer("With batches of 32, group 103 examples and keep the remainder per worker"))
    def test_bad_input(self):
        self.assertIsNone(answer("With batches of 0, group 103 examples and keep the remainder"))
        self.assertIsNone(answer("Batch 103 examples at 32 per batch"))
    def test_oracle(self):
        # Independent small-input oracle physically slices the examples.
        for n in range(130):
            items = list(range(n))
            for b in range(1,33):
                chunks = [items[i:i+b] for i in range(0,n,b)]
                for keep in (True,False):
                    expected=sum(keep or len(chunk)==b for chunk in chunks)
                    self.assertEqual(execute(Request(n,b,keep))["batches"],expected)
if __name__=="__main__":
    unittest.main(verbosity=2)
