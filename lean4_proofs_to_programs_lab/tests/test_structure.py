from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
class TestPackage(unittest.TestCase):
    def test_sources_present(self):
        self.assertTrue((ROOT / 'src/Proofs.lean').exists())
        self.assertTrue((ROOT / 'src/Main.lean').exists())
    def test_no_unfinished_proof(self):
        self.assertNotIn('sorry', (ROOT / 'src/Proofs.lean').read_text())
    def test_manifest_present(self):
        self.assertIn('lean4_lab', (ROOT / 'lakefile.toml').read_text())
if __name__ == '__main__': unittest.main()
