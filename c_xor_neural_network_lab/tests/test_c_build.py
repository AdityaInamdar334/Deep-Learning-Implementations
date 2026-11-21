import subprocess
import unittest
from pathlib import Path

class TestCXor(unittest.TestCase):
    def test_c_xor_tests(self):
        lab_dir = Path(__file__).resolve().parents[1]
        res = subprocess.run(["make", "-C", str(lab_dir), "test"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"make test failed:\n{res.stderr}\n{res.stdout}")
        self.assertIn("tests passed", res.stdout)
        subprocess.run(["make", "-C", str(lab_dir), "clean"], capture_output=True)

if __name__ == '__main__':
    unittest.main()
