import tempfile
import unittest
from pathlib import Path
from defensive_lab.provenance import verify_artifact, sha256_file
from defensive_lab.paths import naive_protected_check, canonical_protected_check
from defensive_lab.versions import langflow_triage
from defensive_lab.ioc import scan_root


class LabTests(unittest.TestCase):
    def test_checksum_known_good(self):
        path = Path('fixtures/publisher-clean.txt')
        self.assertTrue(verify_artifact(path, sha256_file(path))['matches_trusted_manifest'])

    def test_checksum_modified_same_identity(self):
        clean = Path('fixtures/publisher-clean.txt')
        modified = Path('fixtures/mirror-different-bytes.txt')
        self.assertFalse(verify_artifact(modified, sha256_file(clean))['matches_trusted_manifest'])

    def test_raw_path_bypass(self):
        self.assertFalse(naive_protected_check('protected.conf.', {'protected.conf'}))

    def test_normalized_path_caught(self):
        self.assertTrue(canonical_protected_check('protected.conf.', {'protected.conf'}))

    def test_affected_old_langflow(self):
        self.assertEqual(langflow_triage('1.12.8')['status'], 'affected_in_IBM_bulletin')

    def test_backport_needs_vendor_confirmation(self):
        self.assertIn('not_verified', langflow_triage('1.12.10')['status'])

    def test_newer_outside_bulletin(self):
        self.assertEqual(langflow_triage('1.13.1')['status'], 'outside_original_affected_range')

    def test_empty_scan_not_claim_clean(self):
        with tempfile.TemporaryDirectory() as d:
            report = scan_root(d)
            self.assertFalse(report['findings'])
            self.assertIn('does NOT prove', report['assessment'])

    def test_marker_detection(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / '.ssh'
            folder.mkdir()
            (folder / 'authorized_keys').write_text('synthetic-test-only-key svc-telemetry\n')
            report = scan_root(d)
            self.assertTrue(any(f['kind'] == 'ssh_key_marker' for f in report['findings']))

    def test_persistence_path_detection(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'Library' / 'LaunchAgents'
            folder.mkdir(parents=True)
            (folder / 'com.apple.sysmond.plist').write_text('harmless test fixture')
            report = scan_root(d)
            self.assertTrue(any(f['kind'] == 'persistence_path_present' for f in report['findings']))

if __name__ == '__main__':
    unittest.main()
