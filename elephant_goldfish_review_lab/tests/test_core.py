import json
import tempfile
import unittest
from pathlib import Path
from review_lab.core import sha256_text, lint_spec, prepare, gate, packet, REVIEWERS, check_response

class ReviewLabTests(unittest.TestCase):
    def test_digest_stable(self):
        self.assertEqual(sha256_text('hello'), sha256_text('hello'))
    def test_digest_changes(self):
        self.assertNotEqual(sha256_text('hello'), sha256_text('hello!'))
    def test_incomplete_spec_lint(self):
        s = (Path(__file__).parent.parent/'examples/webhook_spec_v1.md').read_text()
        self.assertTrue(any(x['code']=='PLACEHOLDER' for x in lint_spec(s)))
    def test_complete_spec_no_placeholder(self):
        s = (Path(__file__).parent.parent/'examples/webhook_spec_v2.md').read_text()
        self.assertFalse(any(x['code']=='PLACEHOLDER' for x in lint_spec(s)))
    def test_packet_includes_spec_not_chat_history(self):
        p = packet('security', '# design', 'deadbeef')
        self.assertIn('# design', p)
        self.assertIn('Start a NEW conversation', p)
        self.assertNotIn('past discussion:', p)
    def test_prepare_generates_five(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp)/'spec.md'; spec.write_text('# spec')
            out = Path(tmp)/'out'
            m = prepare(spec, out)
            self.assertEqual(len(list((out/'packets').glob('*.md'))), 5)
            self.assertEqual(m['spec_sha256'], sha256_text('# spec'))
    def test_missing_reviews_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec=Path(tmp)/'spec.md'; spec.write_text('# spec')
            self.assertTrue(gate(spec, Path(tmp)/'responses')['verdict'].startswith('NO-GO'))
    def test_wrong_digest_rejected(self):
        p={'reviewer':'security','status':'completed','spec_sha256':'wrong','findings':[]}
        self.assertTrue(check_response(p,'security','right'))
    def test_all_reviews_completed_eligible_not_autoapprove(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec=Path(tmp)/'spec.md'; spec.write_text('# spec')
            d=sha256_text('# spec'); folder=Path(tmp)/'responses'; folder.mkdir()
            for role in REVIEWERS:
                (folder/f'{role}.json').write_text(json.dumps({'reviewer':role,'status':'completed','spec_sha256':d,'findings':[]}))
            self.assertIn('HUMAN REVIEW', gate(spec,folder)['verdict'])
    def test_major_finding_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec=Path(tmp)/'spec.md'; spec.write_text('# spec')
            d=sha256_text('# spec'); folder=Path(tmp)/'responses'; folder.mkdir()
            for role in REVIEWERS:
                findings=([{'id':'S-1','severity':'major','evidence':'Missing secret rotation','recommendation':'Add rotation plan'}] if role=='security' else [])
                (folder/f'{role}.json').write_text(json.dumps({'reviewer':role,'status':'completed','spec_sha256':d,'findings':findings}))
            self.assertEqual(gate(spec,folder)['blocking_count'],1)
            self.assertTrue(gate(spec,folder)['verdict'].startswith('NO-GO'))
    def test_invalid_severity_rejected(self):
        p={'reviewer':'security','status':'completed','spec_sha256':'good','findings':[{'id':'a','severity':'great','evidence':'x','recommendation':'y'}]}
        self.assertTrue(check_response(p,'security','good'))
    def test_failed_review_never_counts_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec=Path(tmp)/'spec.md'; spec.write_text('# spec')
            d=sha256_text('# spec'); folder=Path(tmp)/'responses'; folder.mkdir()
            for role in REVIEWERS:
                status='failed' if role=='security' else 'completed'
                (folder/f'{role}.json').write_text(json.dumps({'reviewer':role,'status':status,'spec_sha256':d,'findings':[]}))
            self.assertTrue(gate(spec,folder)['verdict'].startswith('NO-GO'))

if __name__=='__main__': unittest.main()
