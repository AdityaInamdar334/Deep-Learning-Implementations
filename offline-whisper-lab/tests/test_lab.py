import sys,unittest,tempfile,subprocess,json,wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import word_errors,parse_segments,timestamp_ms,audio_duration,write_srt
from record import command
from transcribe import run
class LabTests(unittest.TestCase):
    def test_wer_exact(self):self.assertEqual(word_errors('Hello world','hello, world!')['wer'],0)
    def test_wer_substitution(self):
        r=word_errors('one two three','one four three');self.assertEqual((r['substitutions'],r['deletions'],r['insertions']), (1,0,0))
    def test_wer_deletion(self):self.assertEqual(word_errors('one two three','one three')['deletions'],1)
    def test_wer_insertion(self):self.assertEqual(word_errors('one three','one two three')['insertions'],1)
    def test_wer_empty_reference(self):
        with self.assertRaises(ValueError):word_errors('...','something')
    def test_parse_whisper_segments(self):
        got=parse_segments({'transcription':[{'offsets':{'from':1200,'to':2300},'text':' hello'}]})
        self.assertEqual(got,[{'start_ms':1200,'end_ms':2300,'text':'hello'}])
    def test_srt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'a.srt';write_srt([{'start_ms':100,'end_ms':1200,'text':'Hello'}],out)
            self.assertIn('00:00:00,100 --> 00:00:01,200',out.read_text())
    def test_record_command(self):
        c=command(5,Path('x.wav'),':1','avfoundation');self.assertIn('16000',c)
    def test_record_invalid(self):
        with self.assertRaises(ValueError):command(0,Path('x.wav'),':0','avfoundation')
    def test_mock_transcriber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audio=root/'a.wav';model=root/'model.bin';model.write_bytes(b'mock')
            with wave.open(str(audio),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0\0'*16000)
            script=root/'fake_whisper'
            script.write_text('#!'+sys.executable+'\nimport sys,json\na=sys.argv\np=a[a.index("-of")+1]+".json"\nopen(p,"w").write(json.dumps({"transcription":[{"offsets":{"from":0,"to":1000},"text":"hello world"}]}))\n')
            script.chmod(0o755)
            ref=root/'ref.txt';ref.write_text('Hello world')
            rows=run(script,model,audio,root/'out',repeat=1,reference=ref)
            self.assertEqual(rows[0]['wer'],0.0)
            self.assertAlmostEqual(rows[0]['audio_seconds'],1.0)
            self.assertTrue((root/'out'/'trial_01.srt').exists())
if __name__=='__main__':unittest.main()
