import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import grade,summaries

class GradingTests(unittest.TestCase):
 def test_exact(self):self.assertEqual(grade({'grader':'exact','expected':'PASS'},'PASS\n'),1)
 def test_exact_extra_text(self):self.assertEqual(grade({'grader':'exact','expected':'PASS'},'It is PASS'),0)
 def test_integer(self):self.assertEqual(grade({'grader':'integer','expected':1234},'1,234'),1)
 def test_integer_failure(self):self.assertEqual(grade({'grader':'integer','expected':12},'12 or 13'),0)
 def test_json_exact(self):self.assertEqual(grade({'grader':'json','expected':{'a':1,'b':False}},'{"a":1,"b":false}'),1)
 def test_json_strict_type(self):self.assertEqual(grade({'grader':'json','expected':{'a':1}},'{"a":true}'),0)
 def test_json_extra(self):self.assertEqual(grade({'grader':'json','expected':{'a':1}},'{"a":1,"b":2}'),0)
 def test_json_not_codefence(self):self.assertEqual(grade({'grader':'json','expected':{'a':1}},'```json\n{"a":1}\n```'),0)
 def test_summary(self):
  rows=[{'model':'a','category':'arithmetic','score':1,'error':'','decode_tps':10,'prefill_tps':20,'wall_s':1,'peak_gpu_mib':100},{'model':'a','category':'arithmetic','score':0,'error':'','decode_tps':12,'prefill_tps':22,'wall_s':2,'peak_gpu_mib':110}]
  s=summaries(rows)[0];self.assertEqual(s['accuracy'],0.5);self.assertEqual(s['median_decode_tps'],11);self.assertEqual(s['max_gpu_mib'],110)
if __name__=='__main__':unittest.main()
