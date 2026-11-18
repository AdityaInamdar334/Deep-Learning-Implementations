import json, sqlite3, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag_lab import generate,read_jsonl,CORPUS,QUESTIONS,chunks_of,build_index,lexical_search,fts_query,grade,normalize_answer
class RagTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):generate(17)
 def test_1000_docs(self):self.assertEqual(len(read_jsonl(CORPUS)),1000)
 def test_50_questions(self):
  q=read_jsonl(QUESTIONS);self.assertEqual(len(q),50);self.assertEqual(sum(x['answerable'] for x in q),40)
 def test_chunk_id_and_overlap(self):
  c=chunks_of(read_jsonl(CORPUS)[0],10,3);self.assertGreater(len(c),1);self.assertEqual(c[0]['text'].split()[-3:],c[1]['text'].split()[:3])
 def test_invalid_overlap(self):
  with self.assertRaises(ValueError):chunks_of(read_jsonl(CORPUS)[0],5,5)
 def test_fts_retrieval(self):
  with tempfile.TemporaryDirectory() as d:
   db=Path(d)/'index.sqlite';build_index(db,100,20);q=read_jsonl(QUESTIONS)[0];result=lexical_search(db,q['question'],20);self.assertTrue(any(x['doc_id'] in q['gold_doc_ids'] for x in result))
 def test_citation_support(self):
  d=read_jsonl(CORPUS)[0];q={'answerable':True,'expected_answer':d['answer'],'gold_doc_ids':[d['doc_id']]};ev=[{'doc_id':d['doc_id'],'chunk_id':d['doc_id']+':c0','text':d['text']}]
  result=grade(q,ev,{'answer':d['answer'],'citations':[ev[0]['chunk_id']]});self.assertTrue(result['grounded_correct'])
 def test_fake_citation_rejected(self):
  d=read_jsonl(CORPUS)[0];q={'answerable':True,'expected_answer':d['answer'],'gold_doc_ids':[d['doc_id']]};result=grade(q,[],{'answer':d['answer'],'citations':['fake']});self.assertFalse(result['grounded_correct'])
 def test_abstain(self):
  q={'answerable':False,'expected_answer':'UNKNOWN','gold_doc_ids':[]};self.assertTrue(grade(q,[],{'answer':'UNKNOWN','citations':[]})['grounded_correct'])
 def test_unjustified_abstention_citation(self):
  q={'answerable':False,'expected_answer':'UNKNOWN','gold_doc_ids':[]};self.assertFalse(grade(q,[],{'answer':'UNKNOWN','citations':['fake']})['grounded_correct'])
 def test_normalize(self):self.assertEqual(normalize_answer('  TIER  Cedar\n'), 'tier cedar')
if __name__=='__main__':unittest.main()
