import tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from searchlab import chunks,fts_query,connect,index,lexical,fuse,metrics,search
class Tests(unittest.TestCase):
 def test_chunk(self):self.assertEqual(chunks('a b c d e f g h i j k',8,2),['a b c d e f g h','g h i j k'])
 def test_invalid_chunk(self):
  with self.assertRaises(ValueError):chunks('x y',8,8)
 def test_fts_escape(self):self.assertIn('"alpha"',fts_query('alpha; drop table'))
 def test_rrf(self):
  a=[dict(id='a'),dict(id='b')];b=[dict(id='b'),dict(id='a')]
  self.assertEqual(len(fuse(a,b)),2)
 def test_metrics(self):
  x=metrics(['a'],['b','a'],k=2);self.assertEqual(x['recall'],1);self.assertEqual(x['mrr'],.5)
 def test_index_change_delete(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'docs';root.mkdir();p=root/'x.md';p.write_text('deployment alpha cactus notes identifier abcdef')
   db=Path(d)/'s.db'; self.assertEqual(index(root,db)['new_or_changed'],1)
   self.assertEqual(index(root,db)['unchanged'],1)
   self.assertEqual(len(search(db,'cactus')[0]),1)
   p.write_text('deployment alpha mango notes identifier abcdef')
   self.assertEqual(index(root,db)['new_or_changed'],1)
   self.assertEqual(search(db,'cactus')[0],[])
   p.unlink();self.assertEqual(index(root,db)['removed'],1)
 def test_no_symlink(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'docs';root.mkdir();other=Path(d)/'private.txt';other.write_text('secret')
   (root/'hidden.txt').symlink_to(other);db=Path(d)/'s.db'
   index(root,db);self.assertEqual(search(db,'secret')[0],[])
if __name__=='__main__':unittest.main()
