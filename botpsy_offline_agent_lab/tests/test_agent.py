import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import execute,run_agent,demo_llm,read_db

class AgentTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.p=Path(self.t.name)/'db.json'
    def tearDown(self):self.t.cleanup()
    def test_notes(self):
        assert 'saved' in execute('add_note',{'text':'test note'},self.p)
        assert len(execute('search_notes',{'query':'TEST'},self.p)['notes'])==1
    def test_reject_unknown_tool(self):
        assert 'error' in execute('run_shell',{'command':'oops'},self.p)
        assert not self.p.exists()
    def test_reject_bad_args(self):
        assert 'error' in execute('add_note',{'text':'x','extra':'y'},self.p)
    def test_approval(self):
        task=execute('add_task',{'title':'Read a paper'},self.p)['saved']
        rejected=execute('complete_task',{'task_id':task['id']},self.p,lambda _:False)
        assert 'error' in rejected
        assert not read_db(self.p)['tasks'][0]['done']
        accepted=execute('complete_task',{'task_id':task['id']},self.p,lambda _:True)
        assert accepted['completed']['done']
    def test_mock_agent(self):
        out=run_agent('remember something',llm=demo_llm,path=self.p)
        assert out['trace'][0]['tool']=='add_note'
        assert len(read_db(self.p)['notes'])==1
if __name__=='__main__':unittest.main()
