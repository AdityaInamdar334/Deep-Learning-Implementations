import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent_os import MiniAgentOS

class Tests(unittest.TestCase):
    def setUp(self): self.agent=MiniAgentOS(':memory:')
    def test_read(self): self.assertEqual(self.agent.request('a','read_mock_page')['headline'],'Build passed')
    def test_approval(self):
        self.assertTrue(self.agent.request('a','save_note',{'text':'x'})['requires_approval'])
        self.assertEqual(self.agent.request('a','list_notes')['notes'],[])
        self.agent.request('a','save_note',{'text':'x'},approved=True)
        self.assertEqual(self.agent.request('a','list_notes')['notes'],['x'])
    def test_denied(self): self.assertIn('error', self.agent.request('a','run_shell',{'cmd':'x'}))
    def test_bad_args(self): self.assertIn('error',self.agent.request('a','save_note',{'text':123},approved=True))
    def test_audit(self):
        self.agent.request('t','read_mock_page')
        self.assertEqual(self.agent.history()[0][2],'completed')

if __name__=='__main__': unittest.main()
