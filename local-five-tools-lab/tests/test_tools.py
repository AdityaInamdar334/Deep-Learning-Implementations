import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import calculate, read_file, search_files, execute, validate_args, ToolError, REPORTS
from agent import run
from evaluate import score_case

class ToolTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculate('(17*23)+5')['result'],396)
    def test_no_python_execution(self):
        with self.assertRaises(ToolError): calculate('__import__("os").system("whoami")')
    def test_path_traversal(self):
        with self.assertRaises(ToolError): read_file('../../agent.py')
    def test_search(self):
        self.assertEqual(search_files('release freeze')['matches'][0]['path'],'docs/release.md')
    def test_schema(self):
        with self.assertRaises(ToolError): validate_args('calculate',{'expression':'2+2','extra':'bad'})
    def test_write_denied(self):
        path=REPORTS/'test_denied_unique.json'
        if path.exists(): path.unlink()
        result=execute('write_json_report',{'filename':path.name,'report':{'a':1}},approved=False)
        self.assertEqual(result['status'],'denied')
        self.assertFalse(path.exists())
    def test_write_approved(self):
        path=REPORTS/'test_approved_unique.json'
        if path.exists(): path.unlink()
        result=execute('write_json_report',{'filename':path.name,'report':{'a':1}},approved=True)
        self.assertEqual(result['status'],'written')
        self.assertEqual(__import__('json').loads(path.read_text()),{'a':1})
        path.unlink()
    def test_mock_agent_call_and_trace(self):
        responses=iter([
            {'message':{'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'calculate','arguments':{'expression':'17*23+5'}}}]}},
            {'message':{'role':'assistant','content':'The result is 396.'}}
        ])
        result=run('calculate',client=lambda messages,model:next(responses),approval='deny')
        self.assertEqual(result['events'][0]['result']['result'],396)
        self.assertIn('396',result['answer'])
    def test_mock_policy_bypass(self):
        responses=iter([
            {'message':{'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'write_json_report','arguments':{'filename':'injected_test.json','report':{'x':1}}}}]}},
            {'message':{'role':'assistant','content':'Permission denied.'}}
        ])
        path=REPORTS/'injected_test.json'
        if path.exists():path.unlink()
        result=run('write',client=lambda messages,model:next(responses),approval='deny')
        self.assertEqual(result['events'][0]['status'],'denied')
        self.assertFalse(path.exists())

if __name__=='__main__':unittest.main()
