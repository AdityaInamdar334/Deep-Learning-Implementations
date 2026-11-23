"""Educational agent-computer simulation, not a Nod integration. Standard library only."""
import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

SCHEMA = '''CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, task TEXT NOT NULL, action TEXT NOT NULL, status TEXT NOT NULL, details TEXT NOT NULL, ts TEXT NOT NULL); CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, text TEXT NOT NULL);'''

class MiniAgentOS:
    def __init__(self, path='demo.db'):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.commit()
    def record(self, task, action, status, details):
        self.db.execute('INSERT INTO events(task,action,status,details,ts) VALUES(?,?,?,?,?)', (task,action,status,json.dumps(details),datetime.now(timezone.utc).isoformat()))
        self.db.commit()
    def request(self, task, action, args=None, approved=False):
        args=args or {}
        allowed={'read_mock_page','save_note','list_notes'}
        if action not in allowed:
            self.record(task,action,'denied',{'reason':'not in allowlist'})
            return {'error':'tool unavailable'}
        if action=='read_mock_page':
            if args:
                return self._invalid(task,action)
            result={'title':'Example Project Status','headline':'Build passed','revision':'v1'}
        elif action=='list_notes':
            if args:
                return self._invalid(task,action)
            result={'notes':[r[0] for r in self.db.execute('SELECT text FROM notes ORDER BY id')]}
        else:
            if set(args)!={'text'} or not isinstance(args['text'],str) or not (1<=len(args['text'])<=500):
                return self._invalid(task,action)
            if not approved:
                self.record(task,action,'approval_required',{'preview':args['text']})
                return {'requires_approval':True,'preview':args['text']}
            self.db.execute('INSERT INTO notes(text) VALUES(?)',(args['text'],))
            self.db.commit()
            result={'saved':args['text']}
        self.record(task,action,'completed',result)
        return result
    def _invalid(self,task,action):
        self.record(task,action,'denied',{'reason':'invalid arguments'})
        return {'error':'invalid arguments'}
    def history(self):
        return list(self.db.execute('SELECT task,action,status FROM events ORDER BY id'))

def demo(path):
    os=MiniAgentOS(path)
    print('Read public mock page:', os.request('morning-brief','read_mock_page'))
    args={'text':'The demo project build passed at revision v1.'}
    print('Propose persistent change:',os.request('morning-brief','save_note',args))
    print('Approve and execute:',os.request('morning-brief','save_note',args,approved=True))
    print('List notes:',os.request('morning-brief','list_notes'))
    print('Reject untrusted capability:',os.request('morning-brief','run_shell',{'cmd':'echo no'}))
    print('Audit:',os.history())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--db',default='demo.db')
    args=p.parse_args()
    demo(args.db)
