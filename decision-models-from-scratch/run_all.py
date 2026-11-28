from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for script,args in [('offline.py',[]),('support.py',[]),('agent.py',[]),('cost.py',[]),('-m',['unittest','discover','-s','tests','-v'])]:
    result=subprocess.run([sys.executable,script,*args],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout,flush=True); (root/'results'/('tests.log' if script=='-m' else script.replace('.py','.log'))).write_text(result.stdout); result.check_returncode()
