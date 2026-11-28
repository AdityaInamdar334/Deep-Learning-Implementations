"""Run the verified offline workflow; forwards flags to lab.py."""
import subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parent
for cmd in [[sys.executable, '-m', 'unittest', '-v'], [sys.executable, 'lab.py', *sys.argv[1:]], [sys.executable, 'baselines.py'], [sys.executable, 'summarize.py']]:
    subprocess.run(cmd, cwd=root, check=True)
