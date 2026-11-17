#!/usr/bin/env python3
"""Create local Ollama model aliases from manually downloaded GGUFs.
Never downloads or quantizes weights. Use official Ollama import support.
"""
import argparse, subprocess, tempfile
from pathlib import Path

NAMES={'q4':'Qwen2.5-3B-Instruct-Q4_K_M.gguf','q8':'Qwen2.5-3B-Instruct-Q8_0.gguf','f16':'Qwen2.5-3B-Instruct-f16.gguf'}
p=argparse.ArgumentParser();p.add_argument('--models-dir',default='models');p.add_argument('--only',nargs='+',choices=list(NAMES),default=list(NAMES));a=p.parse_args()
folder=Path(a.models_dir).resolve()
for k in a.only:
 model_path=folder/NAMES[k]
 if not model_path.is_file():
  print('SKIP missing',model_path);continue
 alias='quant-'+k
 # Raw /api/generate endpoint supplies full prompt including template. Modelfile needs only FROM.
 with tempfile.TemporaryDirectory() as tmp:
  spec=Path(tmp)/'Modelfile';spec.write_text('FROM '+str(model_path)+'\n')
  print('Importing',alias,'from',model_path)
  subprocess.run(['ollama','create',alias,'-f',str(spec)],check=True)
