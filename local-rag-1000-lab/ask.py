#!/usr/bin/env python3
"""Ask ad hoc questions against a locally indexed corpus; always print retrieved evidence."""
import argparse, json, time
from pathlib import Path
from rag_lab import ROOT,lexical_search,setup_dense,dense_search,rerank_cross,ollama_answer

def main():
 p=argparse.ArgumentParser();p.add_argument('question');p.add_argument('--db',default=str(ROOT/'data/chunks.sqlite'));p.add_argument('--model',default='qwen2.5:3b');p.add_argument('--retriever',choices=['lexical','dense'],default='lexical');p.add_argument('--rerank',choices=['none','cross'],default='none');p.add_argument('--k',type=int,default=5);p.add_argument('--candidates',type=int,default=20);p.add_argument('--no-generate',action='store_true');args=p.parse_args()
 if not Path(args.db).exists():p.error('Missing index: python rag_lab.py generate && python rag_lab.py index')
 t=time.perf_counter()
 if args.retriever=='dense':
  chunks,model,index=setup_dense(args.db);c=dense_search(args.question,chunks,model,index,args.candidates)
 else:c=lexical_search(args.db,args.question,args.candidates)
 if args.rerank=='cross':
  from sentence_transformers import CrossEncoder
  c=rerank_cross(args.question,c,CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'),args.k)
 else:c=c[:args.k]
 print('Evidence (retrieved, not yet verified):')
 for item in c:print(' ',item['chunk_id'],item['text'][:180].replace('\n',' '))
 print('Retrieval time:',round(time.perf_counter()-t,3),'s')
 if not args.no_generate:
  a,sec,metrics=ollama_answer(args.question,c,args.model)
  cite_ids={x['chunk_id'] for x in c}; claimed=a.get('citations',[])
  print('Answer:',a.get('answer'));print('Citations:',claimed)
  print('Citation IDs valid:',isinstance(claimed,list) and all(isinstance(x,str) and x in cite_ids for x in claimed))
  print('Generation time:',round(sec,3),'s')
if __name__=='__main__':main()
