#!/usr/bin/env python3
"""Private local document search: SQLite FTS5 + optional dense embeddings + RRF."""
import argparse, hashlib, json, math, os, re, sqlite3, sys, time
from pathlib import Path

EX = {'.md','.markdown','.txt','.pdf'}
TOKEN = re.compile(r'[\w]+',re.UNICODE)

def digest(b): return hashlib.sha256(b).hexdigest()
def tokenize(s): return [x.lower() for x in TOKEN.findall(s) if len(x)>1][:48]
def fts_query(s):
    t=tokenize(s)
    return ' OR '.join('"'+x.replace('"','')+'"' for x in t)

def chunks(s, words=140, overlap=30):
    if words < 8 or overlap < 0 or overlap >= words: raise ValueError('Require words>=8 and 0<=overlap<words')
    tokens=s.split()
    out=[]
    for start in range(0,len(tokens),words-overlap):
        part=tokens[start:start+words]
        if not part: break
        out.append(' '.join(part))
        if start+words>=len(tokens): break
    return out

def pages(path):
    if path.suffix.lower()!='.pdf':
        yield 0,path.read_text(encoding='utf-8',errors='replace')
    else:
        try: import pymupdf
        except ImportError:
            try: import fitz as pymupdf
            except ImportError: raise RuntimeError('Install pymupdf for PDF indexing: pip install pymupdf')
        with pymupdf.open(str(path)) as doc:
            for page in doc:
                yield page.number+1, page.get_text('text',sort=True)

def connect(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, mtime_ns INTEGER, indexed_at TEXT DEFAULT CURRENT_TIMESTAMP);
      CREATE TABLE IF NOT EXISTS passages(id TEXT PRIMARY KEY, path TEXT NOT NULL, page INT NOT NULL, ordinal INT NOT NULL, text TEXT NOT NULL, FOREIGN KEY(path) REFERENCES files(path) ON DELETE CASCADE);
      CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(text, content='passages', content_rowid='rowid', tokenize='unicode61');
      CREATE TRIGGER IF NOT EXISTS pa AFTER INSERT ON passages BEGIN INSERT INTO passage_fts(rowid,text) VALUES(new.rowid,new.text); END;
      CREATE TRIGGER IF NOT EXISTS pd AFTER DELETE ON passages BEGIN INSERT INTO passage_fts(passage_fts,rowid,text) VALUES('delete',old.rowid,old.text); END;
      CREATE TRIGGER IF NOT EXISTS pu AFTER UPDATE ON passages BEGIN INSERT INTO passage_fts(passage_fts,rowid,text) VALUES('delete',old.rowid,old.text); INSERT INTO passage_fts(rowid,text) VALUES(new.rowid,new.text); END;''')
    return db

def index(root,dbpath,words=140,overlap=30,max_mb=12):
    root=Path(root).expanduser().resolve(); db=connect(dbpath)
    if not root.is_dir(): raise ValueError('Input root must be a directory')
    seen=set(); counts={'new_or_changed':0,'unchanged':0,'removed':0,'chunks':0,'skipped':0,'empty_pdf_pages':0}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in EX: continue
        if not path.resolve().is_relative_to(root): continue
        rel=str(path.relative_to(root));seen.add(rel)
        try:
            if path.stat().st_size>max_mb*1024*1024: counts['skipped']+=1; continue
            raw=path.read_bytes(); sha=digest(raw)
            old=db.execute('SELECT sha256 FROM files WHERE path=?',(rel,)).fetchone()
            if old and old['sha256']==sha: counts['unchanged']+=1;continue
            parts=[]
            for page,text in pages(path):
                if path.suffix.lower()=='.pdf' and not text.strip(): counts['empty_pdf_pages']+=1
                for j,part in enumerate(chunks(text,words,overlap)):
                    ident=digest(f'{rel}|{page}|{j}|{part}'.encode())[:20]
                    parts.append((ident,rel,page,j,part))
            with db:
                db.execute('DELETE FROM files WHERE path=?',(rel,))
                db.execute('INSERT INTO files(path,sha256,mtime_ns) VALUES(?,?,?)',(rel,sha,path.stat().st_mtime_ns))
                db.executemany('INSERT INTO passages(id,path,page,ordinal,text) VALUES(?,?,?,?,?)',parts)
            counts['new_or_changed']+=1;counts['chunks']+=len(parts)
        except Exception as exc:
            counts['skipped']+=1;print(f'SKIP {rel}: {exc}',file=sys.stderr)
    # Scan only in this indexed root; separate db per root recommended.
    for row in db.execute('SELECT path FROM files').fetchall():
        if row['path'] not in seen:
            with db: db.execute('DELETE FROM files WHERE path=?',(row['path'],))
            counts['removed']+=1
    db.close();return counts

def lexical(db,q,limit=20):
    expr=fts_query(q)
    if not expr:return []
    result=db.execute('''SELECT p.id,p.path,p.page,p.text,bm25(passage_fts) AS score
      FROM passage_fts JOIN passages p ON p.rowid=passage_fts.rowid WHERE passage_fts MATCH ?
      ORDER BY score LIMIT ?''',(expr,limit)).fetchall()
    return [dict(x) for x in result]

_DENSE_CACHE = {}
def dense(db,q,limit=20,model='sentence-transformers/all-MiniLM-L6-v2',local_only=False):
    try: from sentence_transformers import SentenceTransformer
    except ImportError:raise RuntimeError('Dense search requires sentence-transformers (see requirements-optional.txt)')
    import numpy as np
    rows=db.execute('SELECT id,path,page,text FROM passages ORDER BY id').fetchall()
    if not rows:return []
    signature=digest((model+'\n'+'\n'.join(r['id'] for r in rows)).encode())
    cachekey=(signature,model)
    if cachekey not in _DENSE_CACHE:
        encoder=SentenceTransformer(model,local_files_only=local_only)
        folder=Path('.searchlab_cache');folder.mkdir(exist_ok=True)
        cachefile=folder/(signature+'.npy')
        if cachefile.exists():embeddings=np.load(cachefile,allow_pickle=False)
        else:
            embeddings=encoder.encode([r['text'] for r in rows],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            np.save(cachefile,embeddings)
        _DENSE_CACHE.clear()
        _DENSE_CACHE[cachekey]=(encoder,embeddings)
    encoder,embeddings=_DENSE_CACHE[cachekey]
    query=encoder.encode([q],normalize_embeddings=True,convert_to_numpy=True)[0]
    try:
        import faiss
        idx=faiss.IndexFlatIP(embeddings.shape[1]);idx.add(embeddings.astype('float32'))
        scores,indices=idx.search(query.astype('float32')[None,:],min(limit,len(rows)))
        pairs=list(zip(indices[0],scores[0]))
    except ImportError:
        scores=embeddings@query
        order=np.argsort(-scores)[:limit]
        pairs=[(int(i),float(scores[i])) for i in order]
    return [dict(id=rows[i]['id'],path=rows[i]['path'],page=rows[i]['page'],text=rows[i]['text'],score=float(v)) for i,v in pairs if i>=0]

def fuse(a,b,k=60,limit=10):
    scores={}; objs={}
    for seq in (a,b):
        for rank,item in enumerate(seq,1):
            key=item['id'];scores[key]=scores.get(key,0)+1/(k+rank);objs[key]=item
    return [dict(objs[key],score=scores[key]) for key in sorted(scores,key=lambda x:(-scores[x],x))[:limit]]

def search(dbpath,q,mode='bm25',k=5,candidates=20,model='sentence-transformers/all-MiniLM-L6-v2',local_only=False):
    db=connect(dbpath); t=time.perf_counter()
    if mode=='bm25': items=lexical(db,q,max(k,candidates))[:k]
    elif mode=='dense':items=dense(db,q,max(k,candidates),model,local_only)[:k]
    elif mode=='hybrid':
        a=lexical(db,q,max(k,candidates));b=dense(db,q,max(k,candidates),model,local_only)
        items=fuse(a,b,limit=k)
    else: raise ValueError(mode)
    elapsed=(time.perf_counter()-t)*1000;db.close()
    return items,elapsed

def metrics(golds,ranked,k=5):
    relevant=set(golds); ranked=ranked[:k]
    hits=[i+1 for i,x in enumerate(ranked) if x in relevant]
    # Document-level binary relevance, with duplicate chunks deduped by caller.
    recall=len(set(ranked)&relevant)/len(relevant) if relevant else None
    rr=1/hits[0] if hits else 0
    dcg=sum(1/math.log2(i+2) for i,x in enumerate(ranked) if x in relevant)
    ideal=sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))
    return {'recall':recall,'mrr':rr,'ndcg':dcg/ideal if ideal else None}

def evaluate(dbpath,queries,mode='bm25',k=5,candidates=20,model='sentence-transformers/all-MiniLM-L6-v2',local_only=False):
    data=[json.loads(s) for s in Path(queries).read_text().splitlines() if s.strip()]
    rows=[]
    for case in data:
        found,latency=search(dbpath,case['query'],mode,k,candidates,model,local_only)
        # Rank unique source documents: multiple chunks cannot inflate doc recall.
        doc_ids=list(dict.fromkeys(it['path'] for it in found))
        result=metrics(case['relevant_paths'],doc_ids,k)
        rows.append({'id':case['id'],'query':case['query'],'category':case.get('category',''), 'gold':case['relevant_paths'],'ranked':doc_ids,'latency_ms':round(latency,3),**result})
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
    i=sub.add_parser('index');i.add_argument('root');i.add_argument('--db',default='results/search.sqlite');i.add_argument('--words',type=int,default=140);i.add_argument('--overlap',type=int,default=30)
    s=sub.add_parser('search');s.add_argument('query');s.add_argument('--db',default='results/search.sqlite');s.add_argument('--mode',choices=['bm25','dense','hybrid'],default='bm25');s.add_argument('-k',type=int,default=5);s.add_argument('--local-only',action='store_true')
    e=sub.add_parser('eval');e.add_argument('--queries',default='data/queries.jsonl');e.add_argument('--db',default='results/search.sqlite');e.add_argument('--mode',choices=['bm25','dense','hybrid'],default='bm25');e.add_argument('-k',type=int,default=5);e.add_argument('--out',default='results/evaluation.json');e.add_argument('--local-only',action='store_true')
    args=ap.parse_args()
    if args.cmd=='index': print(json.dumps(index(args.root,args.db,args.words,args.overlap),indent=2))
    elif args.cmd=='search':
        items,latency=search(args.db,args.query,args.mode,args.k,local_only=args.local_only)
        print(json.dumps({'latency_ms':latency,'results':[dict(id=x['id'],source=x['path'],page=x['page'],score=x['score'],snippet=x['text'][:450]) for x in items]},indent=2))
    elif args.cmd=='eval':
        res=evaluate(args.db,args.queries,args.mode,args.k,local_only=args.local_only)
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        valid=[r for r in res if r['recall'] is not None]
        summary={metric:round(sum(r[metric] for r in valid)/len(valid),4) for metric in ['recall','mrr','ndcg']} if valid else {}
        summary.update({'queries':len(res),'mode':args.mode,'k':args.k,'mean_latency_ms':round(sum(r['latency_ms'] for r in res)/len(res),3) if res else None})
        Path(args.out).write_text(json.dumps({'summary':summary,'queries':res},indent=2))
        print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
