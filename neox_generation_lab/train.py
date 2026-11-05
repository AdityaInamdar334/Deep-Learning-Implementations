"""Small synthetic byte-language-model exercise, not pretrained NeoX quality."""
import argparse,json,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from model import TinyNeoX,save
from generation import encode

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=200)
    p.add_argument('--text',default='demo.txt');p.add_argument('--out',default='runs/model.pt');a=p.parse_args()
    if a.steps<1:p.error('steps must be positive')
    torch.set_num_threads(2);torch.manual_seed(42)
    data=Path(a.text).read_bytes();ids=torch.tensor(list(data),dtype=torch.long)
    split=int(.9*len(ids));train,val=ids[:split],ids[split:]
    block=48
    if min(len(train),len(val))<=block:raise ValueError('Need at least 49 bytes in both splits')
    m=TinyNeoX();opt=torch.optim.AdamW(m.parameters(),lr=.001)
    rng=torch.Generator().manual_seed(9)
    def evaluate():
        m.eval();total=0.;n=0
        with torch.inference_mode():
            for start in range(0,len(val)-block,block):
                chunk=val[start:start+block+1];logits,_=m(chunk[:-1][None,:])
                total+=F.cross_entropy(logits.reshape(-1,257),chunk[1:],reduction='sum').item();n+=block
        return total/n,n
    initial,n=evaluate();m.train()
    for _ in range(a.steps):
        starts=torch.randint(len(train)-block,(8,),generator=rng)
        chunks=torch.stack([train[s:s+block+1] for s in starts])
        logits,_=m(chunks[:,:-1]);loss=F.cross_entropy(logits.reshape(-1,257),chunks[:,1:].reshape(-1))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    final,_=evaluate();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);save(m,out)
    result=dict(steps=a.steps,initial_validation_byte_nll=initial,final_validation_byte_nll=final,
                evaluated_target_bytes=n,data_sha256=hashlib.sha256(data).hexdigest(),torch_version=str(torch.__version__),
                note='Synthetic repetitive text; pipeline validation only. EOS not trained in this corpus.')
    out.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
