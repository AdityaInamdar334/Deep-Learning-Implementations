import argparse, hashlib, json, platform, time
from pathlib import Path
import torch
from model import Config,TinyGQA
from checkpoints import load_model,save_model

def device_for(name):
    return name if name!='auto' else ('cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu')

def matched_initialization(model,seed):
    # Same-name, same-shape parameters are identical across head-count variants.
    with torch.no_grad():
        for name,param in model.named_parameters():
            if param.ndim==2:
                digest=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')
                generator=torch.Generator().manual_seed((seed+digest)%(2**63-1))
                param.copy_(torch.randn(param.shape,generator=generator)*0.02)
            elif name.endswith('.bias'): param.zero_()
            else: param.fill_(1.0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--text',default='demo.txt')
    p.add_argument('--kv-heads',type=int,default=2)
    p.add_argument('--backend',choices=['repeat','grouped','native'],default='repeat')
    p.add_argument('--init',default=None,help='Load a lab checkpoint; its head count takes precedence')
    p.add_argument('--out',default='runs/gqa')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--block',type=int,default=64)
    p.add_argument('--eval-blocks',type=int,default=64)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data-seed',type=int,default=123)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    a=p.parse_args()
    if min(a.batch,a.eval_blocks,a.threads)<1 or a.steps<0 or a.block<2: p.error('Invalid run arguments')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device=device_for(a.device)
    if a.init: model=load_model(a.init,backend=a.backend)
    else:
        model=TinyGQA(Config(num_kv_heads=a.kv_heads,backend=a.backend))
        matched_initialization(model,a.seed)
    if a.block>model.config.max_seq_len:p.error('Block exceeds context limit')
    model.to(device)
    raw=Path(a.text).read_bytes()
    cut=int(len(raw)*0.9)
    train_ids=torch.tensor(list(raw[:cut]),dtype=torch.long)
    val_ids=torch.tensor(list(raw[cut:]),dtype=torch.long)
    if min(len(train_ids),len(val_ids))<=a.block:p.error('Split too short for block')
    data_rng=torch.Generator().manual_seed(a.data_seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    def loss(x,y):
        logits,_=model(x)
        return torch.nn.functional.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
    @torch.no_grad()
    def evaluate():
        model.eval()
        count=min((len(val_ids)-1)//a.block,a.eval_blocks)
        total=0.0
        for index in range(count):
            start=index*a.block
            batch=val_ids[start:start+a.block+1].to(device)
            total+=loss(batch[:-1][None],batch[1:][None]).item()
        return total/count,count*a.block
    before,evaluated=evaluate()
    print('Initial validation byte NLL:',before,'evaluated target bytes:',evaluated,flush=True)
    model.train()
    start=time.perf_counter()
    for step in range(a.steps):
        starts=torch.randint(0,len(train_ids)-a.block,(a.batch,),generator=data_rng)
        batch=torch.stack([train_ids[i:i+a.block+1] for i in starts]).to(device)
        optimizer.zero_grad(set_to_none=True)
        value=loss(batch[:,:-1],batch[:,1:])
        if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if (step+1)%25==0:print('step',step+1,'train byte NLL',value.item(),flush=True)
    if device.startswith('cuda'):torch.cuda.synchronize()
    if device=='mps':torch.mps.synchronize()
    elapsed=time.perf_counter()-start
    after,_=evaluate()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    save_model(model,out/'model.pt')
    metrics=dict(initial_byte_nll=before,final_byte_nll=after,
        evaluated_target_bytes=evaluated,training_target_bytes=a.steps*a.batch*a.block,
        num_query_heads=model.config.num_heads,num_kv_heads=model.config.num_kv_heads,
        parameters=sum(p.numel() for p in model.parameters()),training_seconds=elapsed,
        data_sha256=hashlib.sha256(raw).hexdigest(),arguments=vars(a),
        torch_version=str(torch.__version__),python=platform.python_version())
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
    print('demo.txt is synthetic and repetitive; these losses are not quality benchmarks.')
if __name__=='__main__':main()
