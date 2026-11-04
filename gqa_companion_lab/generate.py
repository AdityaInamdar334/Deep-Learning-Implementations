import argparse
import torch
from checkpoints import load_model
from train import device_for

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--prompt',default='The researcher ')
    p.add_argument('--tokens',type=int,default=60)
    p.add_argument('--temperature',type=float,default=0.0)
    p.add_argument('--no-cache',action='store_true')
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    a=p.parse_args()
    if not a.prompt:p.error('Prompt must be nonempty')
    torch.set_num_threads(2)
    device=device_for(a.device)
    model=load_model(a.checkpoint,device).eval()
    ids=torch.tensor([list(a.prompt.encode())],device=device)
    out=model.generate(ids,a.tokens,cached=not a.no_cache,temperature=a.temperature)
    print(bytes(out[0].cpu().tolist()).decode('utf-8',errors='replace'))
if __name__=='__main__':main()
