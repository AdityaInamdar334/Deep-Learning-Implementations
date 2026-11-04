import argparse
import torch
from model import Config, TinyNeoX
from train import device_for

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='runs/demo/model.pt')
    p.add_argument('--prompt', default='The researcher ')
    p.add_argument('--tokens', type=int, default=80)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda','mps'])
    a=p.parse_args()
    torch.set_num_threads(2)
    device=device_for(a.device)
    checkpoint=torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model=TinyNeoX(Config(**checkpoint['config']))
    model.load_state_dict(checkpoint['state'], strict=True)
    model.to(device).eval()
    if not a.prompt:
        p.error('Provide a nonempty prompt')
    ids=torch.tensor([list(a.prompt.encode('utf-8'))], device=device)
    result=model.generate(ids, a.tokens, cached=not a.no_cache,
                          temperature=a.temperature)
    print(bytes(result[0].cpu().tolist()).decode('utf-8', errors='replace'))

if __name__ == '__main__':
    main()
