import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import torch
from model import Config, TinyNeoX

def device_for(name):
    if name != 'auto':
        return name
    return ('cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--text', default='demo.txt')
    p.add_argument('--out', default='runs/demo')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--block', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda','mps'])
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if a.steps < 0 or a.batch < 1 or not 2 <= a.block <= 256 or a.threads < 1:
        p.error('Invalid steps, batch, block or threads')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = device_for(a.device)
    raw = Path(a.text).read_bytes()
    cut = int(len(raw) * 0.9)
    train_ids = torch.tensor(list(raw[:cut]), dtype=torch.long)
    val_ids = torch.tensor(list(raw[cut:]), dtype=torch.long)
    if min(len(train_ids), len(val_ids)) <= a.block:
        p.error('Each split needs more bytes than --block')
    c = Config()
    model = TinyNeoX(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    def loss(x, y):
        logits, _ = model(x)
        return torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    @torch.no_grad()
    def evaluate():
        model.eval()
        total, count = 0.0, 0
        # Non-overlapping target blocks; their input boundary byte is shared.
        for start in range(0, len(val_ids) - a.block, a.block):
            window = val_ids[start:start+a.block+1].to(device)
            total += loss(window[:-1][None], window[1:][None]).item() * a.block
            count += a.block
        return total / count
    initial = evaluate()
    print('device:', device, 'parameters:', sum(x.numel() for x in model.parameters()))
    print('initial validation byte NLL:', initial)
    model.train()
    for step in range(a.steps):
        starts = torch.randint(0, len(train_ids) - a.block, (a.batch,))
        batch = torch.stack([train_ids[i:i+a.block+1] for i in starts]).to(device)
        optimizer.zero_grad(set_to_none=True)
        value = loss(batch[:, :-1], batch[:, 1:])
        if not torch.isfinite(value):
            raise RuntimeError('Nonfinite loss')
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step+1) % 25 == 0:
            print('step:', step+1, 'train byte NLL:', value.item())
    final = evaluate()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'config': asdict(c), 'state': {k:v.detach().cpu() for k,v in
               model.state_dict().items()}}, out / 'model.pt')
    metrics = dict(initial_byte_nll=initial, final_byte_nll=final,
                   data_sha256=hashlib.sha256(raw).hexdigest(), arguments=vars(a),
                   torch_version=str(torch.__version__), python=platform.python_version())
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('final validation byte NLL:', final)
    print('saved:', out/'model.pt')
    print('The bundled demo is synthetic and repetitive; its loss is not a benchmark.')

if __name__ == '__main__':
    main()
