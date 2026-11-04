import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from lora import LoRAProjection

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
parser.add_argument('--text', default='input.txt')
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--rank', type=int, default=8)
parser.add_argument('--alpha', type=float, default=16)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--block', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--out', default='adapter')
parser.add_argument('--load', default=None)
args = parser.parse_args()
if args.steps < 0 or args.batch < 1 or not 2 <= args.block <= 1024:
    raise ValueError('Invalid steps, batch, or block length')
torch.manual_seed(42)
device = ('cuda' if torch.cuda.is_available() else
          'mps' if torch.backends.mps.is_available() else 'cpu')
if args.device != 'auto':
    device = args.device
meta = None
if args.load:
    meta = json.loads((Path(args.load) / 'config.json').read_text())
name = meta['base'] if meta else 'openai-community/gpt2'
revision = meta['revision'] if meta else None
tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
model = AutoModelForCausalLM.from_pretrained(name, revision=revision)
model.requires_grad_(False)
model.config.use_cache = False
rank = meta['rank'] if meta else args.rank
alpha = meta['alpha'] if meta else args.alpha
names = []
for path, module in list(model.named_modules()):
    if path.endswith('attn.c_attn'):
        parent_path, child = path.rsplit('.', 1)
        parent = model.get_submodule(parent_path)
        setattr(parent, child, LoRAProjection(
            module, rank=rank, alpha=alpha, transposed=True))
        names.append(path)
assert len(names) == model.config.n_layer
if meta and names != meta['targets']:
    raise ValueError('Adapter target mismatch')
model.to(device)
trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
assert trainable and all(n.endswith(('.A', '.B')) for n in trainable)
if meta:
    saved = torch.load(Path(args.load) / 'adapter.pt',
                       map_location=device, weights_only=True)
    if set(saved) != set(trainable):
        raise ValueError('Adapter key mismatch')
    with torch.no_grad():
        for key, parameter in trainable.items():
            if saved[key].shape != parameter.shape:
                raise ValueError(f'Shape mismatch: {key}')
            parameter.copy_(saved[key])
print('Device:', device)
print('Trainable parameters:', sum(p.numel() for p in trainable.values()))

text = Path(args.text).read_text(encoding='utf-8')
cut = int(len(text) * 0.9)
def blocks(part):
    ids = tokenizer(part, add_special_tokens=False,
                    verbose=False)['input_ids']
    count = len(ids) // args.block
    if count == 0:
        raise ValueError('Split is shorter than one block')
    return torch.tensor(ids[:count * args.block]).reshape(count, args.block)
train_loader = DataLoader(blocks(text[:cut]), batch_size=args.batch,
                          shuffle=True)
val_loader = DataLoader(blocks(text[cut:]), batch_size=args.batch)

@torch.no_grad()
def evaluate():
    model.eval()
    total, count = 0.0, 0
    for batch in val_loader:
        batch = batch.to(device)
        loss = model(input_ids=batch, labels=batch).loss
        tokens = batch.shape[0] * (batch.shape[1] - 1)
        total += loss.item() * tokens
        count += tokens
    return total / count

before = evaluate()
print('Initial validation loss:', before)
optimizer = torch.optim.AdamW(trainable.values(), lr=args.lr,
                              weight_decay=0.0)
iterator = iter(train_loader)
model.train()
for step in range(args.steps):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(train_loader)
        batch = next(iterator)
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=batch, labels=batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
    optimizer.step()
    if (step + 1) % 25 == 0:
        print(step + 1, 'training loss', loss.item())
after = evaluate()
print('Final validation loss:', after, 'perplexity:', math.exp(after))

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.save({n: p.detach().cpu() for n, p in trainable.items()},
           out / 'adapter.pt')
config = dict(base=name, revision=getattr(model.config, '_commit_hash', revision),
              rank=rank, alpha=alpha, targets=names)
if config['revision'] is None:
    raise RuntimeError('Could not resolve base revision; pin it explicitly')
(out / 'config.json').write_text(json.dumps(config, indent=2))
(out / 'metrics.json').write_text(json.dumps(dict(
    initial_loss=before, final_loss=after, arguments=vars(args),
    data_sha256=hashlib.sha256(text.encode()).hexdigest()), indent=2))

prompt = tokenizer('ROMEO:\n', return_tensors='pt').to(device)
model.eval()
with torch.no_grad():
    probe = model(**prompt).logits
    output = model.generate(**prompt, max_new_tokens=100, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0], skip_special_tokens=True))
for path in names:
    parent_path, child = path.rsplit('.', 1)
    parent = model.get_submodule(parent_path)
    setattr(parent, child, getattr(parent, child).merged_copy())
with torch.no_grad():
    merged_probe = model(**prompt).logits
torch.testing.assert_close(probe, merged_probe, rtol=1e-3, atol=1e-3)
print('Whole-model merge parity passed on the probe prompt.')
