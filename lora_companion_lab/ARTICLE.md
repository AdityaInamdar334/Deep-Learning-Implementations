# Your Model Has 124 Million Parameters. You Don’t Have to Train All of Them.
## Build LoRA from scratch, understand its gradients, and fine-tune GPT-2 with a small, inspectable adapter

You load a pretrained language model. It already knows how sentences work. It recognizes names, follows familiar patterns, and can continue a paragraph without being taught English again.

Now you want it to write dialogue in a different style.

The obvious approach is to update every parameter. But that means allocating gradients and optimizer state for millions of weights, producing another large checkpoint, and repeating that process for every specialization.

What if we kept the original model intact and learned only a small correction?

That is the practical idea behind Low-Rank Adaptation, or LoRA. The interesting part is how little machinery it requires. Two trainable matrices can change the behavior of a much larger frozen projection. Understanding why this works also reveals why some implementations silently fail.

In this tutorial, we will implement the mechanism ourselves, test its gradients, and build a GPT-2 training experiment. You will be able to inspect exactly which parameters change, compare held-out loss before and after training, save only the adapter, reload it, and verify that merging preserves its output.

This is a learning experiment, not a claim that a Shakespeare adapter creates a production assistant. No training scores or speedups below are presented as measured results.

## Contents

1. The correction is the thing we train
2. Following the tensors through LoRA
3. Why one matrix starts at zero
4. A safe implementation from scratch
5. Tests that catch real mistakes
6. Embedding adapters and tied weights
7. Building the GPT-2 experiment
8. Training and evaluating
9. Saving, reloading, and merging
10. Experiments worth running
11. Memory, speed, and deployment tradeoffs
12. Debugging checklist
13. References

## 1. The correction is the thing we train

A standard linear layer transforms its input using a weight matrix and, optionally, a bias. Full fine-tuning changes that matrix directly.

LoRA keeps the pretrained weight frozen. Alongside the original computation, it adds a trainable branch: first project the input into a small intermediate space, then project it back into the output space. The branch’s result is scaled and added to the original output. This is the central construction introduced by Hu and colleagues. [1]

The intermediate width is called the rank. It determines an upper bound on the rank of the learned weight correction. It does not determine the rank of the complete adapted weight, which still includes the original matrix.

Here is a concrete example, calculated directly from matrix dimensions:

Consider a projection that maps 768 input features to 768 output features. Its full weight contains 589,824 parameters. With rank eight, the two adapter factors together contain only 12,288 parameters, a 48-fold reduction in trainable projection parameters.

For a wider projection from 768 to 2,304 features, the full weight contains 1,769,472 parameters. A rank-eight adapter contains 24,576, making the trainable correction 72 times smaller by parameter count.

The difference becomes larger at higher widths. A 4,096-to-4,096 projection contains 16,777,216 weight parameters. With rank 16, its adapter contains 131,072 parameters, a 128-fold reduction.

Biases are excluded. These are parameter counts, not memory or runtime benchmarks.

The factorization constrains each adapted projection’s update. Nevertheless, adapters at multiple depths interact through attention, nonlinearities, and residual connections. A low-rank update at each layer does not make the entire model a single low-rank function.

## 2. Following the tensors through LoRA

Suppose a batch contains two sequences, each with 128 tokens. Each token has 768 features. We adapt a projection with 2,304 output features using rank eight.

Start with the input tensor: its shape is **2 × 128 × 768**. The first two dimensions identify sequences and token positions. The last dimension holds each token’s features.

The frozen projection stores a **2,304 × 768** weight. Applying it produces the original output, shaped **2 × 128 × 2,304**.

Now follow the adapter branch:

1. **Reduce the features.** Matrix A has shape **8 × 768**. Applying it transforms each token’s 768 features into eight adapter features. The intermediate tensor has shape **2 × 128 × 8**.
2. **Expand the correction.** Matrix B has shape **2,304 × 8**. It maps those eight features into a correction with shape **2 × 128 × 2,304**.
3. **Scale and add.** Multiply that correction by the adapter’s scale and add it to the frozen projection’s output. The final shape remains **2 × 128 × 2,304**.

The sequence length and batch size never change. The narrow intermediate space exists only inside the adapter branch.

PyTorch’s `linear` operation stores output features first. It applies the transpose of its stored weight to the input. Two consecutive calls therefore implement the low-rank branch without constructing a full correction matrix on every forward pass.

For ordinary LoRA, the multiplier is alpha divided by rank. Alpha controls the scale of the branch; rank controls its capacity and parameter count. Changing rank while fixing alpha also changes this multiplier. For a clean experiment, record both values and explain which one you hold constant.

## 3. Why one matrix starts at zero

Initialize A randomly and B to zero. The adapter initially contributes exactly zero, so adding it preserves the pretrained function in evaluation mode.

There is a useful asymmetry in the first backward pass. B generally receives a nonzero gradient because A already produces nonzero intermediate features. A initially receives zero gradient because the route from the loss back through B contains a zero matrix. Once B changes, A can learn too.

If both matrices start at zero, both gradient routes are blocked. The adapter cannot escape that initialization through ordinary gradient descent.

Freezing parameters is different from disabling autograd. Frozen layers may still need to propagate gradients to adapters earlier in the network. Wrapping the whole model in `no_grad()` during training breaks learning. Use that context for evaluation and merging, not the training forward pass.

## 4. A safe implementation from scratch

The labml example in the prompt allocates its base weights with `torch.empty`. Those weights must be populated from a checkpoint before inference. Its layer definition is not a standalone pretrained model. [2]

Our implementation instead wraps an existing projection, preserving its loaded weight, bias, device, and dtype. Save the following as `lora.py`.

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

class LoRAProjection(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.0,
                 transposed=False):
        super().__init__()
        if rank < 1:
            raise ValueError('rank must be positive')
        self.base = base
        self.base.requires_grad_(False)
        self.transposed = transposed
        if transposed:
            in_features, out_features = base.weight.shape
        else:
            out_features, in_features = base.weight.shape
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(base.weight.new_empty(rank, in_features))
        self.B = nn.Parameter(base.weight.new_zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        correction = F.linear(F.linear(self.dropout(x), self.A), self.B)
        return self.base(x) + correction * self.scale

    @torch.no_grad()
    def merged_copy(self):
        import copy
        if self.training:
            raise RuntimeError('Call eval() before merging')
        result = copy.deepcopy(self.base)
        delta = (self.B @ self.A) * self.scale
        if self.transposed:
            delta = delta.T
        result.weight.add_(delta)
        return result
```

Dropout affects only the adapter input. The base projection receives the original input. During evaluation, dropout is disabled, making the adapter a fixed linear correction that can be folded into the base weight.

The `transposed` flag handles an important GPT-2 detail: its Transformers `Conv1D` projections store weights with input features first. Despite the class name, these are the dense projections we want to adapt. PEFT exposes the corresponding layout choice through `fan_in_fan_out`. [3]

## 5. Tests that catch real mistakes

Before training a language model, test the mechanism on a small linear layer. Save this as `test_lora.py` and run `python test_lora.py`.

```python
import torch
from torch import nn
from lora import LoRAProjection

torch.manual_seed(7)
base = nn.Linear(12, 9)
layer = LoRAProjection(base, rank=3, alpha=6)
x = torch.randn(4, 5, 12)
original = base.weight.detach().clone()

layer.eval()
torch.testing.assert_close(layer(x), base(x))
layer.train()
layer(x).square().mean().backward()
assert base.weight.grad is None
assert layer.A.grad is not None
assert torch.count_nonzero(layer.A.grad).item() == 0
assert layer.B.grad.abs().sum().item() > 0

optimizer = torch.optim.AdamW([layer.A, layer.B], lr=0.01,
                              weight_decay=0.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
layer(x).square().mean().backward()
assert layer.A.grad.abs().sum().item() > 0
assert torch.equal(original, base.weight)

layer.eval()
merged = layer.merged_copy()
torch.testing.assert_close(layer(x), merged(x), rtol=1e-5, atol=1e-6)
print('Initialization, gradients, freezing, and merge checks passed.')
```

These checks answer four separate questions. Does injection preserve the original function? Does the first gradient reach the correct matrix? Does learning eventually reach both factors? Does merging implement the same computation?

They do not establish that a particular dataset, learning rate, or rank produces useful adaptation. That requires an experiment.

## 6. Embedding adapters and tied weights

An embedding layer looks up rows instead of multiplying a dense one-hot vector. We can exploit that structure: look up a small trainable vector for each token, project it to the embedding dimension, and add it to the frozen embedding.

```python
class LoRAEmbedding(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if rank < 1:
            raise ValueError('rank must be positive')
        if base.padding_idx is not None or base.max_norm is not None:
            raise ValueError('This teaching layer excludes padding/max_norm')
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.codes = nn.Parameter(
            base.weight.new_empty(base.num_embeddings, rank))
        self.up = nn.Parameter(
            base.weight.new_zeros(base.embedding_dim, rank))
        nn.init.normal_(self.codes)

    def forward(self, token_ids):
        low = F.embedding(token_ids, self.codes)
        return self.base(token_ids) + F.linear(low, self.up) * self.scale
```

Add this to `lora.py` if you want to explore embeddings. The main experiment deliberately adapts attention projections only.

Vocabulary size matters: the token-code table contains vocabulary size times rank parameters. A small rank does not automatically make an embedding adapter negligible.

GPT-2 also normally shares its input embedding weight with its output language-model head. An extra input-only branch does not automatically adapt the output head. Worse, merging into a shared weight can change both uses. Treat tied embeddings as a separate design decision and test both input and output behavior before supporting their merge.

## 7. Building the GPT-2 experiment

The downloadable companion lab includes the Python files, requirements, a dataset downloader, and additional tests. Extract `LoRA_Companion_Lab.zip` and follow its `README.md` to run the packaged version. The code below also lets you build the experiment directly from the article.

Create a directory with `lora.py`, `test_lora.py`, and the `train.py` below. Use Python 3.10 or newer in a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers
pip freeze > environment-lock.txt
python test_lora.py
```

Install the appropriate PyTorch build for your hardware. The floating installation command is a starting point, not a reproducibility lock. Keep the generated environment file with your results.

Download Tiny Shakespeare from the educational dataset maintained in Karpathy’s char-rnn repository. [5]

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt
```

Our split uses the first 90 percent of characters for training and the remainder for validation. It splits text before tokenization or window construction so a training window cannot cross into validation. This is a convenient chronological holdout, not a rigorous play-level generalization benchmark.

We use fixed-length blocks without padding. GPT-2 computes the next-token shift internally when supplied with labels; we pass a copy of the same token sequence rather than shifting twice. [4]

## 8. Training and evaluating

Save the following as `train.py`. It uses custom adapters, not PEFT, so the implementation remains inspectable.

```python
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
```

Run a brief smoke experiment first:

```bash
python train.py --steps 2 --out smoke_adapter
python train.py --steps 300 --out shakespeare_adapter
python train.py --load shakespeare_adapter --steps 0 --out reloaded_adapter
```

The first invocation checks the end-to-end plumbing. The second performs adaptation. The third reconstructs the exact base revision and restores the saved adapter. Its initial validation loss should match the previous final validation loss within numerical tolerance when data, environment, and evaluation settings match.

This script uses full precision for clarity. It selects CUDA, then Apple Silicon MPS, then CPU. CPU is useful for debugging but can be slow. Hardware support and available memory must be verified locally. Reduce batch size and block length if necessary.

For GPT-2 small, targeting the fused attention projection in all 12 layers with rank eight gives 294,912 trainable adapter parameters. That is a calculation from its projection dimensions. The script prints the actual count as a check.

The fused projection produces queries, keys, and values together. Our wrapper adapts all three through a shared rank bottleneck. This differs from adapting query and value separately; do not label the experiment “Q/V only.”

## 9. Saving, reloading, and merging

An adapter checkpoint is incomplete without its base model. The script saves factor tensors, rank, alpha, target paths, and the resolved base revision. The metrics file records the data hash and run arguments.

It deliberately excludes optimizer state. Loading it is suitable for evaluation or starting another optimization run, not exact training resumption. Exact resumption also needs optimizer and scheduler state, random-number-generator states, and data-order progress.

The merge operation returns a copy of each base projection with the correction folded into its weight. It never repeatedly adds the correction to the original object. That avoids accidental double merging.

The model-level comparison checks logits on one prompt. Extend it to multiple prompts and sequence lengths before deployment. Reduced precision can introduce larger numerical differences, so inspect errors before loosening tolerances.

The example does not save the merged full model. If you need that artifact, call `model.save_pretrained(...)` after the merge loop and save the tokenizer alongside it. Expect a full model checkpoint rather than an adapter-sized file.

## 10. Experiments worth running

A useful LoRA experiment needs more than a convincing generated paragraph. Keep a fixed held-out split and evaluate the untouched base first.

**Start with a rank sweep.** Try ranks 2, 4, 8, and 16 while keeping the data, training steps, seed, and target modules fixed. Document your scaling policy so changes in capacity are not confused with changes in the adapter multiplier. The question is whether additional adapter capacity improves held-out performance.

**Then vary alpha.** Hold rank, learning rate, and target modules fixed. This tests whether the learned correction is too weak or too aggressive at your chosen scale.

**Compare target modules.** Extend the injection logic to compare attention-only adaptation against attention plus MLP adaptation. Keep the training-token budget fixed, and record the different trainable parameter counts. This asks where adaptation is most useful, although it also changes the parameter budget.

**Sweep the learning rate.** Keep rank, alpha, and data fixed. A poor optimization setting can make an otherwise suitable adapter configuration look ineffective.

**Repeat across seeds.** Change initialization and data-order seeds while preserving the remaining settings. Several runs help distinguish a consistent improvement from a lucky run.

Report validation loss, perplexity, trainable parameters, checkpoint bytes, elapsed time, and peak memory. For CUDA timing, synchronize before and after the measured interval because kernels run asynchronously. Report the hardware and precision with every performance claim.

Fixed steps and fixed runtime answer different questions. A larger configuration may see the same tokens but take longer. A fixed runtime budget may process fewer tokens. State which comparison you intend.

Tiny Shakespeare is also easy to overfit. Inspect repeated phrases and memorize-sensitive continuations. For stronger research conclusions, split by document or play and evaluate on additional domains to measure changes in general behavior.

## 11. Memory, speed, and deployment tradeoffs

LoRA removes base-weight gradients and optimizer state when the base is frozen. It does not remove the base weights from memory. Activations and backward propagation can still be substantial.

For an illustrative FP32 Adam setup, a trainable parameter may require four bytes each for its weight, gradient, first moment, and second moment. Real peak memory also includes temporary buffers, activations, implementation details, and possibly additional precision copies. A parameter-count ratio is therefore not a peak-memory ratio.

Unmerged adapters add computation. Merging a conventional linear LoRA branch can eliminate that separate branch at inference, provided its behavior is deterministic and the target weight can be represented appropriately. Keeping adapters separate is useful when serving several specializations from one base. [6]

QLoRA addresses another cost: storage of the frozen base through quantization, while training adapters through it. It is related to LoRA but introduces separate numerical and implementation concerns. The original QLoRA work combines four-bit base quantization with additional memory techniques. This tutorial does not implement QLoRA. [7]

For production projects, PEFT provides maintained adapter integration and serialization. A comparable configuration for this experiment uses `target_modules=['c_attn']`, `r=8`, `lora_alpha=16`, `bias='none'`, and `fan_in_fan_out=True`. Check the documentation for your installed release. [3]

## 12. Debugging checklist

**The adapter changes outputs before training.** Check initialization, checkpoint loading, and dropout. Compare the original and adapted model in evaluation mode so stochastic behavior does not obscure the result.

**Neither factor learns.** Inspect both gradients and the optimizer’s parameter list. Initializing both factors to zero, detaching the computation graph, or omitting the factors from the optimizer can prevent learning.

**A has zero gradient on the first step.** With zero-initialized B, this is expected. Inspect A again after B has received an update.

**Base weights change during training.** Verify that freezing happened before optimizer construction and that no merge operation ran unexpectedly. Compare weight snapshots before and after a training step.

**GPT-2 produces a shape error.** Inspect the projection’s stored weight dimensions. Its input-first layout requires the correct orientation when constructing and merging the adapter.

**Training loss falls while validation loss rises.** Check for overfitting and differences between your splits. Shorten training and inspect held-out examples before increasing adapter capacity.

**Reloading changes behavior.** Check the base revision, rank, alpha, target paths, and evaluation settings. Compare logits on a fixed input as well as aggregate loss.

**Merging changes outputs substantially.** Check for a wrong transpose, active dropout, or a correction added twice. Debug parity on a single layer before comparing the whole model.

**Memory savings are smaller than expected.** Profile actual allocation. Frozen weights and activations can dominate even when the trainable parameter count is small.

The real value of implementing LoRA yourself is that the abstraction becomes inspectable. You can identify the tensors being trained, explain the first gradient step, count the adapter parameters, and prove that a merged projection computes the same function.

That makes the next paper easier to read too. When a method changes initialization, scaling, factorization, or target selection, you now know exactly which part of the system it is changing and which experiment would test the claim.

## 13. References

[1] Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*. https://arxiv.org/abs/2106.09685

[2] labml.ai, *Low-Rank Adaptation implementation*. Inspiration and the source of the supplied annotated excerpt. https://nn.labml.ai/lora/index.html

[3] Hugging Face, *PEFT LoRA configuration reference*. https://huggingface.co/docs/peft/main/package_reference/lora

[4] Hugging Face, *GPT-2 model documentation*. https://huggingface.co/docs/transformers/model_doc/gpt2

[5] Andrej Karpathy, *char-rnn Tiny Shakespeare dataset*. https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare

[6] Microsoft, *Official LoRA implementation*. https://github.com/microsoft/LoRA

[7] Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*. https://arxiv.org/abs/2305.14314

Suggested Medium topics: Machine Learning, Deep Learning, Artificial Intelligence, Python, Large Language Models.
