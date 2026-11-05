"""Explicit request-local cache, bounded generation, and byte-safe streaming."""
import codecs
import math
import torch
from model import cache_length,cache_bytes
EOS=256

def encode(text):return list(text.encode('utf-8'))
def decode(tokens):return bytes(t for t in tokens if t!=EOS).decode('utf-8',errors='replace')

class ByteStream:
    def __init__(self):self.decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
    def push(self,token):return '' if token==EOS else self.decoder.decode(bytes([token]),final=False)
    def finish(self):return self.decoder.decode(b'',final=True)

def choose(logits,temperature=0.0,top_k=None,generator=None):
    if not math.isfinite(temperature) or temperature<0:raise ValueError('Invalid temperature')
    if top_k is not None and not 1<=top_k<=logits.shape[-1]:raise ValueError('Invalid top_k')
    if temperature==0:return logits.argmax(-1,keepdim=True)
    scores=logits.float()/temperature
    if top_k is not None:
        indices=scores.argsort(dim=-1,descending=True,stable=True)[...,:top_k]
        filtered=torch.full_like(scores,float('-inf'));filtered.scatter_(-1,indices,scores.gather(-1,indices));scores=filtered
    # CPU RNG keeps the sampling implementation identical across device routes.
    # Host transfer is deliberate and included in benchmark scope.
    probabilities=scores.softmax(-1).cpu()
    return torch.multinomial(probabilities,1,generator=generator).to(logits.device)

@torch.inference_mode()
def generate(model,prompt,max_new_tokens=32,cached=True,temperature=0.0,
             top_k=None,eos_id=EOS,seed=42,on_token=None):
    if model.training:raise ValueError('Call model.eval() before generation')
    if not prompt or max_new_tokens<0:raise ValueError('Nonempty prompt and nonnegative budget required')
    if len(prompt)+max_new_tokens>model.config.context:raise ValueError('Prompt plus budget exceeds context policy')
    if any(not 0<=x<model.config.vocab_size for x in prompt):raise ValueError('Invalid prompt token')
    if not math.isfinite(temperature) or temperature<0:raise ValueError('Invalid temperature')
    if top_k is not None and not 1<=top_k<=model.config.vocab_size:raise ValueError('Invalid top_k')
    device=next(model.parameters()).device
    ids=torch.tensor([prompt],dtype=torch.long,device=device)
    past=None;trace=[];rng=torch.Generator(device='cpu').manual_seed(seed)
    for step in range(max_new_tokens):
        incoming=ids if not cached or past is None else ids[:,-1:]
        before=cache_length(past)
        logits,past=model(incoming,past if cached else None,use_cache=cached,last_only=True)
        token=choose(logits[:,-1,:],temperature,top_k,rng)
        ids=torch.cat((ids,token),dim=1)
        value=token.item()
        trace.append(dict(step=step,input_tokens=incoming.shape[1],cache_before=before,
                          cache_after=cache_length(past),cache_bytes=cache_bytes(past),
                          output_length=ids.shape[1],token=value))
        if on_token is not None:on_token(value)
        if value==eos_id:break
    return ids[0].tolist(),trace
