from dataclasses import asdict, replace
from pathlib import Path
import torch
from model import Config, TinyGQA

def save_model(model, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'config':asdict(model.config),
                'state':{k:v.detach().cpu() for k,v in model.state_dict().items()}},path)

def load_model(path, device='cpu', backend=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    settings=dict(checkpoint['config'])
    # Old NeoX lab used num_heads K/V heads and fused head-local QKV.
    settings.setdefault('num_kv_heads',settings['num_heads'])
    if backend is not None: settings['backend']=backend
    c=Config(**settings)
    state=dict(checkpoint['state'])
    for i in range(c.num_layers):
        stem=f'layers.{i}.attn.'
        if stem+'qkv.weight' in state:
            if c.num_kv_heads!=c.num_heads:
                raise ValueError('Import legacy MHA first, then convert explicitly')
            dim=c.hidden_size//c.num_heads
            weight=state.pop(stem+'qkv.weight').view(c.num_heads,3,dim,c.hidden_size)
            bias=state.pop(stem+'qkv.bias').view(c.num_heads,3,dim)
            for j,name in enumerate(('q_proj','k_proj','v_proj')):
                state[stem+name+'.weight']=weight[:,j].reshape(c.hidden_size,c.hidden_size)
                state[stem+name+'.bias']=bias[:,j].reshape(c.hidden_size)
    model=TinyGQA(c)
    model.load_state_dict(state,strict=True)
    return model.to(device)

@torch.no_grad()
def pool_mha(model, num_kv_heads):
    """Mean-pool contiguous groups of MHA K/V weights AND biases."""
    c=model.config
    if c.num_kv_heads!=c.num_heads:
        raise ValueError('Source must be MHA; repeated pooling is not supported')
    target=TinyGQA(replace(c,num_kv_heads=num_kv_heads))
    parameter=next(model.parameters())
    target.to(device=parameter.device,dtype=parameter.dtype)
    state={k:v.clone() for k,v in model.state_dict().items()}
    group_size=c.num_heads//num_kv_heads
    dim=c.hidden_size//c.num_heads
    for i in range(c.num_layers):
        for projection in ('k_proj','v_proj'):
            stem=f'layers.{i}.attn.{projection}'
            state[stem+'.weight']=state[stem+'.weight'].view(
                num_kv_heads,group_size,dim,c.hidden_size).mean(1).reshape(
                num_kv_heads*dim,c.hidden_size)
            state[stem+'.bias']=state[stem+'.bias'].view(
                num_kv_heads,group_size,dim).mean(1).reshape(num_kv_heads*dim)
    target.load_state_dict(state,strict=True)
    target.train(model.training)
    return target
