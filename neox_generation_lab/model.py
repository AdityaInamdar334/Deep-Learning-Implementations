"""Small NeoX-style decoder, NOT a loader for GPT-NeoX-20B weights."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size:int=257
    width:int=96
    heads:int=4
    layers:int=2
    context:int=256
    rotary_fraction:float=.25
    def __post_init__(self):
        if min(self.vocab_size,self.width,self.heads,self.layers,self.context)<1 or self.width%self.heads:
            raise ValueError('Invalid dimensions')
        rd=int(self.width//self.heads*self.rotary_fraction)
        if not 0<self.rotary_fraction<=1 or rd<2 or rd%2:raise ValueError('Invalid rotary dimension')

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__();self.h=c.heads;self.d=c.width//c.heads
        self.rd=int(self.d*c.rotary_fraction)
        self.qkv=nn.Linear(c.width,3*c.width);self.out=nn.Linear(c.width,c.width)
    def rotate(self,x,offset):
        pos=torch.arange(offset,offset+x.shape[-2],device=x.device,dtype=torch.float32)
        inv=10000.0**(-torch.arange(0,self.rd,2,device=x.device,dtype=torch.float32)/self.rd)
        angles=pos[:,None]*inv[None,:];angles=torch.cat((angles,angles),-1)
        a,rest=x[...,:self.rd],x[...,self.rd:];lo,hi=a.chunk(2,-1)
        a=a*angles.cos().to(x.dtype)+torch.cat((-hi,lo),-1)*angles.sin().to(x.dtype)
        return torch.cat((a,rest),-1)
    def forward(self,x,past):
        b,t,w=x.shape
        q,k,v=self.qkv(x).view(b,t,self.h,3*self.d).chunk(3,-1)
        q,k,v=[z.transpose(1,2) for z in (q,k,v)]
        offset=0 if past is None else past[0].shape[-2]
        q,k=self.rotate(q,offset),self.rotate(k,offset)
        if past is not None:k=torch.cat((past[0],k),-2);v=torch.cat((past[1],v),-2)
        query=offset+torch.arange(t,device=x.device)
        key=torch.arange(k.shape[-2],device=x.device)
        allowed=key[None,:]<=query[:,None]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0)
        return self.out(y.transpose(1,2).reshape(b,t,w)),(k,v)

class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.n1=nn.LayerNorm(c.width);self.n2=nn.LayerNorm(c.width)
        self.attn=Attention(c)
        self.mlp=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),nn.Linear(4*c.width,c.width))
    def forward(self,x,past):
        a,kv=self.attn(self.n1(x),past)
        return x+a+self.mlp(self.n2(x)),kv

class TinyNeoX(nn.Module):
    def __init__(self,c=Config()):
        super().__init__();self.config=c
        self.embedding=nn.Embedding(c.vocab_size,c.width)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm=nn.LayerNorm(c.width);self.head=nn.Linear(c.width,c.vocab_size,bias=False)
    def forward(self,ids,past=None,use_cache=False,last_only=False):
        if ids.ndim!=2 or ids.shape[1]<1:raise ValueError('Need [batch, nonempty sequence] IDs')
        offset=cache_length(past)
        if offset+ids.shape[1]>self.config.context:raise ValueError('Context limit exceeded')
        if past is not None:
            if len(past)!=self.config.layers:raise ValueError('Wrong number of cached layers')
            for k,v in past:
                if k.shape!=v.shape or k.shape!=(ids.shape[0],self.config.heads,offset,self.config.width//self.config.heads):
                    raise ValueError('Cache shape mismatch')
        x=self.embedding(ids);cache=[]
        for i,block in enumerate(self.blocks):
            x,kv=block(x,None if past is None else past[i]);cache.append(kv)
        if last_only:x=x[:,-1:,:]
        return self.head(self.norm(x)),cache if use_cache else None

def cache_length(cache):return 0 if cache is None else cache[0][0].shape[-2]
def cache_bytes(cache):return sum(t.numel()*t.element_size() for kv in (cache or []) for t in kv)
def save(model,path):torch.save({'config':asdict(model.config),'state':model.state_dict()},path)
def load(path,device='cpu'):
    data=torch.load(path,map_location='cpu',weights_only=True);m=TinyNeoX(Config(**data['config']))
    m.load_state_dict(data['state']);return m.to(device).eval()
