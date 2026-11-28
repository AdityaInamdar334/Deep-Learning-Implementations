import torch
from torch import nn

class SparseMoE(nn.Module):
    def __init__(self,d,experts=4,k=2):
        super().__init__(); self.k=k; self.router=nn.Linear(d,experts); self.experts=nn.ModuleList([nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d)) for _ in range(experts)])
        if not 1<=k<=experts: raise ValueError('invalid top-k')
    def forward(self,x):
        shape=x.shape; x=x.reshape(-1,shape[-1]); logits=self.router(x); scores,ids=logits.topk(self.k,dim=-1); weights=scores.softmax(-1); out=torch.zeros_like(x)
        for i,e in enumerate(self.experts):
            rows,slots=torch.where(ids==i)
            if rows.numel(): out=out.index_add(0,rows,e(x[rows])*weights[rows,slots,None])
        self.counts=torch.bincount(ids.flatten(),minlength=len(self.experts)).detach()
        self.aux=len(self.experts)*(logits.softmax(-1).mean(0)*self.counts.to(x.dtype)/(len(x)*self.k)).sum()
        return out.reshape(shape)

class Block(nn.Module):
    def __init__(self,d,moe=False):
        super().__init__(); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,4,batch_first=True); self.ff=SparseMoE(d) if moe else nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(self,x):
        q=self.n1(x); x=x+self.attn(q,q,q,need_weights=False)[0]; return x+self.ff(self.n2(x))

class Model(nn.Module):
    def __init__(self,n=8,d=32,depth=2,shared=False,moe=False,gated=False):
        super().__init__(); self.shared=shared; self.depth=depth; self.n=n; self.gated=gated
        self.node=nn.Embedding(n,d); self.hops=nn.Embedding(16,d); self.query=nn.Parameter(torch.zeros(d)); self.blocks=nn.ModuleList([Block(d,moe) for _ in range(depth)]); self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,n)
        if gated: self.gate=nn.Parameter(torch.tensor(-2.))
    def forward(self,edges,start,hops,loops=1,trace=False):
        if loops<1: raise ValueError('loops must be positive')
        b,n=edges.shape; nodes=torch.arange(n,device=edges.device).expand(b,-1)
        x=self.node(nodes)+self.node(edges); q=self.node(start)+self.hops(hops)+self.query; x=torch.cat([x,q[:,None]],1); initial=x; states=[]; aux=[]; counts=[]
        for t in range(loops if self.shared else 1):
            if self.gated and t: x=x+self.gate.sigmoid()*initial
            for block in self.blocks:
                x=block(x)
                if isinstance(block.ff,SparseMoE): aux.append(block.ff.aux); counts.append(block.ff.counts)
            if trace: states.append(x.detach())
        self.aux=torch.stack(aux).mean() if aux else x.new_zeros(()); self.counts=torch.stack(counts).sum(0) if counts else None
        logits=self.head(self.norm(x[:,-1])); return (logits,states) if trace else logits
