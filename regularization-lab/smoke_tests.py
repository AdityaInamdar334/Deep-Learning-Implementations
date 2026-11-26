import numpy as np
import torch
from torch import nn
from lab import model
w=np.array([2.,2.,.4]); h=np.array([10.,.1,1.]); a=.5
print('L2:',h/(h+a)*w)
print('L1:',np.sign(w)*np.maximum(abs(w)-a/h,0))
assert np.allclose(np.sign(w)*np.maximum(abs(w)-a/h,0),[1.95,0,0])
d=nn.Dropout(.3); x=torch.ones(10000); torch.manual_seed(42)
assert abs(d(x).mean().item()-1)<.05
d.eval(); assert torch.equal(d(x),x)
assert model()(torch.zeros(2,64)).shape==(2,10)
print('Smoke tests passed')
