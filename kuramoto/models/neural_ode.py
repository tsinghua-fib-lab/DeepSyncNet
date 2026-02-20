import torch
from torch import nn
if False:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class NeuralODEfunc(nn.Module):

    def __init__(self, in_channels, feature_dim, nhidden, net='MLP'):
        super(NeuralODEfunc, self).__init__()
        self.nfe = 0
        
        if net == 'MLP':
            self.net = nn.Sequential(
                nn.Linear(in_channels*feature_dim, nhidden),
                nn.Tanh(),
                nn.Linear(nhidden, in_channels*feature_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, t, x):
        self.nfe += 1
        out = self.net(x)
        return out


class NeuralODE(nn.Module):

    def __init__(self, in_channels, feature_dim, data_dim, nhidden=64, submodel='MLP'):
        super(NeuralODE, self).__init__()

        self.submodel = submodel
        self.ode = NeuralODEfunc(in_channels, feature_dim, nhidden, submodel)

        self.flatten = nn.Flatten(start_dim=-2)
        self.unflatten = nn.Unflatten(-1, (in_channels, feature_dim))

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, data_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, data_dim, dtype=torch.float32))
    
    def forward(self, x0, t, dt=1e-3):
        
        if self.submodel=='MLP': 
            x0 = self.flatten(x0)[:,0]
            out = odeint(self.ode, x0, t, method='euler', options={'step_size': dt}).permute(1, 0, 2)
            out = self.unflatten(out)
        elif self.submodel=='CNN':
            out = odeint(self.ode, x0, t, method='euler', options={'step_size': dt}).permute(1, 0, 2, 3, 4)
            out = out[:,:,0]
        
        return out

    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-11)
    
    def descale(self, x):
        if x.shape[-1] < self.max.shape[-1]:
            obs_dim = x.shape[-1]
            return x * (self.max[:,:obs_dim]-self.min[:,:obs_dim]+1e-11) + self.min[:,:obs_dim]
        else:
            return x * (self.max-self.min+1e-11) + self.min