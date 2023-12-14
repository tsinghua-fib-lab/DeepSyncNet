import torch
from torch import nn
if False:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

from .weight_init import normal_init
from .layers import EncodeLayer, DecodeLayer, AttentionEncoder


class SynergeticNet(nn.Module):
    
    def __init__(self, in_channels, feature_dim, num_heads=1):
        super(SynergeticNet, self).__init__()
        
        self.flatten = nn.Flatten(1)
        self.Qnet = nn.Sequential(
            nn.Linear(2*in_channels*feature_dim, in_channels*feature_dim, bias=True),
            nn.ReLU(),
        )
        try:
            self.attention = nn.MultiheadAttention(in_channels*feature_dim, num_heads=num_heads, dropout=0.01, batch_first=True)
        except:
            print(f"num_heads {num_heads} is not supported in this system (in_channels={in_channels}, feature_dim={feature_dim}), use num_heads=1 instead.")
            self.attention = nn.MultiheadAttention(in_channels*feature_dim, num_heads=1, dropout=0.01, batch_first=True)
        
    def forward(self, x_f, sync_u_s, decode):
        
        sync_x_s = decode(sync_u_s)[0]
            
        # Flatten
        x_f = self.flatten(x_f)
        sync_x_s = self.flatten(sync_x_s)
        
        sync_x = x_f + sync_x_s
        
        # Query
        try:
            tmp = torch.concatenate((x_f, sync_x_s), dim=-1)
        except:
            tmp = torch.cat((x_f, sync_x_s), dim=-1) # for old version of torch
        query = self.Qnet(tmp).unsqueeze(-2)
                
        # Key
        k1 = sync_x.unsqueeze(-2)
        k2 = sync_x_s.unsqueeze(-2)
        k3 = x_f.unsqueeze(-2)
        key = torch.cat((k1, k2, k3), dim=-2)
        
        # Value
        v1 = sync_x.unsqueeze(-2)
        v2 = sync_x_s.unsqueeze(-2)
        v3 = x_f.unsqueeze(-2)
        value = torch.cat((v1, v2, v3), dim=-2)
        
        # Scaled Dot-Product Attention
        fusion = self.attention(query, key, value)[0].squeeze(-2)
        
        # return fusion
        return torch.cat((fusion, sync_x), dim=-1)


class NeuralODEfunc(nn.Module):

    def __init__(self, in_channels, feature_dim, nhidden, fast, net='MLP'):
        super(NeuralODEfunc, self).__init__()
        
        if fast:
            self.fc1 = nn.Linear(3*in_channels*feature_dim, in_channels*feature_dim)
            self.fc2 = nn.Linear(in_channels*feature_dim, in_channels*feature_dim)
        else:
            self.fc1 = nn.Linear(in_channels*feature_dim, nhidden)
            self.fc2 = nn.Linear(nhidden, in_channels*feature_dim)
        
        self.tanh = nn.Tanh()
        
        self.nfe = 0
        self.net = net
        self.fast = fast
        self.background = None

    def set_background(self, background):
        self.background = background

    def forward(self, t, x):
        self.nfe += 1

        if self.fast:
            out = torch.cat((x, self.background), dim=-1)
            out = self.fc1(out)
            out = self.tanh(out)
            out = self.fc2(out)
        else:
            out = self.fc1(x)
            out = self.tanh(out)
            out = self.fc2(out)

        return out


class NeuralODE(nn.Module):

    def __init__(self, in_channels, feature_dim, nhidden=64, fast=False):
        super(NeuralODE, self).__init__()
        
        net = 'MLP'
        self.net = net

        self.ode = NeuralODEfunc(in_channels, feature_dim, nhidden, fast, net=net)

        self.flatten = nn.Flatten(start_dim=-2)
        self.unflatten = nn.Unflatten(-1, (in_channels, feature_dim))

        self.fast = fast
    
    def forward(self, x0, t, dt=1e-3, background=None):
        
        if self.fast: 
            x0 = self.flatten(x0)[:,0]
            
            self.ode.set_background(background)
        
        out = odeint(self.ode, x0, t, method='euler', options={'step_size': dt}).permute(1, 0, 2)

        if self.fast:
            out = self.unflatten(out)
        
        return out


class Slow_Fast_Synergetic_ODE(nn.Module):
    
    def __init__(self, in_channels, feature_dim, embed_dim, slow_dim, redundant_dim, tau_s, tau_1, device, data_dim, enc_net='MLP', e1_layer_n=3, sync=True, inter_p='nearest_neighbour', num_heads=1):
        super(Slow_Fast_Synergetic_ODE, self).__init__()

        self.sync = sync
        self.slow_dim = slow_dim
        self.redundant_dim = redundant_dim
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        ################
        # Decomposition
        ################
        
        # (batchsize, embed_dim)-->(batchsize, slow_dim)
        self.encoder_2 = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, slow_dim, bias=True),
        )

        # (batchsize, slow_dim)-->(batchsize, redundant_dim)
        self.encoder_3 = nn.Sequential(
            nn.Linear(slow_dim, 32, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(32, redundant_dim, bias=True),
        )

        # (batchsize, slow_dim)-->(batchsize, embed_dim)
        self.decoder_1 = nn.Sequential(
            nn.Linear(slow_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
        )

        self.unflatten = nn.Unflatten(-1, (1, in_channels, feature_dim))

        if enc_net == 'GRU':
            class EncodeCell(nn.Module):

                def __init__(self, input_dim, hidden_dim, flatten=False, dropout=False, net='MLP', act=True):
                    super(EncodeCell, self).__init__()

                    assert net in ['MLP', 'GRU1', 'GRU', 'LSTM', 'LSTM-changed'], f"Encoder Net Error, {net} not implemented!"

                    self.net = net

                    self.act = act
                    self.flatten = nn.Flatten() if flatten else None
                    self.dropout = nn.Dropout(p=0.01) if dropout else None

                    self.w1 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, hidden_dim)))
                    self.b1 = nn.Parameter(torch.zeros(hidden_dim))

                    if net != 'MLP': # GRU1/2 or LSTM
                        self.w2 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, hidden_dim)))
                        self.b2 = nn.Parameter(torch.zeros(hidden_dim))

                    if net != 'GRU1': # GRU2 or LSTM
                        self.w3 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, hidden_dim)))
                        self.b3 = nn.Parameter(torch.zeros(hidden_dim))

                    if 'LSTM' not in net: # GRU2
                        self.b4 = nn.Parameter(torch.zeros(hidden_dim))

                    self.sigmoid = nn.Sigmoid()
                    self.tanh = nn.Tanh()
                
                def forward(self, x):

                    if self.flatten:
                        x = self.flatten(x)
                    
                    if self.net == 'GRU':
                        z = self.sigmoid(x @ self.w1 + self.b1)
                        r = self.sigmoid(x @ self.w2 + self.b2)
                        h = self.tanh(x @ self.w3 + self.b3 + r * self.b4)
                        y = (1 - z) * h

                    if self.dropout:
                        y = self.dropout(y)

                    return y
            self.encoder_1 = []
            self.encoder_1.append(EncodeCell(in_channels*feature_dim, 64, True, True, enc_net))
            [self.encoder_1.append(EncodeCell(64, 64, False, True, enc_net)) for _ in range(e1_layer_n-2)]
            self.encoder_1.append(EncodeCell(64, embed_dim, False, False, enc_net, False))
            self.encoder_1 = nn.Sequential(*self.encoder_1)
            self.decoder_2 = nn.Sequential(
                nn.Linear(embed_dim, 64, bias=True),
                nn.Tanh(),
                nn.Dropout(p=0.01),
                nn.Linear(64, in_channels*feature_dim, bias=True),
                nn.Unflatten(-1, (1, in_channels, feature_dim))
            )
        
        else:
            # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)
            self.encoder_1 = []
            for i in range(e1_layer_n):
                self.encoder_1.append(EncodeLayer(
                    input_dim  = 128 if i>0 else in_channels*feature_dim, 
                    output_dim = 128 if i<e1_layer_n-1 else embed_dim, 
                    flatten    = False if i>0 else True, 
                    dropout    = True if i<e1_layer_n-1 else False, 
                    net        = enc_net, 
                    activate   = True if i<e1_layer_n-1 else False, 
                    layernorm  = True if i<e1_layer_n-1 else False,
                    in_channel = in_channels))
            self.encoder_1 = nn.Sequential(*self.encoder_1)
            
            # (batchsize, embed_dim)-->(batchsize,1,channel_num,feature_dim)
            self.decoder_2 = []
            for i in range(e1_layer_n):
                self.decoder_2.append(DecodeLayer(
                    input_dim  = 128 if i>0 else embed_dim, 
                    output_dim = 128 if i<e1_layer_n-1 else in_channels*feature_dim, 
                    flatten    = False if i>0 else True, 
                    dropout    = True if i<e1_layer_n-1 else False, 
                    net        = 'MLP',
                    activate   = True if i<e1_layer_n-1 else False, 
                    layernorm  = True if i<e1_layer_n-1 else False,
                    in_channel = in_channels))
            self.decoder_2 = nn.Sequential(*self.decoder_2)
        
        
        ################
        # SFS-ODE
        ################
        
        self.node_s = NeuralODE(1, slow_dim, fast=False)
        self.node_f = NeuralODE(in_channels, feature_dim, fast=True)
        self.sync_block = SynergeticNet(in_channels, feature_dim, num_heads)
        
        self.enc_net = enc_net
        
        self.inter_p = inter_p
        if inter_p == 'automatic':
            self.att_w = nn.Sequential(
                nn.Linear(2, 1, bias=True),
                nn.Sigmoid(),
            ) # learnable weights for interpolation

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, data_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, data_dim, dtype=torch.float32))

        # init
        self.apply(normal_init)
    
    def obs2slow(self, obs):
        # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)-->(batchsize, slow_dim)
        embed = self.encoder_1(obs)
        slow_var = self.encoder_2(embed)
        
        return slow_var, embed

    def slow2obs(self, slow_var):
        # (batchsize, slow_dim)-->(batchsize,1,channel_num,feature_dim)
        embed = self.decoder_1(slow_var)
        obs = self.decoder_2(embed.detach())

        if self.enc_net not in ['GRU', 'Attention']:
            obs = self.unflatten(obs) # TODO

        return obs, embed

    def synergetic(self, x_f, u_s, u_s_, delta_t_prior, delta_t_posterior):
        
        # Interploate
        if self.inter_p == 'nearest_neighbour':
            sync_u_s = u_s if delta_t_prior < delta_t_posterior else u_s_
        elif self.inter_p == 'linear':
            sync_u_s = u_s + (u_s_ - u_s) * delta_t_prior / (delta_t_prior + delta_t_posterior)
        elif self.inter_p == 'automatic':
            w1 = self.att_w(torch.tensor([delta_t_prior, delta_t_posterior], device=x_f.device))
            w2 = 1 - w1
            sync_u_s = u_s * w1 + u_s_ * w2
        else:
            raise NotImplementedError
        
        return self.sync_block(x_f, sync_u_s, decode=self.slow2obs)
    
    def node_s_evolve(self, u_s, t, dt=1e-3):
        return self.node_s(u_s, t, dt)[:, -1]
    
    def node_f_evolve(self, x_f, u_s, t, fast_dt=1e-3, slow_dt=1e-3):
        
        current_t = t[0].item() # 0.0
        # interploate_interval = 1e-2 # for FHN
        # interploate_interval = fast_dt
        interploate_interval = (slow_dt - fast_dt) / 2
        assert interploate_interval <= fast_dt, "interploate_interval can not larger than fast_dt"
        
        u_s = u_s
        u_s_ = self.node_s_evolve(u_s, torch.tensor([0., slow_dt], device=t.device), slow_dt)
        counter = 1
        while current_t < t[-1]:
        
            # Interploate and Sync
            if self.sync and interploate_interval > 0.0:
                if current_t > counter * slow_dt:
                    counter += 1
                    u_s = u_s_
                    u_s_ = self.node_s_evolve(u_s, torch.tensor([0., slow_dt], device=t.device), slow_dt)
                
                background = self.synergetic(x_f=x_f, u_s=u_s, u_s_=u_s_, delta_t_prior=current_t-(counter-1)*slow_dt, delta_t_posterior=counter*slow_dt)
            else:
                background = torch.zeros([x_f.shape[0], 2*self.in_channels*self.feature_dim], device=x_f.device)

            # Evolve
            if current_t + interploate_interval <= t[-1] and interploate_interval > 0.0:
                x_f = self.node_f(x_f, torch.tensor([0., interploate_interval], device=t.device), fast_dt, background=background)[:, -1:]
            else:
                x_f = self.node_f(x_f, torch.tensor([0., t[-1]-current_t], device=t.device), fast_dt, background=background)[:, -1:]
                break
            
            # Time Update
            current_t += interploate_interval
        
        return x_f
    
    def scale(self, x):
        if x.shape[-1] < self.max.shape[-1]:
            obs_dim = x.shape[-1]
            return (x-self.min[:,:obs_dim]) / (self.max[:,:obs_dim]-self.min[:,:obs_dim]+1e-11)
        else:
            return (x-self.min) / (self.max-self.min+1e-11)
    
    def descale(self, x):
        if x.shape[-1] < self.max.shape[-1]:
            obs_dim = x.shape[-1]
            return x * (self.max[:,:obs_dim]-self.min[:,:obs_dim]+1e-11) + self.min[:,:obs_dim]
        else:
            return x * (self.max-self.min+1e-11) + self.min