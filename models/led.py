import torch
from torch import nn

from .weight_init import normal_init


class LSTM(nn.Module):
    
    def __init__(self, data_dim, hidden_dim=64, layer_num=2):
        super(LSTM, self).__init__()
        
        # (batchsize,data_dim)-->(batchsize, hidden_dim)
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(
            input_size=data_dim, 
            hidden_size=hidden_dim, 
            num_layers=layer_num, 
            dropout=0.01, 
            batch_first=True # input: (batch_size, squences, features)
            )
        
        # (batchsize, hidden_dim)-->(batchsize, data_dim)
        self.fc = nn.Linear(hidden_dim, data_dim)

        # init
        self.apply(normal_init)
    
    def forward(self, x, n=1):
        
        device = x.device
        h0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=device)
        c0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=device)
        
        x = x.unsqueeze(-2) # (batch_size, data_dim) ——> (batch_size, 1, data_dim)
        _, (h, c)  = self.cell(x, (h0, c0))
        y = [self.fc(h[-1])]
        
        for _ in range(1, n):
            _, (h, c)  = self.cell(y[-1].unsqueeze(-2), (h, c))
            y.append(self.fc(h[-1]))
                
        return y[-1], y


class AE(nn.Module):

    def __init__(self, in_channels, feature_dim, latent_size, net='MLP'):
        super(AE, self).__init__()
        
        if net == 'MLP':
            # (batchsize,1,channel_num,feature_dim)-->(batchsize,latent_size)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels*feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, latent_size),
            )

            # (batchsize,latent_size)-->(batchsize,1,channel_num,feature_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 64),
                nn.ReLU(),
                nn.Linear(64, in_channels*feature_dim),
                nn.Unflatten(-1, (1, in_channels, feature_dim)),
            )
        else:
            raise NotImplementedError

        # init
        self.apply(normal_init)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


class LED(nn.Module):
    
    def __init__(self, in_channels, feature_dim, data_dim, tau_unit, dt, system_name='FHN', delta1=0.0, delta2=0.0, du=0.5, xdim=30, latent_dim=2, submodel='MLP'):
        super(LED, self).__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.tau_unit = tau_unit
        self.dt = dt
        self.xdim = xdim
        self.system_name = system_name
        
        self.latent_dim = latent_dim
        self.ae = AE(in_channels, feature_dim, self.latent_dim, submodel)
        self.latent_propagator = self._create_propagator(self.latent_dim)

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, data_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, data_dim, dtype=torch.float32))


    def _create_propagator(self, latent_dim, propagator='LSTM'):
        if propagator == 'LSTM':
            model = LSTM(latent_dim)
            
        return model
    
    def macro_propagate(self, x0, n):

        latent = self.ae.encoder(x0)
        latent = self.latent_propagator(latent, n=n)[0]
        y = self.ae.decoder(latent)

        return y

    def micro_propagate(self, x0, n):
        
        pass # unavailable

    def multiscale_pred(self, x, n=1, macro_n=0, micro_n=0):

        if macro_n==0 and micro_n==0:
            macro_n = n
        
        y = x
        count = 0
        while count < n:
            
            if macro_n != 0:
                if n - count <= macro_n:
                    y = self.macro_propagate(y, n=n-count)
                    break
                else:
                    y = self.macro_propagate(y, n=macro_n)
                    count += macro_n
            
            if micro_n != 0:
                if n - count <= micro_n:
                    y = self.micro_propagate(y, n=n-count)
                    break
                else:
                    y = self.micro_propagate(y, n=micro_n)
                    count += macro_n
        
        return y

    def forward(self, x, n=1):
        return self.macro_propagate(x, n)
    
    def latent_forward(self, x, n):
        latent = self.ae.encoder(x)
        latent = self.latent_propagator(latent, n=n)[1]
        return [self.ae.decoder(latent[i]) for i in range(n)]
        
    
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-11)
    
    def descale(self, x):
        if x.shape[-1] < self.max.shape[-1]:
            obs_dim = x.shape[-1]
            return x * (self.max[:,:obs_dim]-self.min[:,:obs_dim]+1e-11) + self.min[:,:obs_dim]
        else:
            return x * (self.max-self.min+1e-11) + self.min