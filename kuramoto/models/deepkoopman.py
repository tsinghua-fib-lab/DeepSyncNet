import torch
import torch.nn as nn


class DeepKoopman(nn.Module):
    
    def __init__(self, in_channels, feature_dim, data_dim, kdim=8, submodel='MLP'):
        super(DeepKoopman, self).__init__()
        
        if submodel == 'MLP':
            self.encoder = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(in_channels*feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.01),
                nn.Linear(64, kdim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(kdim, 64),
                nn.ReLU(),
                nn.Dropout(0.01),
                nn.Linear(64, in_channels*feature_dim),
                nn.Unflatten(-1, (1, in_channels, feature_dim))
            )
        else:
            raise NotImplementedError
        
        self.Knet = nn.Linear(kdim, kdim, bias=False)
        
        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, data_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, data_dim, dtype=torch.float32))
    
    def evolve(self, x0, n):
        
        latent = self.encoder(x0)
        for _ in range(n):
            latent = self.Knet(latent)
        out = self.decoder(latent)
            
        return out, latent
    
    def evolve_latent(self, latent, n):
        for _ in range(n):
            latent = self.Knet(latent)
        return latent
    
    def encode_decode(self, x):
        
        latent = self.encoder(x)
        out = self.decoder(latent)
            
        return out, latent
        
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-11)
    
    def descale(self, x):
        if x.shape[-1] < self.max.shape[-1]:
            obs_dim = x.shape[-1]
            return x * (self.max[:,:obs_dim]-self.min[:,:obs_dim]+1e-11) + self.min[:,:obs_dim]
        else:
            return x * (self.max-self.min+1e-11) + self.min