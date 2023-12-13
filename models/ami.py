import torch
from torch import nn
import numpy as np

from .weight_init import normal_init
from .layers import EncodeLayer, DecodeLayer, AttentionEncoder


class AMINetwork(nn.Module):
    
    def __init__(self, in_channels, feature_dim, embed_dim, data_dim, enc_net='MLP', e1_layer_n=3):
        super(AMINetwork, self).__init__()

        self.channel_num = in_channels
        self.feature_dim = feature_dim
        self.enc_net = enc_net

        if enc_net == 'GRU': # HalfMoon
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
            self.encoder = []
            self.encoder.append(EncodeCell(in_channels*feature_dim, 64, True, True, enc_net))
            [self.encoder.append(EncodeCell(64, 64, False, True, enc_net)) for _ in range(e1_layer_n-2)]
            self.encoder.append(EncodeCell(64, embed_dim, False, False, enc_net, False))
            self.encoder = nn.Sequential(*self.encoder)
            self.decoder_prior = nn.Sequential(
                nn.Linear(embed_dim, 64, bias=True),
                nn.Tanh(),
                nn.Dropout(p=0.01),
                nn.Linear(64, in_channels*feature_dim, bias=True),
                nn.Unflatten(-1, (1, in_channels, feature_dim))
            )
            self.decoder_reverse = nn.Sequential(
                nn.Linear(embed_dim, 64, bias=True),
                nn.Tanh(),
                nn.Dropout(p=0.01),
                nn.Linear(64, embed_dim, bias=True)
            )
        
        else:
            # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)
            self.encoder = []
            for i in range(e1_layer_n):
                self.encoder.append(EncodeLayer(
                    input_dim  = 128 if i>0 else in_channels*feature_dim, 
                    output_dim = 128 if i<e1_layer_n-1 else embed_dim, 
                    flatten    = False if i>0 else True, 
                    dropout    = True if i<e1_layer_n-1 else False, 
                    net        = enc_net, 
                    activate   = True if i<e1_layer_n-1 else False, 
                    layernorm  = True if i<e1_layer_n-1 else False,
                    in_channel = in_channels))
            self.encoder = nn.Sequential(*self.encoder)
            
            self.decoder_prior = []
            self.decoder_reverse = []
            for i in range(e1_layer_n):
                self.decoder_prior.append(DecodeLayer(
                    input_dim  = 128 if i>0 else embed_dim, 
                    output_dim = 128 if i<e1_layer_n-1 else in_channels*feature_dim, 
                    flatten    = False if i>0 else True, 
                    dropout    = True if i<e1_layer_n-1 else False, 
                    net        = 'MLP', 
                    activate   = True if i<e1_layer_n-1 else False, 
                    layernorm  = True if i<e1_layer_n-1 else False,
                    in_channel = in_channels))
                self.decoder_reverse.append(DecodeLayer(
                    input_dim  = 128 if i>0 else embed_dim, 
                    output_dim = 128 if i<e1_layer_n-1 else embed_dim,
                    flatten    = False if i>0 else True, 
                    dropout    = True if i<e1_layer_n-1 else False, 
                    net        = 'MLP', 
                    activate   = True if i<e1_layer_n-1 else False, 
                    layernorm  = True if i<e1_layer_n-1 else False,
                    in_channel = in_channels))
            self.decoder_prior = nn.Sequential(*self.decoder_prior)
            self.decoder_reverse = nn.Sequential(*self.decoder_reverse)
            
        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, data_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, data_dim, dtype=torch.float32))

        # init
        self.apply(normal_init)
                
    def forward(self, x, direct='prior'):

        embed = self.encoder(x)
        if direct=='prior':
            out = self.decoder_prior(embed).view(-1, 1, self.channel_num, self.feature_dim)
        else:
            out = self.decoder_reverse(embed)
        
        return out, embed
    
    def enc(self, x):
        return self.encoder(x)
    
    def dec(self, embed, direct='prior'):
        
        if direct=='prior':
            out = self.decoder_prior(embed).view(-1, 1, self.channel_num, self.feature_dim)
        else:
            out = self.decoder_reverse(embed)
            
        return out

    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-11)
    
    def descale(self, x):
        if x.shape[-1] < self.max.shape[-1]: # for HalfMoon
            return x * (self.max[:,:x.shape[-1]]-self.min[:,:x.shape[-1]]+1e-11) + self.min[:,:x.shape[-1]]
        return x * (self.max-self.min+1e-11) + self.min