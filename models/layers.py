import torch
from torch import nn


class EncodeLayer(nn.Module):

    def __init__(self, input_dim, output_dim, flatten=False, dropout=False, net='MLP', activate=False, layernorm=False, in_channel=2):
        super(EncodeLayer, self).__init__()

        assert net in ['MLP', 'Conv1d', 'GRU'], f"Encoder Net Error, {net} not implemented!"

        self.net = net

        self.activate = activate
        self.flatten = nn.Flatten() if flatten else None
        self.dropout = nn.Dropout(p=0.01) if dropout else None
        self.layernorm = nn.LayerNorm(output_dim) if layernorm else None

        self.w1 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, output_dim)))
        self.b1 = nn.Parameter(torch.zeros(output_dim))
        
        if net == 'Conv1d':
            # input: (N, 2, 100)
            self.input_dim = input_dim
            self.conv = nn.Sequential(
                nn.Conv1d(in_channel, 16, kernel_size=5, stride=1, padding=2), # (N, 2, 400) -> (N, 16, 400)
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2), # (N, 16, 400) -> (N, 16, 200)
                nn.Conv1d(16, 1, kernel_size=5, stride=1, padding=2), # (N, 16, 200) -> (N, 1, 200)
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2), # (N, 1, 200) -> (N, 1, 100)
                nn.Flatten(),
                nn.Linear(int(self.input_dim/2), output_dim)  # (N, 100) -> (N, output_dim)
            )
            self.linear = nn.Sequential(
                self.flatten,
                nn.Linear(input_dim, input_dim*4),
                nn.ReLU(),
            )
        
        if net == 'GRU':
            self.w2 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, output_dim)))
            self.w3 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, output_dim)))
            self.b2 = nn.Parameter(torch.zeros(output_dim))
            self.b3 = nn.Parameter(torch.zeros(output_dim))
            self.b4 = nn.Parameter(torch.zeros(output_dim))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    
    def forward(self, x):

        if self.net == 'Conv1d':
            
            x = self.linear(x)
            
            x = x.view(-1, 2, int(self.input_dim/2))
            y = self.conv(x)
        elif self.net == 'MLP':
            if self.flatten:
                x = self.flatten(x)

            y = x @ self.w1 + self.b1
        
            if self.layernorm:
                y = self.layernorm(y)
            
            if self.activate:
                if self.net == 'MLP':
                    y = self.tanh(y)

            if self.dropout:
                y = self.dropout(y)
        elif self.net == 'GRU':
            if self.flatten:
                x = self.flatten(x)

            z = self.sigmoid(x @ self.w1 + self.b1)
            r = self.sigmoid(x @ self.w2 + self.b2)
            h = self.tanh(x @ self.w3 + self.b3 + r * self.b4)
            y = (1 - z) * h

            if self.dropout:
                y = self.dropout(y)

        return y


class DecodeLayer(nn.Module):

    def __init__(self, input_dim, output_dim, flatten=False, dropout=False, net='MLP', activate=False, layernorm=False, in_channel=2):
        super(DecodeLayer, self).__init__()

        assert net in ['MLP', 'Conv1d'], f"Decoder Net Error, {net} not implemented!"

        self.net = net

        self.activate = activate
        self.flatten = nn.Flatten() if flatten else None
        self.dropout = nn.Dropout(p=0.01) if dropout else None
        self.layernorm = nn.LayerNorm(output_dim) if layernorm else None

        self.w1 = nn.Parameter(torch.normal(mean=0., std=0.01, size=(input_dim, output_dim)))
        self.b1 = nn.Parameter(torch.zeros(output_dim))
        
        if net == 'Conv1d':
            self.output_dim = output_dim
            self.conv = nn.Sequential(
                nn.Linear(input_dim, int(self.output_dim/2)),
                nn.ReLU(),
                nn.Unflatten(-1, (1, int(self.output_dim/2))),
                nn.ConvTranspose1d(1, 16, kernel_size=5, stride=1, padding=2), # (N, 1, 100) -> (N, 16, 100)
                # nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2), # (N, 1, 100) -> (N, 16, 100)
                nn.ReLU(),
                nn.ConvTranspose1d(16, in_channel, kernel_size=5, stride=1, padding=2), # (N, 16, 100) -> (N, 2, 100)
                # nn.Conv1d(16, in_channel, kernel_size=5, stride=1, padding=2), # (N, 16, 100) -> (N, 2, 100)
            )
            self.linear = nn.Sequential(
                self.flatten,
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
            )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    
    def forward(self, x):

        if self.net == 'Conv1d':
            y = self.conv(x)
            y = self.linear(y)
            
        elif self.net == 'MLP':
            if self.flatten:
                x = self.flatten(x)
            
            y = x @ self.w1 + self.b1
        
            if self.layernorm:
                y = self.layernorm(y)
            
            if self.activate:
                y = self.tanh(y)

            if self.dropout:
                y = self.dropout(y)

        return y


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(AttentionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
        self.linear = nn.Linear(hidden_dim, input_dim)
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):        

        query = self.query_linear(x).unsqueeze(0)
        key = self.key_linear(x).unsqueeze(0)
        value = self.value_linear(x).unsqueeze(0)
        
        output, _ = self.attention(query, key, value)
        output = output[0]
        
        output = self.layer_norm1(output)
        
        return output