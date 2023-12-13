import torch
import numpy as np
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(self, file_path, mode='train', length=None, model=None, horizon=None):
        super().__init__()
        
        self.length = length
        self.model = model

        # Search for txt files
        if length is None:
            self.data = np.load(file_path+f'/{mode}.npz')['data'] # (N, sequence_length, channel, feature_dim)
        else:
            if horizon is None:
                self.data = np.load(file_path+f'/{mode}_{length}.npz')['data']
            else:
                self.data = np.load(file_path+f'/{mode}_{length}_{horizon}.npz')['data']

    def __getitem__(self, index):

        trace = self.data[index] # (sequence_length, channel, feature_dim)
        
        input = trace[0]
        if self.length is None:
            target = trace[0] if len(trace)==1 else trace[1]
        else:
            target = trace[self.length-1]

        input = torch.from_numpy(input).float() # (1, channel, feature_dim)
        target = torch.from_numpy(target).float()

        if self.length is None:
            return input.unsqueeze(0), target.unsqueeze(0)
        else:
            return input.unsqueeze(0), target.unsqueeze(0), [torch.from_numpy(trace[i]).float().unsqueeze(0) for i in range(self.length)] 

    def __len__(self):
        return len(self.data)