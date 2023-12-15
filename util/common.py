import os
import torch
import random
import subprocess
import numpy as np


def set_cpu_num(cpu_num):
    if cpu_num <= 0: return
    
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class AutoGPU():
    
    def __init__(self, memory_size):
        
        self.memory_size = memory_size
        
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")

        self.free_memory = []
        for i, free_memory_str in enumerate(output):
            self.free_memory.append(int(free_memory_str))
     
    def choice_gpu(self):

        flag = False
        for i, free_memory in enumerate(self.free_memory):

            if free_memory >= self.memory_size:
                
                flag = True
                self.free_memory[i] -= self.memory_size
                print(f"GPU-{i}: {free_memory}MB -> {self.free_memory[i]}MB")
                
                return i
        
        if not flag:
            raise Exception(f"SubProcess[{os.getpid()}]: No GPU can use!")