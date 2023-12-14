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
    
    
def choice_gpu(memory_size):
    # 使用nvidia-smi命令获取GPU信息
    cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
    output = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")

    # 遍历每个GPU的剩余显存
    flag = False
    for i, free_memory_str in enumerate(output):
        free_memory = int(free_memory_str)

        # 检查剩余显存是否足够
        if free_memory >= memory_size:
            # 设置当前进程使用的GPU
            torch.cuda.set_device(i)
            flag = True
            print(f"SubProcess[{os.getpid()}]: GPU-{i}")
            break
    
    if not flag:
        raise Exception(f"SubProcess[{os.getpid()}]: No GPU can use!")