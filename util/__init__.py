import os
import torch
import warnings;warnings.simplefilter('ignore')
from .common import set_cpu_num, seed_everything, AutoGPU
from .intrinsic_dimension import eval_id_data, eval_id_embedding
from .visual_config import *
from .results_visual import plot_id_per_tau