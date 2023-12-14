# -*- coding: utf-8 -*-
import signal
import argparse

from main import *


# ctrl c
def term_sig_handler(signum, frame):
    exit(0)
signal.signal(signal.SIGINT, term_sig_handler)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ours')
    parser.add_argument('--submodel', type=str, default='neural_ode')
    parser.add_argument('--xdim', type=int, default=1)
    parser.add_argument('--system', type=str, default='FHN')
    parser.add_argument('--system_name', type=str, default='FHN')
    parser.add_argument('--channel_num', type=int, default=4)
    parser.add_argument('--obs_dim', type=int, default=4)
    parser.add_argument('--data_dim', type=int, default=4)
    parser.add_argument('--trace_num', type=int, default=200)
    parser.add_argument('--total_t', type=float, default=100.1)
    parser.add_argument('--start_t', type=float, default=0.0)
    parser.add_argument('--end_t', type=float, default=5.1)
    parser.add_argument('--step_t', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--delta1', type=float, default=0.0)
    parser.add_argument('--delta2', type=float, default=0.0)
    parser.add_argument('--du', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tau_unit', type=float, default=0.1)
    parser.add_argument('--sliding_length', type=float, default=0.1)
    parser.add_argument('--train_horizon', type=float, default=0.1)
    parser.add_argument('--test_horizon', type=float, default=0.1)
    parser.add_argument('--learn_n', type=int, default=0.1)
    parser.add_argument('--predict_n', type=int, default=0.1)
    parser.add_argument('--stride_t', type=float, default=0.1)
    parser.add_argument('--tau_1', type=float, default=0.1)
    parser.add_argument('--tau_N', type=float, default=3.0)
    parser.add_argument('--tau_s', type=float, default=1.0)
    parser.add_argument('--tau_list', type=float, nargs='+', default=0.8)
    parser.add_argument('--embedding_dim', type=int, default=2)        
    parser.add_argument('--auto', type=int, default=1)
    parser.add_argument('--bi_info', type=int, default=0)
    parser.add_argument('--slow_dim', type=int, default=2)
    parser.add_argument('--fast', type=int, default=1)
    parser.add_argument('--mask_slow', type=int, default=0)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--sync', type=int, default=1)
    parser.add_argument('--rho', type=int, default=1)
    parser.add_argument('--inter_p', type=str, default='nearest_neighbour')
    parser.add_argument('--koopman_dim', type=int, default=4)
    parser.add_argument('--enc_net', type=str, default='MLP')
    parser.add_argument('--e1_layer_n', type=int, default=3)
    parser.add_argument('--id_epoch', type=int, default=100)
    parser.add_argument('--learn_epoch', type=int, default=100)
    parser.add_argument('--baseline_epoch', type=int, default=100)
    parser.add_argument('--seed_num', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--cpu_num', type=int, default=-1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--data_dir', type=str, default='Data/FHN/data/')
    parser.add_argument('--id_log_dir', type=str, default='logs/FHN/TimeSelection/')
    parser.add_argument('--learn_log_dir', type=str, default='logs/FHN/LearnDynamics/')
    parser.add_argument('--baseline_log_dir', type=str, default='logs/FHN/LearnDynamics/')
    parser.add_argument('--memory_size', type=int, default=3000)
    args = parser.parse_args()

    if not args.parallel and args.cpu_num==1:
        print('Not recommand to limit the cpu num when non-parallellism!')
    
    # main pipeline
    if args.model == 'ours':
        # Data_Generate(args, mode='tracjectory')

        # Data_Generate(args, mode='id')
        # ID_Estimate(args)
        
        Data_Generate(args, mode='learn')
        Learn_Slow_Fast(args, 'train')
        Learn_Slow_Fast(args, 'test')
    else:
        Baseline(args, 'train')
        Baseline(args, 'test')
        