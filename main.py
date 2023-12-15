# -*- coding: utf-8 -*-
import os
import time
import fcntl
import torch
import numpy as np
from multiprocessing import Process

import models
from util import *
from methods import *
from Data.generator import *


def Data_Generate(args, mode='tracjectory'):
    
    if mode=='tracjectory':
        # generate original data
        print('Generating original simulation data')
        origin_dir = args.data_dir.replace('data', 'origin')
        generate_original_data(args.trace_num, args.total_t, args.dt, save=True, plot=True, parallel=args.parallel, xdim=args.xdim, 
                            delta1=args.delta1, delta2=args.delta2, du=args.du, data_dir=origin_dir)

    # load original data
    origin_dir = args.data_dir.replace('data', 'origin')
    
    tmp = np.load(origin_dir+"origin.npz")
    
    # generate dataset for ID estimating
    if mode=='id':
        print('Generating training data for ID estimating')
        
        if 'HalfMoon' in args.system:
            args.stride_t = 1.0
            
        T = args.tau_list
        for tau in T:
            tau = round(tau, 4)
            generate_dataset_slidingwindow(tmp, args.trace_num, tau, is_print=True, data_dir=args.data_dir, start_t=args.start_t, end_t=args.end_t, 
                                            sliding_length=args.sliding_length, stride_t=args.stride_t)
    
    # generate dataset for learning fast-slow dynamics
    if mode=='learn':
        # training
        n = args.learn_n
        print('Generating training data for learning fast-slow dynamics')
        generate_dataset_slidingwindow(tmp, args.trace_num, tau=args.tau_unit, is_print=True, sequence_length=n, data_dir=args.data_dir, start_t=args.start_t, 
                                        end_t=args.end_t, stride_t=args.stride_t, horizon=args.train_horizon)
        # testing
        print('Generating testing data for learning fast-slow dynamics')
        generate_dataset_slidingwindow(tmp, args.trace_num, tau=args.tau_unit, is_print=True, sequence_length=args.predict_n, data_dir=args.data_dir, start_t=args.start_t, end_t=args.end_t, 
                                        stride_t=args.stride_t, only_test=True, horizon=args.test_horizon)


def AutoEmbedSize(args, tau, random_seed=729, is_print=False, gpu_id=0):

    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(args.cpu_num)
    
    if args.device == 'cuda':
        torch.cuda.set_device(gpu_id)

    embed_size_list = []
    nmse_list = []
    next_embed_size = args.embedding_dim

    def id_subworker(args, tau, is_print, embed_size, test=False):
        
        # check if the embedding size is repeated, if so, direatly return the nmse
        log_dir = args.id_log_dir+f'st{args.start_t}_et{args.end_t}/sliding_length-{args.sliding_length}/tau_{tau}'
        if os.path.exists(log_dir+f"/embed_size-{embed_size}/seed{random_seed}/checkpoints/epoch-{args.id_epoch}.ckpt"):
            try:
                with open(log_dir+f'/embed_size-{embed_size}/test_log.txt', 'r') as f:
                    for line in f.readlines():
                        seed = int(line.split(',')[1])
                        nmse = float(line.split(',')[-2])
                        if seed == random_seed and not test:
                            return nmse, False
            except:
                pass
        else:
            # train
            log_dir = log_dir + f'/embed_size-{embed_size}/seed{random_seed}'
            train_ami(args.system, embed_size, args.channel_num, args.obs_dim, tau, args.id_epoch, is_print, random_seed, args.data_dir, log_dir, args.device, 
                            args.data_dim, args.lr, args.batch_size, args.enc_net, args.e1_layer_n, args.start_t, args.end_t, args.sliding_length, random_seed, args.bi_info)

        if not test:
            # test and not calculating ID
            checkpoint_dir = args.id_log_dir+f"st{args.start_t}_et{args.end_t}/sliding_length-{args.sliding_length}/tau_{tau}/embed_size-{embed_size}/seed{random_seed}"
            log_dir = args.id_log_dir+f'st{args.start_t}_et{args.end_t}/sliding_length-{args.sliding_length}/tau_{tau}/embed_size-{embed_size}'
            nmse, _ = test_ami(args.system, embed_size, args.channel_num, args.obs_dim, tau, args.id_epoch, checkpoint_dir, 
                                                        is_print, random_seed, args.data_dir, log_dir, args.device, args.data_dim, args.batch_size, args.enc_net, 
                                                        args.e1_layer_n, args.dt, args.total_t, args.start_t, args.end_t, args.sliding_length, False)
        else:
            # test and calculating ID
            checkpoint_dir = args.id_log_dir+f"st{args.start_t}_et{args.end_t}/sliding_length-{args.sliding_length}/tau_{tau}/embed_size-{embed_size}/seed{random_seed}"
            log_dir = args.id_log_dir+f'st{args.start_t}_et{args.end_t}/sliding_length-{args.sliding_length}/tau_{tau}/final'
            os.makedirs(log_dir, exist_ok=True)
            nmse, _ = test_ami(args.system, embed_size, args.channel_num, args.obs_dim, tau, args.id_epoch, checkpoint_dir, 
                                                        is_print, random_seed, args.data_dir, log_dir, args.device, args.data_dim, args.batch_size, args.enc_net, 
                                                        args.e1_layer_n, args.dt, args.total_t, args.start_t, args.end_t, args.sliding_length, True)
            # record the embedding size
            with open(log_dir+'/embed_size.txt', 'a') as f:
                f.writelines(f'{embed_size} ')

        return nmse, False
    
    def search_smaller(value_list, value):
        # find the largest value that is smaller than the given value
        tmp = []
        for v in value_list:
            if v < value:
                tmp.append(v)
        return 0 if len(tmp) == 0 else max(tmp)
    
    def search_larger(value_list, value):
        # find the smallest value that is larger than the given value
        tmp = []
        for v in value_list:
            if v > value:
                tmp.append(v)
        return value if len(tmp) == 0 else min(tmp)

    iter = 0
    while args.auto:
        
        # An pipeline for ID-driven Time Scale Selection
        if is_print: print(f'Tau[{tau}] Current embed size: {next_embed_size}')
        nmse, final = id_subworker(args, tau, is_print, next_embed_size)
        if is_print: print(f'Tau[{tau}] Current nmse: {nmse:.5f}')
        
        if final:
            print(f'Tau[{tau}] Final embed size: {next_embed_size}')
            break

        # Update the embed size by comparing the current embed size with the last larger embed size
        current_embed_size = next_embed_size
        if iter == 0:
            next_embed_size = np.floor(current_embed_size/2).astype(int)
        else:
            # find the last embed size that is 2-times larger than current embed size
            for i in range(1, len(embed_size_list)+1):
                if 2*current_embed_size <= embed_size_list[-i] or i == len(embed_size_list):
                    compared_nmse = nmse_list[-i]
                    break
            # update the embed size by rules
            if nmse < (1+0.2)*compared_nmse:
                next_embed_size = np.floor((current_embed_size+search_smaller(embed_size_list, current_embed_size))/2).astype(int)
            else:
                next_embed_size = np.floor((current_embed_size+search_larger(embed_size_list, current_embed_size))/2).astype(int)

        if is_print: print(f'Tau[{tau}] Updated embed size: {next_embed_size}')
        
        # Record the embed size and nmse
        embed_size_list.append(current_embed_size)
        nmse_list.append(nmse)

        # Stop condition
        if next_embed_size == 1:
            if is_print: print(f'Tau[{tau}] Embed size is 1, stop!')
            next_embed_size += 1
            break
        elif len(embed_size_list) > 1 and embed_size_list[-1] == embed_size_list[-2]:
            if is_print: print(f'Tau[{tau}] Embed size is not changed, stop!')
            next_embed_size += 1
            break
        else:
            iter += 1
        
        if is_print: print('\n---------------------------------------------------------\n')
    
    # test the final embed size and calculate the ID
    final_embed_size = next_embed_size
    if is_print: print(f'Tau[{tau}] Final embed size: {final_embed_size}')
    nmse = id_subworker(args, tau, is_print, final_embed_size, test=True)
    
    if args.auto:
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(111)
        ax1.plot(embed_size_list, 'o-', label='Embedding Size')
        ax1.set_ylabel('Embedding Size')
        ax1.legend(loc='upper right')
        ax1.set_ylim([0, 1.2*max(embed_size_list)])
        for a, b in zip(range(len(embed_size_list)), embed_size_list):
            ax1.text(a, b+0.1, '%.0f' % b, ha='center', va='bottom')
        ax2 = ax1.twinx()
        ax2.plot(nmse_list, '+-', label='NMSE', color='r', alpha=0.5)
        ax2.set_ylabel('NMSE')
        ax2.legend(loc='upper left')
        ax2.set_ylim([0, 1.2*max(nmse_list)])
        for a, b in zip(range(len(nmse_list)), nmse_list):
            ax2.text(a, b, '%.5f' % b, ha='center', va='bottom')
        plt.xlabel('Iteration')
        plt.tight_layout(); plt.savefig(f'{args.id_log_dir}st{args.start_t}_et{args.end_t}/sliding_length-{args.sliding_length}/tau_{tau}/AutoEmbedSize_seed{random_seed}.png', dpi=300)
        plt.close()

    np.savez(f'{args.id_log_dir}st{args.start_t}_et{args.end_t}/sliding_length-{args.sliding_length}/tau_{tau}/AutoEmbedSize.npz', embed_size_list=embed_size_list, 
             nmse_list=nmse_list, allow_pickle=True)


def learn_autoEmbedSize(args, n, random_seed=729, is_print=False, mode='train', only_extract=False, gpu_id=0):
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(args.cpu_num)
    
    if args.device == 'cuda':
        torch.cuda.set_device(gpu_id)

    embed_size = np.mean(np.loadtxt(args.id_log_dir + f'st{args.start_t}_et{args.end_t}/sliding_length-{args.sliding_length}/tau_{round(args.tau_s,4)}/final/embed_size.txt')).astype(int)
    
    if mode == 'train':
        ckpt_path = args.id_log_dir + f'st{args.start_t}_et{args.end_t}/sliding_length-{args.sliding_length}/tau_{args.tau_s}/embed_size-{embed_size}/seed1/checkpoints/epoch-{args.id_epoch}.ckpt'
        train_sfs(args.system, args.submodel, embed_size, args.channel_num, args.obs_dim, args.tau_s, args.tau_unit, args.slow_dim, args.koopman_dim, 
                                      n, ckpt_path, is_print, random_seed, args.learn_epoch, args.data_dir, args.learn_log_dir, args.device, args.data_dim, args.lr, 
                                      args.batch_size, args.enc_net, args.e1_layer_n, args.dt, args.total_t, args.start_t, args.end_t, only_extract, random_seed, 
                                      args.stride_t, horizon=args.train_horizon, sliding_length=args.sliding_length, fast=args.fast, rho=args.rho, mask_slow=args.mask_slow, sync=args.sync,
                                      inter_p=args.inter_p, num_heads=args.num_heads)
    elif mode == 'test':
        os.makedirs(f'results/{args.system}/', exist_ok=True)
        
        interval = 50
        n_list = range(1,args.predict_n+1,interval)
        
        nmse_per_tau, time_cost = test_sfs(args.system, args.submodel, embed_size, args.channel_num, args.obs_dim, args.tau_s, args.learn_epoch, args.slow_dim, args.koopman_dim, 
                                args.tau_unit, args.predict_n, n_list, is_print, random_seed, args.data_dir, args.learn_log_dir, args.device, args.data_dim, args.batch_size, args.enc_net, 
                                args.e1_layer_n, args.dt, args.total_t, args.start_t, args.end_t, args.stride_t, horizon=args.test_horizon, sliding_length=args.sliding_length, fast=args.fast, 
                                rho=args.rho, mask_slow=args.mask_slow, test_horizon=args.test_horizon, predict_n=args.predict_n, sync=args.sync, inter_p=args.inter_p, num_heads=args.num_heads)

        with open(f'results/{args.system}/ours-{args.submodel}-sfs-{args.slow_dim}_fast{args.fast}_sync{args.sync}_rho{args.rho}_{args.inter_p}_evolve_test_{args.tau_s}.txt','a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            for i in range(args.predict_n):
                f.writelines(f'{(i+1)*args.tau_unit:.4f}, {random_seed}, {nmse_per_tau[i]}, {time_cost[i]}\n')
                f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        print('save at ', f'results/{args.system}/ours-{args.submodel}-sfs-{args.slow_dim}_fast{args.fast}_sync{args.sync}_rho{args.rho}_{args.inter_p}_evolve_test_{args.tau_s}.txt')
    else:
        raise TypeError(f"Wrong mode of {mode}!")


def baseline_subworker(args, is_print=False, random_seed=729, mode='train', gpu_id=0):

    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(1)
    
    if args.device == 'cuda':
        torch.cuda.set_device(gpu_id)

    if 'neural_ode' in args.model:
        model = models.NeuralODE(in_channels=args.channel_num, feature_dim=args.obs_dim, data_dim=args.data_dim, submodel=args.submodel)
    elif 'led' in args.model:
        model = models.LED(in_channels=args.channel_num, feature_dim=args.obs_dim, data_dim=args.data_dim, tau_unit=args.tau_unit, dt=args.dt, system_name=args.system_name, delta1=args.delta1, delta2=args.delta2, 
                           du=args.du, xdim=args.xdim, latent_dim=args.slow_dim, submodel=args.submodel)
    elif 'deepkoopman' in args.model:
        model = models.DeepKoopman(in_channels=args.channel_num, feature_dim=args.obs_dim, data_dim=args.data_dim, kdim=args.slow_dim, submodel=args.submodel)
        
    if mode == 'train':
        # train
        baseline_train(model, args.obs_dim, args.data_dim, args.channel_num, args.tau_s, args.tau_unit, is_print, random_seed, 
                       args.baseline_epoch, args.data_dir, args.baseline_log_dir, args.device, args.lr, args.batch_size, args.dt, args.total_t, args.start_t, 
                       args.end_t, args.stride_t, horizon=args.train_horizon, sliding_length=args.sliding_length, learn_n=args.learn_n)
    elif mode == 'test':
        os.makedirs(f'results/{args.system}/', exist_ok=True)
        if 'FHN' in args.system:
            interval = 10
            n_list = range(1, args.predict_n+1, interval)
        elif 'Coupled_Lorenz' in args.system:
            interval = 50
            n_list = range(1,args.predict_n+1,interval)
        else: 
            interval = 50
            n_list = range(1, args.predict_n+1, interval)
        
        nmse_per_tau = baseline_test(model, args.obs_dim, args.system, args.tau_s, args.tau_unit, args.predict_n,n_list, random_seed, args.data_dir, args.baseline_log_dir, 
                                    args.device, args.batch_size, args.dt, args.total_t, args.start_t, args.end_t, args.stride_t, args.baseline_epoch, horizon=args.test_horizon, 
                                    sliding_length=args.sliding_length, test_horizon=args.test_horizon, predict_n=args.predict_n)
        with open(f'results/{args.system}/{args.model}_evolve_test_{args.tau_s}.txt','a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            for i in range(args.predict_n):
                f.writelines(f'{(i+1)*args.tau_unit:.4f}, {random_seed}, {nmse_per_tau[i]}\n')
                f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
    else:
        raise TypeError(f"Wrong mode of {mode}!")


def ID_Estimate(args):
    
    print('Estimating the ID per tau')

    # id estimate process
    T = args.tau_list
    workers = []
    gpu_controller, gpu_id = AutoGPU(args.memory_size), 0
    for tau in T:
        tau = round(tau, 4)
        random_seed = 1
        if args.parallel: # multi-process to speed-up
            is_print = True if len(workers)==0 else False
            if args.device == 'cuda':
                gpu_id = gpu_controller.choice_gpu()
            workers.append(Process(target=AutoEmbedSize, args=(args, tau, random_seed, is_print, gpu_id), daemon=True))
            workers[-1].start()
            time.sleep(0.1)
        else:
            AutoEmbedSize(args, tau, random_seed, True)
                
    # block
    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass
    workers = []

    plot_id_per_tau(T, [1], args.id_epoch, args.id_log_dir, args.start_t, args.end_t, args.sliding_length, True)
    
    if 'cuda' in args.device: torch.cuda.empty_cache()
    print('\nID Estimate Over')


def Learn_Slow_Fast(args, mode='train', only_extract=False):
    
    print(f'{mode.capitalize()} the learning of slow and fast dynamics')
    
    # slow evolve sub-process
    n = args.learn_n
    workers = []
    gpu_controller, gpu_id = AutoGPU(args.memory_size), 0
    for random_seed in range(1, args.seed_num+1):
        if args.parallel:
            is_print = True if len(workers)==0 else False
            if args.device == 'cuda':
                gpu_id = gpu_controller.choice_gpu()
            workers.append(Process(target=learn_autoEmbedSize, args=(args, n, random_seed, is_print, mode, only_extract, gpu_id), daemon=True))
            workers[-1].start()
            time.sleep(0.1)
        else:
            learn_autoEmbedSize(args, n, random_seed, True, mode, only_extract)
    # block
    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass
    
    if 'cuda' in args.device: torch.cuda.empty_cache()
    print('\nSlow-Fast Evolve Over')


def Baseline(args, mode='train'):

    print(f'Running the {args.model.upper()}')

    workers = []
    gpu_controller, gpu_id = AutoGPU(args.memory_size), 0
    for random_seed in range(1, args.seed_num+1):
        if args.parallel:
            is_print = True if len(workers)==0 else False
            if args.device == 'cuda':
                gpu_id = gpu_controller.choice_gpu()
            workers.append(Process(target=baseline_subworker, args=(args, is_print, random_seed, mode, gpu_id), daemon=True))
            workers[-1].start()
        else:
            baseline_subworker(args, True, random_seed, mode)
    # block
    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass

    if 'cuda' in args.device: torch.cuda.empty_cache()
    print(f'{args.model.upper()} running Over')