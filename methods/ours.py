# -*- coding: utf-8 -*-
import os
import time
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

import models
from util import *
from Data.dataset import Dataset


def train_ami(
        system,
        embedding_dim,
        channel_num,
        obs_dim,
        tau, 
        max_epoch, 
        is_print=False, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/TimeSelection/', 
        device=torch.device('cpu'),
        data_dim=4,
        lr=0.001,
        batch_size=128,
        enc_net='MLP',
        e1_layer_n=3,
        start_t=0.0,
        end_t=None,
        sliding_length=None,
        seed=None,
        bi_info=False
        ):
    
    # prepare
    if sliding_length is None:
        data_filepath = data_dir + f'st{start_t}_et{end_t}/' + 'tau_' + str(tau)
    else:
        data_filepath = data_dir + f'st{start_t}_et{end_t}/sliding_length-{sliding_length}/' + 'tau_' + str(tau)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)
    
    # init model
    model = models.AMINetwork(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, data_dim=data_dim, enc_net=enc_net, e1_layer_n=e1_layer_n)
    scale_path = data_filepath.replace(f'/tau_{tau}', '')
    model.min = torch.from_numpy(np.loadtxt(scale_path+"/data_min.txt").reshape(channel_num,data_dim).astype(np.float32))
    model.max = torch.from_numpy(np.loadtxt(scale_path+"/data_max.txt").reshape(channel_num,data_dim).astype(np.float32))
    model.to(device)
    
    # training params
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    # dataset
    train_dataset = Dataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = Dataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    losses = []
    loss_curve = []
    iter = tqdm(range(1, max_epoch+1)) if is_print else range(1, max_epoch+1)
    for epoch in iter:
        
        # train
        model.train()
        for x_t0, x_t1 in train_loader:
            x_t0 = model.scale(x_t0.to(device))[..., :obs_dim] # (batchsize,1,channel_num,feature_dim)
            x_t1 = model.scale(x_t1.to(device))[..., :obs_dim]
            
            prior, embed1 = model.forward(x_t0, direct='prior')
            prior_loss = mse_loss(prior, x_t1)

            if bi_info:
                latent_reverse, _ = model.forward(x_t1, direct='reverse')
                latent_t0 = model.enc(x_t0)
                reverse_loss = mse_loss(latent_reverse, latent_t0)
                loss = prior_loss + 0.01*reverse_loss
            else:
                loss = prior_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append([prior_loss.detach().item(), 0., 0.])
            
        loss_curve.append(np.mean(losses, axis=0))
        
        # validate
        with torch.no_grad():
            
            model.eval()
            for x_t0, x_t1 in val_loader:
                x_t0 = model.scale(x_t0.to(device))[..., :obs_dim]
                x_t1 = model.scale(x_t1.to(device))[..., :obs_dim]
            
                prior, embed1 = model.forward(x_t0, direct='prior')

                if bi_info:
                    latent_reverse, _ = model.forward(x_t1, direct='reverse')
                    latent_t0 = model.enc(x_t0)
                
            prior_loss = mse_loss(prior, x_t1)
            
            if bi_info:
                reverse_loss = mse_loss(latent_reverse, latent_t0)
                if is_print: 
                    iter.set_description(f'Tau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE: prior={prior_loss:.5f}, reverse={reverse_loss:.5f}')
            else:
                if is_print:
                    iter.set_description(f'Tau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE: prior={prior_loss:.5f}')
        
        # save last epoch model
        if epoch==max_epoch or (epoch%50==0 and epoch>1):
            model.train()
            torch.save({'model': model.state_dict()}, log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
        
    # plot loss curve
    plt.figure(figsize=(10,6))
    plt.plot(np.array(loss_curve)[:,0], label='prior', c=colors[0])
    plt.legend()
    plt.xlabel('epoch')
    plt.title('Training Loss Curve')
    plt.tight_layout(); plt.savefig(log_dir+'/training_loss.png', dpi=300)
    np.save(log_dir+'/training_loss.npy', loss_curve)
    if is_print: print()
        

def test_ami(
        system,
        embedding_dim,
        channel_num,
        obs_dim,
        tau, 
        max_epoch, 
        checkpoint_filepath=None, 
        is_print=False, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/TimeSelection/', 
        device=torch.device('cpu'),
        data_dim=4,
        batch_size=128,
        enc_net='MLP',
        e1_layer_n=3,
        dt=0.001,
        total_t=0.9,
        start_t=0.0,
        end_t=None,
        sliding_length=None,
        test_id=True
        ):
    
    # prepare
    data_filepath = data_dir + f'st{start_t}_et{end_t}/sliding_length-{sliding_length}/' + 'tau_' + str(tau)
    
    # testing params
    loss_func = nn.MSELoss().to(device)
    
    # init model
    model = models.AMINetwork(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, data_dim=data_dim, enc_net=enc_net, e1_layer_n=e1_layer_n)
    if checkpoint_filepath is None: # not trained
        scale_path = data_filepath.replace(f'/tau_{tau}', '/')
        model.min = torch.from_numpy(np.loadtxt(scale_path+"/data_min.txt").reshape(channel_num,data_dim).astype(np.float32))
        model.max = torch.from_numpy(np.loadtxt(scale_path+"/data_max.txt").reshape(channel_num,data_dim).astype(np.float32))

    # testing pipeline
    fp = open(log_dir + '/test_log.txt', 'w')
    for ep in [max_epoch]:
        
        # load weight file
        epoch = ep
        if checkpoint_filepath is not None:
            epoch = ep + 1 if ep==0 else ep
        ckpt_path = checkpoint_filepath + f"/checkpoints/" + f'epoch-{epoch}.ckpt'
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
        model.eval()
        
        all_embeddings = []
        test_outputs, test_targets, test_hidden = [], [], []
        var_log_dir = log_dir + f'/seed{random_seed}/test/epoch-{epoch}'
        os.makedirs(var_log_dir, exist_ok=True)
        
        # testing
        save_batch_num = int(np.ceil(sliding_length / dt / batch_size))
        with torch.no_grad():

            count = 0
            for index, item in enumerate(['train', 'val', 'test']):
                # dataset
                dataset = Dataset(data_filepath, item)
                Loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                for input, target in Loader:
                    input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,1,4)
                    target = model.scale(target.to(device))[..., :obs_dim]

                    if 'HalfMoon' in system:
                        hidden = input[..., obs_dim:]
                    
                    output, embeddings = model.forward(input)
                    
                    # save the embedding vectors
                    for embedding in embeddings:
                        all_embeddings.append(embedding.cpu().numpy())

                    # save the test outputs and targets
                    if count <= save_batch_num and index==2:
                        count += 1
                        test_outputs = output.cpu() if not len(test_outputs) else torch.cat((test_outputs, output.cpu()), dim=0)
                        test_targets = target.cpu() if not len(test_targets) else torch.cat((test_targets, target.cpu()), dim=0)
                        if 'HalfMoon' in system:
                            test_hidden = hidden if not len(test_hidden) else torch.cat((test_hidden, hidden), dim=0)
            
            # descale
            test_targets = model.descale(test_targets)
            test_outputs = model.descale(test_outputs)
            
            # test mse, nmse
            mse_ = []
            nmse_ = []
            mse_ = loss_func(test_outputs, test_targets)
            nmse_ = torch.mean((test_targets - test_outputs)**2) / torch.var(test_targets)
            mae_ = torch.mean(torch.abs(test_targets - test_outputs))
            if is_print: print(f'MAE: {mae_:.4f}')
                
        if 'FHN' in system:
            plt.figure(figsize=(16,5))
            for j in range(2):
                data = []
                for i in range(len(test_outputs)):
                    data.append([test_outputs[i,0,j,1], test_targets[i,0,j,1]])
                ax = plt.subplot(1,2,j+1)
                ax.set_title(['u_x0', 'v_x0'][j] + f' | xdim[{1}]')
                ax.plot(np.array(data)[:,1], label='true', c=colors[0])
                ax.plot(np.array(data)[:,0], label='predict', c=colors[1])
                ax.legend()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
            plt.tight_layout(); plt.savefig(var_log_dir+"/result.png", dpi=300)
            plt.close()

            # heatmap
            plt.figure(figsize=(13,18))
            ax1 = plt.subplot(321)
            ax1.set_xlabel('x')
            ax1.set_ylabel('t')
            ax1.set_title('true u')
            im1 = ax1.imshow(test_targets.cpu().numpy()[::-1,0,0,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im1, ax=ax1)
            ax2 = plt.subplot(322)
            ax2.set_title('true v')
            ax2.set_xlabel('x')
            ax2.set_ylabel('t')
            im2 = ax2.imshow(test_targets.cpu().numpy()[::-1,0,1,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im2, ax=ax2)
            ax3 = plt.subplot(323)
            ax3.set_xlabel('x')
            ax3.set_ylabel('t')
            ax3.set_title('pred u')
            im3 = ax3.imshow(test_outputs.cpu().numpy()[::-1,0,0,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im3, ax=ax3)
            ax4 = plt.subplot(324)
            ax4.set_title('pred v')
            ax4.set_xlabel('x')
            ax4.set_ylabel('t')
            im4 = ax4.imshow(test_outputs.cpu().numpy()[::-1,0,1,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im4, ax=ax4)
            ax5 = plt.subplot(325)
            ax5.set_title('error u')
            ax5.set_xlabel('x')
            ax5.set_ylabel('t')
            im5 = ax5.imshow((test_outputs-test_targets).cpu().numpy()[::-1,0,0,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im5, ax=ax5)
            ax6 = plt.subplot(326)
            ax6.set_title('error v')
            ax6.set_xlabel('x')
            ax6.set_ylabel('t')
            im6 = ax6.imshow((test_outputs-test_targets).cpu().numpy()[::-1,0,1,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im6, ax=ax6)
            plt.tight_layout(); plt.savefig(var_log_dir+"/error_heatmap.png", dpi=300)
            plt.close()
        
        elif 'HalfMoon_2D' in system:
            plt.figure(figsize=(10,5))
            ax = plt.subplot(121)
            ax.plot(test_targets[:,0,0,0], label='true', c=colors[0])
            ax.plot(test_outputs[:,0,0,0], label='predict', c=colors[1])
            ax.set_ylabel('x')
            ax.legend()
            ax = plt.subplot(122)
            ax.plot(test_targets[:,0,0,1], label='true', c=colors[0])
            ax.plot(test_outputs[:,0,0,1], label='predict', c=colors[1])
            ax.set_ylabel('y')
            ax.legend()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
            plt.tight_layout(); plt.savefig(var_log_dir+"/result.png", dpi=300)
            plt.close()
        
        elif 'PNAS18' in system or 'Coupled_Lorenz' in system:
            # heatmap
            plt.figure(figsize=(18,18))
            ax1 = plt.subplot(331)
            ax1.set_xlabel('space')
            ax1.set_ylabel('t')
            ax1.set_title('true x')
            im1 = ax1.imshow(test_targets.cpu().numpy()[::-1,0,0,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im1, ax=ax1)
            ax2 = plt.subplot(332)
            ax2.set_title('true y')
            ax2.set_xlabel('space')
            ax2.set_ylabel('t')
            im2 = ax2.imshow(test_targets.cpu().numpy()[::-1,0,1,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im2, ax=ax2)
            ax7 = plt.subplot(333)
            ax7.set_title('true z')
            ax7.set_xlabel('space')
            ax7.set_ylabel('t')
            im7 = ax7.imshow(test_targets.cpu().numpy()[::-1,0,2,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im7, ax=ax7)
            ax3 = plt.subplot(334)
            ax3.set_xlabel('space')
            ax3.set_ylabel('t')
            ax3.set_title('pred x')
            im3 = ax3.imshow(test_outputs.cpu().numpy()[::-1,0,0,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im3, ax=ax3)
            ax4 = plt.subplot(335)
            ax4.set_title('pred y')
            ax4.set_xlabel('space')
            ax4.set_ylabel('t')
            im4 = ax4.imshow(test_outputs.cpu().numpy()[::-1,0,1,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im4, ax=ax4)
            ax8 = plt.subplot(336)
            ax8.set_title('pred z')
            ax8.set_xlabel('space')
            ax8.set_ylabel('t')
            im8 = ax8.imshow(test_outputs.cpu().numpy()[::-1,0,2,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im8, ax=ax8)
            ax5 = plt.subplot(337)
            ax5.set_title('error x')
            ax5.set_xlabel('space')
            ax5.set_ylabel('t')
            im5 = ax5.imshow((test_outputs-test_targets).cpu().numpy()[::-1,0,0,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im5, ax=ax5)
            ax6 = plt.subplot(338)
            ax6.set_title('error y')
            ax6.set_xlabel('space')
            ax6.set_ylabel('t')
            im6 = ax6.imshow((test_outputs-test_targets).cpu().numpy()[::-1,0,1,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im6, ax=ax6)
            ax9 = plt.subplot(339)
            ax9.set_title('error z')
            ax9.set_xlabel('space')
            ax9.set_ylabel('t')
            im9 = ax9.imshow((test_outputs-test_targets).cpu().numpy()[::-1,0,2,:], aspect='auto', cmap=my_cmap, extent=[0, test_targets.shape[-1], start_t+tau, sliding_length+tau])
            plt.colorbar(im9, ax=ax9)
            plt.tight_layout(); plt.savefig(var_log_dir+"/error_heatmap.png", dpi=300)
            plt.close()

        if test_id:

            all_embeddings = np.array(all_embeddings)
            
            # calculae ID
            def cal_id_embedding(method='MLE', is_print=False, max_point=1000, k_list=20, all_embeddings=None):
                dims = eval_id_embedding(var_log_dir, method=method, is_print=is_print, max_point=max_point, k_list=k_list, embedding=all_embeddings)
                return np.mean(dims)

            # hyper-parameters            
            max_point = 1000
            if 'FHN' in system:
                if 'FHNv' in system:
                    k_list = np.arange(5, 12+1, 1).astype(int)
                else:
                    k_list = np.arange(3, 7+1, 1).astype(int)
            elif 'HalfMoon' in system:
                k_list = np.arange(15, 50+1, 1).astype(int)
            elif 'Coupled_Lorenz' in system:
                k_list = np.arange(3, 10+1, 1).astype(int)
            
            MLE_id, MADA_id, MOM_id, TLE_id, MIND_id = [], [], [], [], []
            for i in range(20):
                MLE_id.append(cal_id_embedding('MLE', i==0, max_point, k_list, all_embeddings))
                MADA_id.append(cal_id_embedding('MADA', i==0, max_point, k_list, all_embeddings))
                MOM_id.append(cal_id_embedding('MOM', i==0, max_point, k_list, all_embeddings))
                TLE_id.append(cal_id_embedding('TLE', i==0, max_point, k_list, all_embeddings))
                MIND_id.append(cal_id_embedding('MiND_ML', i==0, max_point, k_list, all_embeddings))
                if is_print: print(f'\rTau[{tau}] | MLE={MLE_id[-1]:.1f}, MADA={MADA_id[-1]:.1f}, TLE={TLE_id[-1]:.1f}, MOM={MOM_id[-1]:.1f}, MIND={MIND_id[-1]:.1f} (iter[{i+1}])', end='')
            if is_print: print()
            
            if is_print:
                plt.figure(figsize=(8,6))
                plt.plot(np.arange(1, len(MLE_id)+1), MLE_id, label='MLE', c=colors[0])
                plt.plot(np.arange(1, len(MADA_id)+1), MADA_id, label='MADA', c=colors[1])
                plt.plot(np.arange(1, len(TLE_id)+1), TLE_id, label='TLE', c=colors[2])
                plt.plot(np.arange(1, len(MOM_id)+1), MOM_id, label='MOM', c=colors[3])
                plt.plot(np.arange(1, len(MIND_id)+1), MIND_id, label='MIND', c=colors[4])
                plt.legend()
                plt.xlabel('iter')
                plt.ylabel('ID')
                plt.tight_layout(); plt.savefig(var_log_dir+'/ID_iter.png', dpi=300)
            
            MLE_id, MADA_id, MOM_id, TLE_id, MIND_id = np.mean(MLE_id), np.mean(MADA_id), np.mean(MOM_id), np.mean(TLE_id), np.mean(MIND_id)
            if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MLE={MLE_id:.1f}, MADA={MADA_id:.1f}, TLE={TLE_id:.1f}, MOM={MOM_id:.1f}, MIND={MIND_id:.1f}', end='')
        else:
            MLE_id, MADA_id, MOM_id, TLE_id, MIND_id = 0., 0., 0., 0., 0.
            
        # logging
        fp.write(f"{tau},{random_seed},{mse_},{epoch},{MLE_id},{MOM_id},{MADA_id},{TLE_id},{MIND_id},{nmse_},{embedding_dim}\n")
        fp.flush()

        if checkpoint_filepath is None: break
    
    fp.close()
    
    return nmse_, MLE_id


def train_sfs(
        system,
        submodel,
        embedding_dim,
        channel_num,
        obs_dim,
        tau_s,
        tau_unit,
        slow_dim, 
        koopman_dim, 
        n,
        ckpt_path,
        is_print=False, 
        random_seed=729, 
        learn_max_epoch=100, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/LearnDynamics/', 
        device='cpu',
        data_dim=4,
        lr=0.01,
        batch_size=128,
        enc_net='MLP',
        e1_layer_n=3,
        dt=0.001,
        total_t=0.9,
        start_t=0.0,
        end_t=None,
        only_extract=False,
        seed=None,
        stride_t=0.01,
        tau_1=0.1,
        horizon=None,
        sliding_length=None,
        fast=1,
        rho=1,
        mask_slow=0,
        sync=True,
        inter_p='nearest_neighbour',
        num_heads=1
        ):
        
    # prepare
    data_filepath = data_dir + f'st{start_t}_et{end_t if end_t else total_t}/' + 'tau_' + str(tau_unit)
    log_dir = log_dir + f'seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    assert koopman_dim>=slow_dim, f"Value Error, koopman_dim is smaller than slow_dim({koopman_dim}<{slow_dim})"
    model = models.Slow_Fast_Synergetic_ODE(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, slow_dim=slow_dim, redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, tau_1=tau_1, device=device, data_dim=data_dim, enc_net=enc_net, e1_layer_n=e1_layer_n, sync=sync, inter_p=inter_p, num_heads=num_heads)
    scale_path = data_filepath.replace(f'/tau_{tau_unit}', '')
    model.min = torch.from_numpy(np.loadtxt(scale_path+"/data_min.txt").reshape(channel_num,data_dim).astype(np.float32))
    model.max = torch.from_numpy(np.loadtxt(scale_path+"/data_max.txt").reshape(channel_num,data_dim).astype(np.float32))
    
    # time-lagged autoencoder model
    ckpt = torch.load(ckpt_path)
    time_lagged = models.AMINetwork(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, data_dim=data_dim, enc_net=enc_net, e1_layer_n=e1_layer_n)
    time_lagged.load_state_dict(ckpt['model'])
    time_lagged = time_lagged.to(device)

    # load pretrained time-lagged AE
    model.encoder_1.load_state_dict(time_lagged.encoder.state_dict())
    model = model.to(device)
    
    # training params
    weight_decay = 0.001
    MSE_loss = nn.MSELoss().to(device)
    if inter_p == 'automatic':
        optimizer = torch.optim.AdamW(
            [{'params': model.encoder_2.parameters()},
            {'params': model.decoder_1.parameters()},
            {'params': model.decoder_2.parameters()}, 
            {'params': model.node_s.parameters()},
            {'params': model.node_f.parameters()},
            {'params': model.sync_block.parameters()},
            {'params': model.att_w.parameters()},
            ],
            lr=lr, weight_decay=weight_decay) # not involve encoder_1 (freezen)
    else:
        optimizer = torch.optim.AdamW(
            [{'params': model.encoder_2.parameters()},
            {'params': model.decoder_1.parameters()},
            {'params': model.decoder_2.parameters()},
            {'params': model.node_s.parameters()},
            {'params': model.node_f.parameters()},
            {'params': model.sync_block.parameters()},
            ],
            lr=lr, weight_decay=weight_decay) # not involve encoder_1 (freezen)
    model.encoder_1.requires_grad_(False)
    
    # dataset
    train_dataset = Dataset(data_filepath, 'train', length=n, horizon=horizon)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = Dataset(data_filepath, 'val', length=n, horizon=horizon)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    if 'FHN' in data_filepath:
        if 'FHNv' in data_filepath:
            ode_dt = 1e-1
        else:
            ode_dt = 1e-3
    elif 'HalfMoon' in data_filepath:
        ode_dt = 1.0
    elif 'Coupled_Lorenz' in data_filepath:
        ode_dt = 1e-2
        
    val_loss = []
    for epoch in range(1, learn_max_epoch+1):

        # warm up
        if epoch <= 1:
            only_extract = True
        else:
            only_extract = False
            
        # freeze Separation Module after half epoch
        if epoch > int(learn_max_epoch/2):
            model.encoder_2.requires_grad_(False)
            model.decoder_1.requires_grad_(False)
            model.decoder_2.requires_grad_(False)
        
        # train
        model.train()
        if is_print:
            from tqdm import tqdm
            train_loader = tqdm(train_loader)
        for input, target, internl_units in train_loader:
                        
            t1 = time.time()
                        
            # scale data
            input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,feature_dim,xdim)
            target = model.scale(target.to(device))[..., :obs_dim] # (batchsize,1,feature_dim,xdim)
            
            #################
            # slow extracting
            #################
            # obs —— embedding —— slow representation —— embedding(reconstruction) —— obs(reconstruction)
            slow_var, embed = model.obs2slow(input)
            recons_obs, recons_embed = model.slow2obs(slow_var)
            
            # calculate loss value
            adiabatic_loss = 0.0
            embed_reconstruct_loss = MSE_loss(recons_embed, embed)
            obs_reconstruct_loss = MSE_loss(recons_obs, input)

            t2 = time.time()

            ################
            # n-step evolve
            ################
            obs_evol_loss, slow_evol_loss = 0.0, 0.0
            if not only_extract:

                fast_obs = input - recons_obs
                unit_fast_obs_list = [fast_obs]  # fast obs evolve
                unit_slow_var_list = [slow_var]  # slow var evolve
                unit_slow_obs_list = [recons_obs]  # slow obs evolve
                t = torch.tensor([0., tau_unit], device=device)
                
                if submodel == 'neural_ode':
                                    
                    step_n = 1
                    if ode_dt*rho <= tau_unit:
                        slow_t = torch.tensor([0., tau_unit], device=device)
                        slow_n = 1
                    else:
                        t_step = ode_dt*rho - ode_dt*rho % tau_unit + tau_unit
                        slow_t = torch.tensor([0., t_step], device=device)
                        slow_n = int(t_step / tau_unit)
                    
                    for i in range(n):
                        
                        unit = model.scale(internl_units[i].to(device))[..., :obs_dim] # t+i
                                                
                        # slow evolve
                        if step_n % slow_n == 0:
                            unit_slow_var_list.append(model.node_s_evolve(unit_slow_var_list[-1], slow_t, dt=ode_dt*rho))
                        else:
                            unit_slow_var_list.append(unit_slow_var_list[-1])
                        step_n += 1
                        unit_slow_var, _ = model.obs2slow(unit)
                        slow_evol_loss += MSE_loss(unit_slow_var_list[-1], unit_slow_var) # for slow
                                            
                        # fast evolve
                        if fast:
                            unit_fast_obs_list.append(model.node_f_evolve(unit_fast_obs_list[-1], unit_slow_var_list[-2], t, fast_dt=ode_dt, slow_dt=ode_dt*rho))
                        else:
                            unit_fast_obs_list.append(torch.zeros_like(unit_fast_obs_list[-1]))
                        unit_slow_obs_list.append(model.slow2obs(unit_slow_var_list[-1])[0])
                        unit_fast_obs_next = unit_fast_obs_list[-1]
                        unit_slow_obs_next = unit_slow_obs_list[-1]
                        
                        # total obs evolve
                        unit_obs_next = unit_slow_obs_next + unit_fast_obs_next
                        
                        # evolve loss
                        obs_evol_loss += MSE_loss(unit_obs_next, unit)

                t3 = time.time()

            # optimize
            optimizer.zero_grad()
            all_loss = 0.1*adiabatic_loss + embed_reconstruct_loss + obs_reconstruct_loss + (slow_evol_loss + obs_evol_loss) / n
            all_loss.backward()
            optimizer.step()

            t4 = time.time()
            
            if is_print and not only_extract:
                train_loader.set_description(f'Tau[{tau_s}] | Epoch[{epoch}/{learn_max_epoch}] | Train | Extract: embed={embed_reconstruct_loss:.4f}, obs={obs_reconstruct_loss:.4f} | Evolve: slow={slow_evol_loss:.4f}, obs={obs_evol_loss:.4f}, t={(t3-t2):.2f}s | BP: t={(t4-t3):.2f}s')
            elif is_print and only_extract:
                train_loader.set_description(f'Tau[{tau_s}] | Epoch[{epoch}/{learn_max_epoch}] | Train | Extract: embed={embed_reconstruct_loss:.4f}, obs={obs_reconstruct_loss:.4f}, t={(t2-t1):.2f}s | BP: t={(t4-t2):.2f}s')

        if epoch % 10 == 0 or epoch in [1, learn_max_epoch]:

            # validate 
            with torch.no_grad():
                inputs = []
                uv_inputs = []
                slow_vars = []
                targets = []
                recons_obses = []
                embeds = []
                recons_embeds = []
                slow_obses_next = []
                total_obses_next = []
                
                save_batch_num = 2*int(np.ceil(horizon/stride_t / batch_size))
                count = 0
                predict_length = int(n*tau_unit)

                model.eval()
                for input, _, internl_units in val_loader:
                    
                    if 'HalfMoon' in system:
                        uv_input = input.to(device)[..., obs_dim:]
                    
                    input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,channel_num,feature_dim)
                    target = model.scale(internl_units[n-1].to(device))[..., :obs_dim]
                    
                    slow_var, embed = model.obs2slow(input)
                    recons_obs, recons_embed = model.slow2obs(slow_var)

                    if not only_extract:
                        
                        unit_fast_obs_list = [input-recons_obs]  # fast obs evolve
                        unit_slow_var_list = [slow_var]  # slow var evolve
                        unit_slow_obs_list = [recons_obs]  # slow obs evolve
                        t = torch.tensor([0., tau_unit], device=device)
                        for _ in range(n):
                            
                            if submodel == 'neural_ode':
                                unit_slow_var_list.append(model.node_s_evolve(unit_slow_var_list[-1], t, dt=ode_dt*rho))
                                
                                # fast
                                if fast:
                                    unit_fast_obs_list.append(model.node_f_evolve(unit_fast_obs_list[-1], unit_slow_var_list[-2], t, fast_dt=ode_dt, slow_dt=ode_dt*rho))
                                else:
                                    unit_fast_obs_list.append(torch.zeros_like(unit_fast_obs_list[-1]))
                                unit_slow_obs_list.append(model.slow2obs(unit_slow_var_list[-1])[0])
                        
                        fast_obs_next = unit_fast_obs_list[-1]
                        slow_obs_next = unit_slow_obs_list[-1]

                        # total obs evolve
                        total_obs_next = slow_obs_next + fast_obs_next

                    # record results
                    if count <= save_batch_num:
                        count += 1
                        inputs.append(input)
                        slow_vars.append(slow_var)
                        embeds.append(embed)
                        recons_obses.append(recons_obs)
                        recons_embeds.append(recons_embed)
                        
                        if 'HalfMoon' in system:
                            uv_inputs.append(uv_input)

                        if not only_extract:
                            targets.append(target)
                            slow_obses_next.append(slow_obs_next)
                            total_obses_next.append(total_obs_next)
                    else:
                        break
                
                # trans to tensor
                inputs = torch.cat(inputs, dim=0)
                slow_vars = torch.cat(slow_vars, dim=0)
                recons_obses = torch.cat(recons_obses, dim=0)
                embeds = torch.cat(embeds, dim=0)
                recons_embeds = torch.cat(recons_embeds, dim=0)
                if 'HalfMoon' in system:
                    uv_inputs = torch.cat(uv_inputs, dim=0)
                if not only_extract:
                    targets = torch.cat(targets, dim=0)
                    slow_obses_next = torch.cat(slow_obses_next, dim=0)
                    total_obses_next = torch.cat(total_obses_next, dim=0)
                    
                # loss
                embed_reconstruct_loss = MSE_loss(recons_embeds, embeds)
                obs_reconstruct_loss = MSE_loss(recons_obses, inputs)
                obs_evol_loss = MSE_loss(total_obses_next, targets) if not only_extract else 0.0
                if is_print: print(f'Tau[{tau_s}] | Epoch[{epoch}/{learn_max_epoch}] | Val | Extract: emebd={embed_reconstruct_loss:.5f}, obs={obs_reconstruct_loss:.5f} | Evolve: obs={obs_evol_loss:.5f}\n')
                
                # record loss
                if not only_extract:
                    val_loss.append([obs_evol_loss.item()])

                # time-lagged outputs
                time_lagged_outputs = time_lagged.forward(inputs)[0]
                time_lagged_outputs = model.descale(time_lagged_outputs)

                # descale
                inputs = model.descale(inputs)
                recons_obses = model.descale(recons_obses)
                
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)

                # plot slow variable
                wid = round(np.sqrt(slow_dim))
                hig = int(np.ceil(slow_dim/wid))
                plt.figure(figsize=(4*wid, 4*hig))
                plt.title('Slow variable Curve')
                for id_var in range(slow_dim):
                    ax = plt.subplot(hig, wid, id_var+1)
                    ax.plot(np.linspace(start_t, end_t, len(slow_vars[::10].cpu().numpy())), slow_vars[::10, id_var].cpu().numpy(), label=f'U{id_var+1}', c=colors[0])
                    ax.set_xlabel('t')
                    ax.legend()
                plt.subplots_adjust(wspace=0.3, hspace=0.3)
                plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_variable.png", dpi=150)
                plt.close()

                # plot observation and reconstruction
                end = end_t if horizon is None else horizon
                if 'FHN' in system:
                    
                    width = 6.5 if 'FHNv' in system else 6
                    
                    plt.figure(figsize=(12,22))
                    ax1 = plt.subplot(421)
                    ax1.set_xlabel(r'$i$')
                    ax1.set_ylabel(r'$t$')
                    ax1.set_title('data u')
                    im1 = ax1.imshow(inputs.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, inputs.shape[-1], start_t, end])
                    plt.colorbar(im1, ax=ax1)
                    ax2 = plt.subplot(422)
                    ax2.set_title('data v')
                    ax2.set_xlabel(r'$i$')
                    ax2.set_ylabel(r'$t$')
                    im2 = ax2.imshow(inputs.cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, inputs.shape[-1], start_t, end])
                    plt.colorbar(im2, ax=ax2)
                    ax3 = plt.subplot(423)
                    ax3.set_xlabel(r'$i$')
                    ax3.set_ylabel(r'$t$')
                    ax3.set_title('time-lagged u')
                    im3 = ax3.imshow(time_lagged_outputs.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, time_lagged_outputs.shape[-1], start_t+predict_length, end+predict_length])
                    plt.colorbar(im3, ax=ax3)
                    ax4 = plt.subplot(424)
                    ax4.set_title('time-lagged v')
                    ax4.set_xlabel(r'$i$')
                    ax4.set_ylabel(r'$t$')
                    im4 = ax4.imshow(time_lagged_outputs.cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, time_lagged_outputs.shape[-1], start_t+predict_length, end+predict_length])
                    plt.colorbar(im4, ax=ax4)
                    ax5 = plt.subplot(425)
                    ax5.set_title('slow u')
                    ax5.set_xlabel(r'$i$')
                    ax5.set_ylabel(r'$t$')
                    im5 = ax5.imshow(recons_obses.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, inputs.shape[-1], start_t, end])
                    plt.colorbar(im5, ax=ax5)
                    ax6 = plt.subplot(426)
                    ax6.set_title('slow v')
                    ax6.set_xlabel(r'$i$')
                    ax6.set_ylabel(r'$t$')
                    im6 = ax6.imshow(recons_obses.cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, inputs.shape[-1], start_t, end])
                    plt.colorbar(im6, ax=ax6)
                    ax7 = plt.subplot(427)
                    ax7.set_title('fast u')
                    ax7.set_xlabel(r'$i$')
                    ax7.set_ylabel(r'$t$')
                    im7 = ax7.imshow((recons_obses-inputs).cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, inputs.shape[-1], start_t, end])
                    plt.colorbar(im7, ax=ax7)
                    ax8 = plt.subplot(428)
                    ax8.set_title('fast v')
                    ax8.set_xlabel(r'$i$')
                    ax8.set_ylabel(r'$t$')
                    im8 = ax6.imshow((recons_obses-inputs).cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, inputs.shape[-1], start_t, end])
                    plt.colorbar(im8, ax=ax8)
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_obs.png", dpi=150)
                    plt.close()

                    plt.figure(figsize=(width,6))
                    ax1 = plt.subplot(111)
                    ax1.set_xlabel(r'$i$')
                    ax1.set_ylabel(r'$t$')
                    im1 = ax1.imshow(inputs.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, inputs.shape[-1], start_t, end])
                    plt.colorbar(im1, ax=ax1)
                    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_true.pdf", dpi=300)
                    plt.close()
                    
                    plt.figure(figsize=(width,6))
                    ax1 = plt.subplot(111)
                    ax1.set_xlabel(r'$i$')
                    ax1.set_ylabel(r'$t$')
                    im1 = ax1.imshow(recons_obses.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, inputs.shape[-1], start_t, end])
                    plt.colorbar(im1, ax=ax1)
                    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_slow.pdf", dpi=300)
                    plt.close()
                    
                    plt.figure(figsize=(width,6))
                    ax1 = plt.subplot(111)
                    ax1.set_xlabel(r'$i$')
                    ax1.set_ylabel(r'$t$')
                    im1 = ax1.imshow((recons_obses-inputs).cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, inputs.shape[-1], start_t, end])
                    plt.colorbar(im1, ax=ax1)
                    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_fast.pdf", dpi=300)
                    plt.close()
                    
                    np.savez(log_dir+f"/val/epoch-{epoch}/recons_obs.npz", inputs=inputs.cpu().numpy(), recons_obses=recons_obses.cpu().numpy(), time_lagged_outputs=time_lagged_outputs.cpu().numpy())

                elif 'HalfMoon_2D' in system:
                    plt.figure(figsize=(20,10))
                    for i, ylabel in enumerate(['x', 'y']):
                        ax = plt.subplot(2,4,i+1)
                        ax.plot(inputs.cpu().numpy()[:,0,0,i], label='true', c=colors[0])
                        ax.plot(recons_obses.cpu().numpy()[:,0,0,i], label='slow', c=colors[1])
                        ax.set_ylabel(ylabel)
                        ax.legend()
                        ax = plt.subplot(2,4,i+1+4)
                        ax.plot((inputs-recons_obses).cpu().numpy()[:,0,0,i], c=colors[2])
                        ax.set_ylabel('fast  '+ylabel)
                    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_obs.png", dpi=150)
                    plt.close()

                    plt.figure(figsize=(6,6))
                    ax1 = plt.subplot(221)
                    ax1.scatter(inputs.cpu().numpy()[:,0,0,0], slow_vars[:, 0], c=colors[0])
                    ax1.set_xlabel('x')
                    ax1.set_ylabel('z')
                    ax1 = plt.subplot(222)
                    ax1.scatter(inputs.cpu().numpy()[:,0,0,1], slow_vars[:, 0], c=colors[1])
                    ax1.set_xlabel('y')
                    ax1.set_ylabel('z')
                    ax1 = plt.subplot(223)
                    ax1.scatter(uv_inputs.cpu().numpy()[:,0,0,0], slow_vars[:, 0], c=colors[2])
                    ax1.set_xlabel('u')
                    ax1.set_ylabel('z')
                    ax1 = plt.subplot(224)
                    ax1.scatter(uv_inputs.cpu().numpy()[:,0,0,1], slow_vars[:, 0], c=colors[3])
                    ax1.set_xlabel('v')
                    ax1.set_ylabel('z')
                    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_obs.png", dpi=150)
                    plt.close()

                    plt.figure(figsize=(6,6))
                    ax1 = plt.subplot(111)
                    ax1.scatter(inputs.cpu().numpy()[:,0,0,0], slow_vars[:, 0], c=colors[0], s=11)
                    ax1.set_xlabel('x')
                    ax1.set_ylabel('z')
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/xz.pdf", dpi=300)
                    plt.close()
                    plt.figure(figsize=(6,6))
                    ax1 = plt.subplot(111)
                    ax1.scatter(inputs.cpu().numpy()[:,0,0,1], slow_vars[:, 0], c=colors[1], s=11)
                    ax1.set_xlabel('y')
                    ax1.set_ylabel('z')
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/yz.pdf", dpi=300)
                    plt.close()
                    plt.figure(figsize=(6,6))
                    ax1 = plt.subplot(111)
                    ax1.scatter(uv_inputs.cpu().numpy()[:,0,0,0], slow_vars[:, 0], c=colors[2], s=11)
                    ax1.set_xlabel('u')
                    ax1.set_ylabel('z')
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/uz.pdf", dpi=300)
                    plt.close()
                    plt.figure(figsize=(6,6))
                    ax1 = plt.subplot(111)
                    ax1.scatter(uv_inputs.cpu().numpy()[:,0,0,1], slow_vars[:, 0], c=colors[3], s=11)
                    ax1.set_xlabel('v')
                    ax1.set_ylabel('z')
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/vz.pdf", dpi=300)
                    plt.close()
                    
                    np.savez(log_dir+f"/val/epoch-{epoch}/recons_obs.npz", inputs=inputs.cpu().numpy(), uv_inputs=uv_inputs.cpu().numpy(), slow_vars=slow_vars.cpu().numpy())
                    
                elif 'Coupled_Lorenz' in system:
                    plt.figure(figsize=(20,20))
                    for idx in range(2):
                        for i, ylabel in enumerate(['x', 'y', 'z']):
                            ax = plt.subplot(4,3,idx*6+i+1)
                            ax.plot(inputs.cpu().numpy()[:,0,i,idx], label='true', c=colors[0])
                            ax.plot(recons_obses.cpu().numpy()[:,0,i,idx], label='slow', c=colors[1])
                            ax.set_ylabel(ylabel + str(idx+1))
                            ax.legend()
                            ax = plt.subplot(4,3,idx*6+i+1+3)
                            ax.plot((inputs-recons_obses).cpu().numpy()[:,0,i,idx], c=colors[2])
                            ax.set_ylabel('fast  '+ylabel + str(idx+1))
                    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                    plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/recons_obs.png", dpi=150)
                    plt.close()

                    np.savez(log_dir+f"/val/epoch-{epoch}/recons_obs.npz", inputs=inputs.cpu().numpy(), recons_obses=recons_obses.cpu().numpy())
                    
                if not only_extract:
                    # descale
                    targets = model.descale(targets)
                    slow_obses_next = model.descale(slow_obses_next)
                    total_obses_next = model.descale(total_obses_next)
                    
                    # plot prediction vs true
                    if 'FHN' in system:
                        plt.figure(figsize=(12,18))
                        ax1 = plt.subplot(321)
                        ax1.set_xlabel(r'$i$')
                        ax1.set_ylabel(r'$t$')
                        ax1.set_title('true u')
                        im1 = ax1.imshow(targets.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im1, ax=ax1)
                        ax2 = plt.subplot(322)
                        ax2.set_title('true v')
                        ax2.set_xlabel(r'$i$')
                        ax2.set_ylabel(r'$t$')
                        im2 = ax2.imshow(targets.cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im2, ax=ax2)
                        ax3 = plt.subplot(323)
                        ax3.set_xlabel(r'$i$')
                        ax3.set_ylabel(r'$t$')
                        ax3.set_title('pred u')
                        im3 = ax3.imshow(total_obses_next.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, total_obses_next.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im3, ax=ax3)
                        ax4 = plt.subplot(324)
                        ax4.set_title('pred v')
                        ax4.set_xlabel(r'$i$')
                        ax4.set_ylabel(r'$t$')
                        im4 = ax4.imshow(total_obses_next.cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, total_obses_next.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im4, ax=ax4)
                        ax5 = plt.subplot(325)
                        ax5.set_title('Absolute Error u')
                        ax5.set_xlabel(r'$i$')
                        ax5.set_ylabel(r'$t$')
                        im5 = ax5.imshow(np.abs((total_obses_next-targets).cpu().numpy())[::-1,0,0,:], aspect='auto', vmin=0, vmax=1, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im5, ax=ax5)
                        ax6 = plt.subplot(326)
                        ax6.set_title('Absolute Error v')
                        ax6.set_xlabel(r'$i$')
                        ax6.set_ylabel(r'$t$')
                        im6 = ax6.imshow(np.abs((total_obses_next-targets).cpu().numpy())[::-1,0,1,:], aspect='auto', vmin=0, vmax=1, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im6, ax=ax6)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/prediction.png", dpi=150)
                        plt.close()

                        plt.figure(figsize=(12,18))
                        ax1 = plt.subplot(321)
                        ax1.set_xlabel(r'$i$')
                        ax1.set_ylabel(r'$t$')
                        ax1.set_title('pred u')
                        im1 = ax1.imshow(total_obses_next.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im1, ax=ax1)
                        ax2 = plt.subplot(322)
                        ax2.set_title('pred v')
                        ax2.set_xlabel(r'$i$')
                        ax2.set_ylabel(r'$t$')
                        im2 = ax2.imshow(total_obses_next.cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im2, ax=ax2)
                        ax3 = plt.subplot(323)
                        ax3.set_xlabel(r'$i$')
                        ax3.set_ylabel(r'$t$')
                        ax3.set_title('pred slow-u')
                        im3 = ax3.imshow(slow_obses_next.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im3, ax=ax3)
                        ax4 = plt.subplot(324)
                        ax4.set_title('pred slow-v')
                        ax4.set_xlabel(r'$i$')
                        ax4.set_ylabel(r'$t$')
                        im4 = ax4.imshow(slow_obses_next.cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im4, ax=ax4)
                        ax5 = plt.subplot(325)
                        ax5.set_title('pred fast-u')
                        ax5.set_xlabel(r'$i$')
                        ax5.set_ylabel(r'$t$')
                        im5 = ax5.imshow((total_obses_next-slow_obses_next).cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im5, ax=ax5)
                        ax6 = plt.subplot(326)
                        ax6.set_title('pred fast-v')
                        ax6.set_xlabel(r'$i$')
                        ax6.set_ylabel(r'$t$')
                        im6 = ax6.imshow((total_obses_next-slow_obses_next).cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im6, ax=ax6)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/pred_slow_fast.png", dpi=150)
                        plt.close()

                        plt.figure(figsize=(6,6))
                        ax1 = plt.subplot(111)
                        ax1.set_xlabel(r'$i$')
                        ax1.set_ylabel(r'$t$')
                        im1 = ax1.imshow(targets.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im1, ax=ax1)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/pred_true.pdf", dpi=300)
                        plt.close()
                        plt.figure(figsize=(6,6))
                        ax1 = plt.subplot(111)
                        ax1.set_xlabel(r'$i$')
                        ax1.set_ylabel(r'$t$')
                        im1 = ax1.imshow(total_obses_next.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im1, ax=ax1)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/pred_pred.pdf", dpi=300)
                        plt.close()
                        plt.figure(figsize=(6,6))
                        ax1 = plt.subplot(111)
                        ax1.set_xlabel(r'$i$')
                        ax1.set_ylabel(r'$t$')
                        im1 = ax1.imshow(np.abs((total_obses_next-targets).cpu().numpy())[::-1,0,0,:], aspect='auto', vmin=0, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im1, ax=ax1)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/pred_error.pdf", dpi=300)
                        plt.close()
                        plt.figure(figsize=(6,6))
                        ax1 = plt.subplot(111)
                        ax1.set_xlabel(r'$i$')
                        ax1.set_ylabel(r'$t$')
                        im1 = ax1.imshow(slow_obses_next.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im1, ax=ax1)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/pred_slow.pdf", dpi=300)
                        plt.close()
                        plt.figure(figsize=(6,6))
                        ax1 = plt.subplot(111)
                        ax1.set_xlabel(r'$i$')
                        ax1.set_ylabel(r'$t$')
                        im1 = ax1.imshow((total_obses_next-slow_obses_next).cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+predict_length, end+predict_length])
                        plt.colorbar(im1, ax=ax1)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/pred_fast.pdf", dpi=300)
                        plt.close()

                    elif 'HalfMoon_2D' in system:
                        plt.figure(figsize=(10,5))
                        for i, ylabel in enumerate(['x', 'y']):
                            ax = plt.subplot(1,2,i+1)
                            ax.plot(targets.cpu().numpy()[:,0,0,i], label='true', c=colors[0])
                            ax.plot(total_obses_next.cpu().numpy()[:,0,0,i], label='pred', c=colors[1])
                            ax.set_ylabel(ylabel)
                            ax.legend()
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/prediction.png", dpi=150)
                        plt.close()
                        
                        plt.figure(figsize=(10,15))
                        for i, ylabel in enumerate(['x', 'y']):
                            ax = plt.subplot(3,2,i+1)
                            ax.plot(total_obses_next.cpu().numpy()[:,0,0,i], c=colors[0])
                            ax.set_ylabel(ylabel)
                            ax = plt.subplot(3,2,i+1+2)
                            ax.plot(slow_obses_next.cpu().numpy()[:,0,0,i], c=colors[1])
                            ax.set_ylabel('slow'+ylabel)
                            ax = plt.subplot(3,2,i+1+2*2)
                            ax.plot((total_obses_next-slow_obses_next).cpu().numpy()[:,0,0,i], c=colors[2])
                            ax.set_ylabel('fast'+ylabel)
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/pred_slow_fast.png", dpi=150)
                        plt.close()
                        
                        plt.figure(figsize=(10,15))
                        plt.scatter(slow_obses_next.cpu().numpy()[:,0,0,0], slow_obses_next.cpu().numpy()[:,0,0,1])
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_phase.png", dpi=150)
                        plt.close()
                    
                    elif 'Coupled_Lorenz' in system:
                        plt.figure(figsize=(20,10))
                        for idx in range(2):
                            for i, ylabel in enumerate(['x', 'y', 'z']):
                                ax = plt.subplot(2,3,idx*3+i+1)
                                ax.plot(targets.cpu().numpy()[:,0,i,idx], label='true', c=colors[0])
                                ax.plot(total_obses_next.cpu().numpy()[:,0,i,idx], label='pred', c=colors[1])
                                ax.set_ylabel(ylabel + str(idx+1))
                                if idx==0 and i==0: ax.legend(frameon=False)
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/prediction.png", dpi=150)
                        plt.close()
                        
                        plt.figure(figsize=(24,12))
                        for idx in range(2):
                            for i, ylabel in enumerate(['x', 'y', 'z']):
                                ax = plt.subplot(6,3,idx*9+i+1)
                                ax.plot(total_obses_next.cpu().numpy()[:,0,i,idx], c=colors[0])
                                ax.set_ylabel(ylabel + str(idx+1))
                                ax = plt.subplot(6,3,idx*9+i+3+1)
                                ax.plot(slow_obses_next.cpu().numpy()[:,0,i,idx], c=colors[1])
                                ax.set_ylabel('slow'+ylabel + str(idx+1))
                                ax = plt.subplot(6,3,idx*9+i+6+1)
                                ax.plot((total_obses_next-slow_obses_next).cpu().numpy()[:,0,i,idx], c=colors[2])
                                ax.set_ylabel('fast'+ylabel + str(idx+1))
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/pred_slow_fast.png", dpi=150)
                        plt.close()
                
        # save model
        if epoch == learn_max_epoch:
            torch.save(model.state_dict(), log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
        
    # plot loss curve
    val_loss = np.array(val_loss)
    plt.figure(figsize=(6,6))
    for i, item in enumerate(['obs_evol_loss']):
        plt.plot(val_loss[:, i], label=item, c=colors[i])
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.tight_layout(); plt.savefig(log_dir+'/val_loss_curve.png', dpi=150)


def test_sfs(
        system,
        submodel,
        embedding_dim,
        channel_num,
        obs_dim,
        tau_s, 
        ckpt_epoch, 
        slow_dim,
        koopman_dim, 
        tau_unit, 
        n,
        n_list,
        is_print=False, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/LearnDynamics/', 
        device='cpu',
        data_dim=4,
        batch_size=128,
        enc_net='MLP',
        e1_layer_n=3,
        dt=0.001,
        total_t=0.9,
        start_t=0.0,
        end_t=None,
        stride_t=0.01,
        tau_1=0.1,
        horizon=None,
        sliding_length=None,
        fast=1,
        rho=1,
        mask_slow=0,
        test_horizon=None,
        predict_n=None,
        sync=True,
        inter_p='nearest_neighbour',
        num_heads=1
        ):
        
    # prepare
    data_filepath = data_dir + f'st{start_t}_et{end_t if end_t else total_t}/' + 'tau_' + str(tau_unit)
    log_dir = log_dir + f'seed{random_seed}'
    os.makedirs(log_dir+f"/test/", exist_ok=True)

    # load model
    model = models.Slow_Fast_Synergetic_ODE(in_channels=channel_num, feature_dim=obs_dim, embed_dim=embedding_dim, slow_dim=slow_dim, redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, tau_1=tau_1, device=device, data_dim=data_dim, enc_net=enc_net, e1_layer_n=e1_layer_n, sync=sync, inter_p=inter_p, num_heads=num_heads)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    # dataset
    test_dataset = Dataset(data_filepath, 'test', length=n, horizon=horizon)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # result
    pred_per_tau = []
    true_per_tau = []
    slow_per_tau = []

    def draw(true, pred, slow, k):

        start = start_t + k*tau_unit
        points_num = int(np.ceil(test_horizon/stride_t))
        end_t = start + predict_n * tau_unit

        if 'FHN' in system:
            
            width = 6.5 if 'FHNv' in system else 6
            
            plt.figure(figsize=(13,18))
            ax1 = plt.subplot(321)
            ax1.set_xlabel(r'$i$')
            ax1.set_ylabel(r'$t$')
            ax1.set_title('true u')
            im1 = ax1.imshow(true[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im1, ax=ax1)
            ax2 = plt.subplot(322)
            ax2.set_title('true v')
            ax2.set_xlabel(r'$i$')
            ax2.set_ylabel(r'$t$')
            im2 = ax2.imshow(true[:points_num][::-1,k,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im2, ax=ax2)
            ax3 = plt.subplot(323)
            ax3.set_xlabel(r'$i$')
            ax3.set_ylabel(r'$t$')
            ax3.set_title('pred u')
            im3 = ax3.imshow(pred[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, pred.shape[-1], start, end_t])
            plt.colorbar(im3, ax=ax3)
            ax4 = plt.subplot(324)
            ax4.set_title('pred v')
            ax4.set_xlabel(r'$i$')
            ax4.set_ylabel(r'$t$')
            im4 = ax4.imshow(pred[:points_num][::-1,k,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, pred.shape[-1], start, end_t])
            plt.colorbar(im4, ax=ax4)
            ax5 = plt.subplot(325)
            ax5.set_title('Absolute Error u')
            ax5.set_xlabel(r'$i$')
            ax5.set_ylabel(r'$t$')
            im5 = ax5.imshow(np.abs(true-pred)[:points_num][::-1,k,0,:], aspect='auto', vmin=-1, vmax=1, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im5, ax=ax5)
            ax6 = plt.subplot(326)
            ax6.set_title('Absolute Error v')
            ax6.set_xlabel(r'$i$')
            ax6.set_ylabel(r'$t$')
            im6 = ax6.imshow(np.abs(true-pred)[:points_num][::-1,k,1,:], aspect='auto', vmin=-1, vmax=1, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im6, ax=ax6)
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.1f}_prediction.png", dpi=150)
            plt.close()

            plt.figure(figsize=(13,18))
            ax1 = plt.subplot(321)
            ax1.set_xlabel(r'$i$')
            ax1.set_ylabel(r'$t$')
            ax1.set_title('pred u')
            im1 = ax1.imshow(pred[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im1, ax=ax1)
            ax2 = plt.subplot(322)
            ax2.set_title('pred v')
            ax2.set_xlabel(r'$i$')
            ax2.set_ylabel(r'$t$')
            im2 = ax2.imshow(pred[:points_num][::-1,k,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im2, ax=ax2)
            ax3 = plt.subplot(323)
            ax3.set_xlabel(r'$i$')
            ax3.set_ylabel(r'$t$')
            ax3.set_title('pred slow u')
            im3 = ax3.imshow(slow[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, pred.shape[-1], start, end_t])
            plt.colorbar(im3, ax=ax3)
            ax4 = plt.subplot(324)
            ax4.set_title('pred slow v')
            ax4.set_xlabel(r'$i$')
            ax4.set_ylabel(r'$t$')
            im4 = ax4.imshow(slow[:points_num][::-1,k,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, pred.shape[-1], start, end_t])
            plt.colorbar(im4, ax=ax4)
            ax5 = plt.subplot(325)
            ax5.set_title('pred fast u')
            ax5.set_xlabel(r'$i$')
            ax5.set_ylabel(r'$t$')
            im5 = ax5.imshow((pred-slow)[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im5, ax=ax5)
            ax6 = plt.subplot(326)
            ax6.set_title('pred fast v')
            ax6.set_xlabel(r'$i$')
            ax6.set_ylabel(r'$t$')
            im6 = ax6.imshow((pred-slow)[:points_num][::-1,k,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im6, ax=ax6)
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_slow_fast.png", dpi=150)
            plt.close()

            plt.figure(figsize=(width,6))
            ax1 = plt.subplot(111)
            ax1.set_xlabel(r'$i$')
            ax1.set_ylabel(r'$t$')
            im1 = ax1.imshow(true[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im1, ax=ax1)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_pred_true.pdf", dpi=300)
            plt.close()
            
            plt.figure(figsize=(width,6))
            ax1 = plt.subplot(111)
            ax1.set_xlabel(r'$i$')
            ax1.set_ylabel(r'$t$')
            im1 = ax1.imshow(pred[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im1, ax=ax1)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_pred_pred.pdf", dpi=300)
            plt.close()
            
            plt.figure(figsize=(width,6))
            ax1 = plt.subplot(111)
            ax1.set_xlabel(r'$i$')
            ax1.set_ylabel(r'$t$')
            im1 = ax1.imshow(np.abs(true-pred)[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im1, ax=ax1)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_pred_error.pdf", dpi=300)
            plt.close()
            
            plt.figure(figsize=(width,6))
            ax1 = plt.subplot(111)
            ax1.set_xlabel(r'$i$')
            ax1.set_ylabel(r'$t$')
            im1 = ax1.imshow(slow[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im1, ax=ax1)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_pred_slow.pdf", dpi=300)
            plt.close()
            
            plt.figure(figsize=(width,6))
            ax1 = plt.subplot(111)
            ax1.set_xlabel(r'$i$')
            ax1.set_ylabel(r'$t$')
            im1 = ax1.imshow((pred-slow)[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im1, ax=ax1)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_pred_fast.pdf", dpi=300)
            plt.close()

        elif 'HalfMoon_2D' in system:
            plt.figure(figsize=(20,5))
            for i, ylabel in enumerate(['x', 'y']):
                ax = plt.subplot(1,4,i+1)
                ax.plot(true[:,0,0,i], label='true', c=colors[0])
                ax.plot(pred[:,0,0,i], label='pred', c=colors[1])
                ax.set_ylabel(ylabel)
                ax.legend()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.1f}_prediction.png", dpi=150)
            plt.close()
    
    # testing pipeline  
    if 'FHN' in data_filepath:
        if 'FHNv' in data_filepath:
            ode_dt = 1e-1
        else:
            ode_dt = 1e-3
    elif 'HalfMoon' in data_filepath:
        ode_dt = 1.0
    elif 'Coupled_Lorenz' in data_filepath:
        ode_dt = 1e-2
          
    with torch.no_grad():
        
        model.eval()
        iter = tqdm(test_loader) if is_print else test_loader
        time_cost = []
        
        for input, _, internl_units in iter:
            
            input = model.scale(input.to(device))[..., :obs_dim]
        
            # obs —— embedding —— slow representation
            slow_var, _ = model.obs2slow(input)
            recons_obs, _ = model.slow2obs(slow_var)

            # slow & fast evolve by neural ode
            unit_fast_obs_list = [input-recons_obs]  # fast obs evolve
            unit_slow_var_list = [slow_var]  # slow var evolve
            unit_slow_obs_list = [recons_obs]  # slow obs evolve
            pred_total_obs_list = [input] # total obs evolve
            true_total_obs_list = [input] # total obs ground-true
            t = torch.tensor([0., tau_unit], device=device)
            
            if submodel == 'neural_ode':
            
                step_n = 1
                if ode_dt*rho <= tau_unit:
                    slow_t = torch.tensor([0., tau_unit], device=device)
                    slow_n = 1
                else:
                    t_step = ode_dt*rho - ode_dt*rho % tau_unit + tau_unit
                    slow_t = torch.tensor([0., t_step], device=device)
                    slow_n = int(t_step / tau_unit)

                t0 = time.time()
                time_cost_iter = []
                for i in range(n):
                    # slow
                    if step_n % slow_n == 0:
                        unit_slow_var_list.append(model.node_s_evolve(unit_slow_var_list[-1], slow_t, ode_dt*rho))
                    else:
                        unit_slow_var_list.append(unit_slow_var_list[-1])
                    step_n += 1
                        
                    # fast
                    if fast:
                        if mask_slow:
                            unit_fast_obs_list.append(model.node_f_evolve(unit_fast_obs_list[-1], t, ode_dt, slow=torch.zeros_like(unit_slow_obs_list[-1])))
                        else:
                            unit_fast_obs_list.append(model.node_f_evolve(unit_fast_obs_list[-1], t, ode_dt, slow=unit_slow_obs_list[-1]))
                    else:
                        unit_fast_obs_list.append(torch.zeros_like(unit_fast_obs_list[-1]))
                    
                    unit_slow_obs_list.append(model.slow2obs(unit_slow_var_list[-1])[0])
                    pred_total_obs_list.append(unit_slow_obs_list[-1]+unit_fast_obs_list[-1])
                    true_total_obs_list.append(internl_units[i][...,:obs_dim]) # not scale
                    
                    t1 = time.time()
                    time_cost_iter.append(t1-t0)
            

            time_cost.append(time_cost_iter)
            
            # record results
            pred_per_tau.append(torch.cat(pred_total_obs_list, dim=1))
            true_per_tau.append(torch.cat(true_total_obs_list, dim=1))
            slow_per_tau.append(torch.cat(unit_slow_obs_list, dim=1))
        
        pred_per_tau = torch.cat(pred_per_tau, dim=0)
        slow_per_tau = torch.cat(slow_per_tau, dim=0)
        pred_per_tau = pred_per_tau * (model.max-model.min+1e-6) + model.min
        slow_per_tau = slow_per_tau * (model.max-model.min+1e-6) + model.min
        true_per_tau = torch.cat(true_per_tau, dim=0)

    # metrics
    pred = pred_per_tau.detach().cpu().numpy()
    true = true_per_tau.detach().cpu().numpy()
    slow = slow_per_tau.detach().cpu().numpy()
    
    nmse_per_tau = []
    for i in range(1, n+1):
        pred_tau_i = pred[:,i]
        true_tau_i = true[:,i]

        num = int(pred_tau_i.shape[0]/50)
        tmp = []
        for j in range(50):
            true_group = true_tau_i[num*j:num*(j+1)]
            pred_group = pred_tau_i[num*j:num*(j+1)]
            tmp.append(np.mean((true_group-pred_group)**2)/np.var(true_group))
        nmse_per_tau.append(tmp)

        if i in n_list:
            draw(true, pred, slow, i)
    
    return nmse_per_tau, np.mean(time_cost, axis=0)