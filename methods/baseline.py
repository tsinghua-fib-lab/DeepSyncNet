# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from Data.dataset import Dataset

from util import *


def baseline_train(
        model,
        obs_dim,
        data_dim,
        channel_num,
        tau_s, 
        tau_unit, 
        is_print=False, 
        random_seed=729,
        max_epoch=50,
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/lstm/', 
        device='cpu',
        lr=0.01,
        batch_size=128,
        dt=0.001,
        total_t=0.9,
        start_t=0.0,
        end_t=None,
        stride_t=None,
        horizon=None,
        sliding_length=None,
        learn_n=None
        ):
        
    # prepare
    data_filepath = data_dir + f'st{start_t}_et{end_t if end_t else total_t}/' + 'tau_' + str(tau_unit)
    log_dir = log_dir + f'tau_{tau_s}/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    scale_path = data_filepath.replace(f'/tau_{tau_unit}', '')
    model.min = torch.from_numpy(np.loadtxt(scale_path+"/data_min.txt").reshape(channel_num,data_dim).astype(np.float32))
    model.max = torch.from_numpy(np.loadtxt(scale_path+"/data_max.txt").reshape(channel_num,data_dim).astype(np.float32))
    model.to(device)
    
    # training params
    weight_decay = 0.001
    MSE_loss = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # dataset
    train_dataset = Dataset(data_filepath, 'train', length=learn_n, horizon=horizon)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = Dataset(data_filepath, 'val', length=learn_n, horizon=horizon)
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
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        if is_print:
            from tqdm import tqdm
            train_loader = tqdm(train_loader)
        for input, _, internl_units in train_loader:

            # scale
            input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,channel_num,feature_dim)

            loss = 0
            output_list = [input]
            z_list = []
            h_list, c_list = [None], [None]
            t = torch.tensor([0., tau_unit], device=device)
            for i in range(1, len(internl_units)):
                
                unit = model.scale(internl_units[i].to(device))[..., :obs_dim] # t+i

                if model.__class__.__name__ == 'NeuralODE':
                    output_list.append(model(output_list[i-1], t, ode_dt)[:, -1:])
                    output = output_list[-1]
                    loss += MSE_loss(output, unit)
                elif model.__class__.__name__ == 'LED':
                    output = model(input, n=i)
                    loss += MSE_loss(output, unit)
                elif model.__class__.__name__ == 'DeepKoopman':
                    if i == 1:
                        pred_x, pred_z = model.evolve(input, n=1)
                    else:
                        pred_z = model.evolve_latent(z_list[-1], n=1)
                        pred_x = model.decoder(pred_z)
                    z_list.append(pred_z)
                    recons_x, true_z = model.encode_decode(unit)
                    loss += MSE_loss(unit, recons_x) + MSE_loss(pred_z, true_z) + MSE_loss(unit, pred_x)

            loss /= len(internl_units)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_print:
                train_loader.set_description(f'Tau[{tau_s}] | epoch[{epoch}/{max_epoch}] | train-mse={loss:.5f}')
        
        if epoch % 10 == 0 or epoch in [1, max_epoch]:
            
            # validate
            with torch.no_grad():
                targets = []
                outputs = []

                if horizon is None:
                    save_batch_num = int(np.ceil(((end_t if end_t else total_t)-start_t-tau_s)/stride_t / batch_size))
                else:
                    save_batch_num = int(np.ceil(horizon/stride_t / batch_size))
                count = 0
                
                model.eval()
                for input, target, _ in val_loader:
                    
                    input = model.scale(input.to(device))[..., :obs_dim] # (batchsize,1,channel_num,feature_dim)
                    target = model.scale(target.to(device))[..., :obs_dim]
                    
                    if model.__class__.__name__ == 'NeuralODE':
                        t = torch.tensor([0., (learn_n-1)*tau_unit], device=device)
                        output = model(input, t, ode_dt)[:, -1:]
                    elif model.__class__.__name__ == 'LED':
                        output = model(input, n=learn_n-1)
                    elif model.__class__.__name__ == 'DeepKoopman':
                        output, _ = model.evolve(input, n=learn_n-1)

                    # record results
                    if count <= save_batch_num:
                        count += 1
                        outputs.append(output)
                        targets.append(target)
                    else:
                        break
                
                # trans to tensor
                outputs = torch.cat(outputs, dim=0)
                targets = torch.cat(targets, dim=0)
                
                # cal loss
                loss = MSE_loss(outputs, targets)
                val_loss.append(loss.detach().item())
                if is_print: print(f'Tau[{tau_s}] | epoch[{epoch}/{max_epoch}] | val-mse={loss:.5f}\n')
                            
                # plot per 1 epoch
                if epoch % 1 == 0:
                    
                    os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)
                    
                    end = end_t if horizon is None else horizon

                    # plot total infomation one-step prediction curve
                    targets = model.descale(targets)
                    outputs = model.descale(outputs)
                    
                    if 'FHN' in data_filepath:
                        plt.figure(figsize=(13,18))
                        ax1 = plt.subplot(321)
                        ax1.set_xlabel(r'$i$')
                        ax1.set_ylabel(r'$t$')
                        ax1.set_title('true u')
                        im1 = ax1.imshow(targets.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+tau_s, end+tau_s])
                        plt.colorbar(im1, ax=ax1)
                        ax2 = plt.subplot(322)
                        ax2.set_title('true v')
                        ax2.set_xlabel(r'$i$')
                        ax2.set_ylabel(r'$t$')
                        im2 = ax2.imshow(targets.cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+tau_s, end+tau_s])
                        plt.colorbar(im2, ax=ax2)
                        ax3 = plt.subplot(323)
                        ax3.set_xlabel(r'$i$')
                        ax3.set_ylabel(r'$t$')
                        ax3.set_title('pred u')
                        im3 = ax3.imshow(outputs.cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, outputs.shape[-1], start_t+tau_s, end+tau_s])
                        plt.colorbar(im3, ax=ax3)
                        ax4 = plt.subplot(324)
                        ax4.set_title('pred v')
                        ax4.set_xlabel(r'$i$')
                        ax4.set_ylabel(r'$t$')
                        im4 = ax4.imshow(outputs.cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, outputs.shape[-1], start_t+tau_s, end+tau_s])
                        plt.colorbar(im4, ax=ax4)
                        ax5 = plt.subplot(325)
                        ax5.set_title('MAE u')
                        ax5.set_xlabel(r'$i$')
                        ax5.set_ylabel(r'$t$')
                        im5 = ax5.imshow((targets-outputs).cpu().numpy()[::-1,0,0,:], aspect='auto', vmin=-1, vmax=1, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+tau_s, end+tau_s])
                        plt.colorbar(im5, ax=ax5)
                        ax6 = plt.subplot(326)
                        ax6.set_title('MAE v')
                        ax6.set_xlabel(r'$i$')
                        ax6.set_ylabel(r'$t$')
                        im6 = ax6.imshow((targets-outputs).cpu().numpy()[::-1,0,1,:], aspect='auto', vmin=-1, vmax=1, cmap=my_cmap, extent=[0, targets.shape[-1], start_t+tau_s, end+tau_s])
                        plt.colorbar(im6, ax=ax6)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/prediction.png", dpi=200)
                        plt.close()
                    
                    elif 'HalfMoon_2D' in data_filepath:
                        plt.figure(figsize=(20,5))
                        for i, ylabel in enumerate(['x', 'y']):
                            ax = plt.subplot(1,4,i+1)
                            ax.plot(targets.cpu().numpy()[:,0,0,i], label='true', c=colors[0])
                            ax.plot(outputs.cpu().numpy()[:,0,0,i], label='pred', c=colors[1])
                            ax.set_ylabel(ylabel)
                            ax.legend()
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/prediction.png", dpi=150)
                        plt.close()
                    
                    elif 'Coupled_Lorenz' in data_filepath:
                        plt.figure(figsize=(20,10))
                        for idx in range(2):
                            for i, ylabel in enumerate(['x', 'y', 'z']):
                                ax = plt.subplot(2,3,idx*3+i+1)
                                ax.plot(targets.cpu().numpy()[:,0,i,idx], label='true', c=colors[0])
                                ax.plot(outputs.cpu().numpy()[:,0,i,idx], label='pred', c=colors[1])
                                ax.set_ylabel(ylabel + str(idx+1))
                                if idx==0 and i==0: ax.legend(frameon=False)
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                        plt.tight_layout(); plt.savefig(log_dir+f"/val/epoch-{epoch}/prediction.png", dpi=150)
                        plt.close()
                
        # save model
        if epoch==max_epoch:
            torch.save(model.state_dict(), log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    
    # plot loss curve
    plt.figure()
    plt.plot(val_loss)
    plt.xlabel('epoch')
    plt.tight_layout(); plt.savefig(log_dir+'/val_loss_curve.jpg', dpi=300)


def baseline_test(
        model,
        obs_dim,
        system,
        tau_s, 
        tau_unit, 
        n,
        n_list,
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/lstm/', 
        device='cpu',
        batch_size=128,
        dt=0.001,
        total_t=0.9,
        start_t=0.0,
        end_t=None,
        stride_t=None,
        max_epoch=None,
        horizon=None,
        sliding_length=None,
        test_horizon=None,
        predict_n=None
        ):
        
    # prepare
    data_filepath = data_dir + f'st{start_t}_et{end_t if end_t else total_t}/' + 'tau_' + str(tau_unit)
    log_dir = log_dir + f'tau_{tau_s}/seed{random_seed}'
    os.makedirs(log_dir+f"/test/", exist_ok=True)

    # load model
    ckpt_path = log_dir+f'/checkpoints/epoch-{max_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    # dataset
    test_dataset = Dataset(data_filepath, 'test', length=n, horizon=horizon)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # result
    pred_per_tau = []
    true_per_tau = []

    def draw(true, pred, k):
        
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
            im5 = ax5.imshow(np.abs(true-pred)[:points_num][::-1,k,0,:], aspect='auto', vmin=0, vmax=1, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im5, ax=ax5)
            ax6 = plt.subplot(326)
            ax6.set_title('Absolute Error v')
            ax6.set_xlabel(r'$i$')
            ax6.set_ylabel(r'$t$')
            im6 = ax6.imshow(np.abs(true-pred)[:points_num][::-1,k,1,:], aspect='auto', vmin=0, vmax=1, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im6, ax=ax6)
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_prediction.png", dpi=150)
            plt.close()

            plt.figure(figsize=(width,6))
            ax1 = plt.subplot(111)
            ax1.set_xlabel(r'$i$')
            ax1.set_ylabel(r'$t$')
            im1 = ax1.imshow(pred[:points_num][::-1,k,0,:], aspect='auto', vmin=-3, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im1, ax=ax1)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_pred.pdf", dpi=300)
            plt.close()
            
            plt.figure(figsize=(width,6))
            ax1 = plt.subplot(111)
            ax1.set_xlabel(r'$i$')
            ax1.set_ylabel(r'$t$')
            im1 = ax1.imshow(np.abs(true-pred)[:points_num][::-1,k,0,:], aspect='auto', vmin=0, vmax=3, cmap=my_cmap, extent=[0, true.shape[-1], start, end_t])
            plt.colorbar(im1, ax=ax1)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_error.pdf", dpi=300)
            plt.close()
            
        elif 'HalfMoon_2D' in system:
            plt.figure(figsize=(20,5))
            for i, ylabel in enumerate(['x', 'y']):
                ax = plt.subplot(1,4,i+1)
                ax.plot(true[:,k,0,i], label='true', c=colors[0])
                ax.plot(pred[:,k,0,i], label='pred', c=colors[1])
                ax.set_ylabel(ylabel)
                ax.legend()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.1f}_prediction.png", dpi=150)
            plt.close()
        
        elif 'Coupled_Lorenz' in data_filepath:
            plt.figure(figsize=(20,10))
            for idx in range(2):
                for i, ylabel in enumerate(['x', 'y', 'z']):
                    ax = plt.subplot(2,3,idx*3+i+1)
                    ax.plot(true[:points_num,k,i,idx], label='true', c=colors[0])
                    ax.plot(pred[:points_num,k,i,idx], label='pred', c=colors[1])
                    ax.set_ylabel(ylabel + str(idx+1))
                    if idx==0 and i==0: ax.legend(frameon=False)
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
            plt.tight_layout(); plt.savefig(log_dir+f"/test/{k*tau_unit:.2f}_prediction.png", dpi=150)
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
        iter = tqdm(test_loader)
        for input, _, internl_units in iter:
            input = model.scale(input.to(device))[..., :obs_dim]
            
            if model.__class__.__name__ == 'LED':
                target_list = [input]
                output_list = [input]
                pred = model.latent_forward(input, n=n)
                for i in range(n):
                    output_list.append(pred[i])
                    target_list.append(internl_units[i][...,:obs_dim]) # not scale
            else:

                t = torch.tensor([0., tau_unit], dtype=torch.float32, device=device)
                h_list, c_list = [None], [None]
                z_list = []
                output_list = [input]
                target_list = [input]
                for i in range(n):
                    if model.__class__.__name__ == 'NeuralODE':
                        output_list.append(model(output_list[-1], t, ode_dt)[:, -1:])
                    elif model.__class__.__name__ == 'DeepKoopman':
                        if i == 0:
                            pred_x, pred_z = model.evolve(input, n=1)
                        else:
                            pred_z = model.evolve_latent(z_list[-1], n=1)
                            pred_x = model.decoder(pred_z)
                        z_list.append(pred_z)
                        output_list.append(pred_x)
                
                    target_list.append(internl_units[i][...,:obs_dim]) # not scale
            
            # record results
            pred_per_tau.append(torch.cat(output_list, dim=1))
            true_per_tau.append(torch.cat(target_list, dim=1))

        pred_per_tau = model.descale(torch.cat(pred_per_tau, dim=0))
        true_per_tau = torch.cat(true_per_tau, dim=0)

    # metrics
    pred = pred_per_tau.detach().cpu().numpy()
    true = true_per_tau.detach().cpu().numpy()
    
    nmse_per_tau = []
    mae_per_tau = []
    for i in range(1, n+1):
        pred_tau_i = pred[:,i]
        true_tau_i = true[:,i]

        num = int(pred_tau_i.shape[0]/50)
        tmp = []
        tmp_mae = []
        for j in range(50):
            true_group = true_tau_i[num*j:num*(j+1)]
            pred_group = pred_tau_i[num*j:num*(j+1)]
            tmp.append(np.mean((true_group-pred_group)**2)/np.var(true_group))
            tmp_mae.append(np.mean(np.abs(true_group-pred_group)))
        nmse_per_tau.append(tmp)
        mae_per_tau.append(tmp_mae)
    
        if i in n_list:
            draw(true, pred, i)
    
    return nmse_per_tau