import os
import numpy as np
from tqdm import tqdm
from sdeint import itoEuler

from util import *
from .system import *


def generate_original_data(trace_num, total_t=6280, dt=0.001, save=True, plot=False, parallel=False, xdim=1, delta1=0., delta2=0., du=0.5, data_dir='Data/FHN_xdim1/origin'):
    
    def solve_1_trace(trace_id, du, plot=False):
        
        seed_everything(trace_id)

        if 'FHN' in data_dir:
            
            if 'FHNv' in data_dir:
                # initial condition
                u0, v0 = np.loadtxt(f'Data/IC/y0{trace_id-1}u.txt').tolist(), np.loadtxt(f'Data/IC/y0{trace_id-1}v.txt').tolist()
                x0 = u0 + v0

                Du, Dv, epsilon, a0, a1 = 1.0, 4.0, 0.01, -0.03, 2.0
                assert xdim==101, 'xdim must be 101 for FHNv'
                sde = FHNv(xdim, Du, Dv, epsilon, a0, a1, delta1, delta2)
            
            else:
                # initial condition
                v0, u0 = [], []
                for _ in range(xdim):
                    # angle = np.random.normal(loc=np.pi/2, scale=1.0) if np.random.randint(0, 2)==0 else np.random.normal(loc=-np.pi/2, scale=1.0)
                    angle = np.random.uniform(low=-np.pi, high=np.pi)
                    u0.append(10*np.cos(angle))
                    v0.append(10*np.sin(angle))
                if xdim>2:
                    u0[0], u0[-1] = -1.57, 2.66
                    v0[0], v0[-1] = -0.85, 0.46
                x0 = np.concatenate([u0, v0])

                a, b, I, epsilon = 0.7, 0.8, 0.5, 0.010
                sde = FHN(a=a, b=b, I=I, epsilon=epsilon, delta1=delta1, delta2=delta2, du=du, xdim=xdim)
            
        elif'HalfMoon_2D' in data_dir:
            assert xdim==1, 'xdim must be 1 for HalfMoon_2D'
            u0, v0 = np.random.uniform(-10, 10), np.random.uniform(-10, 10)
            x0 = [u0, v0]
            sde = HalfMoon_2D()
        
        elif 'Coupled_Lorenz' in data_dir:
            assert xdim==2, 'xdim must be 2 for Coupled_Lorenz'
            x0 = np.random.choice(np.linspace(0, 1, 3*xdim), 3*xdim, replace=True)

            sigma, rho, beta = 10.0, 28.0, 8/3
            sde = Coupled_Lorenz(sigma, rho, beta)
        
        # solve SDE
        tspan = np.arange(0, total_t, dt)
        sol = itoEuler(sde.f, sde.g, x0, tspan)
        
        if 'FHN' in data_dir:
            u = sol[:, :xdim]
            v = sol[:, xdim:]
            sol = np.stack((u,v), axis=0).transpose(1,0,2)  # (time_length, 2, xdim)
        elif 'HalfMoon_2D' in data_dir:
            u = sol[:, 0]
            v = sol[:, 1]
            x = v * np.cos(u+v-1)
            y = v * np.sin(u+v-1)
            sol = np.concatenate((x[...,np.newaxis],y[...,np.newaxis],sol), axis=1)[:,np.newaxis]  # (time_length, 1, 4)
        elif 'Coupled_Lorenz' in data_dir:
            xx = sol[:, :2]
            yy = sol[:, 2:4]
            zz = sol[:, 4:]
            sol = np.stack((xx, yy, zz), axis=0).transpose(1,0,2)  # (time_length, 3, 2)

        if plot:
            if 'FHN' in data_dir:
                # heatmap
                fig, ax = plt.subplots(figsize=(6,6))
                im = ax.imshow(u[::-1], vmin=-3, vmax=3, cmap=my_cmap, aspect='auto', extent=[0, xdim, 0, total_t])
                ax.set_xlabel('x')
                ax.set_ylabel('t')
                ax.set_title('u')
                fig.colorbar(im, ax=ax)
                fig.savefig(f'{data_dir}/{trace_id}_fhn.png', dpi=150)
                plt.close()
            elif 'HalfMoon_2D' in data_dir:
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(131)
                ax1.plot(tspan, sol[:,0, 2], label='u', c=colors[0])
                ax1.plot(tspan, sol[:,0, 3], label='v', c=colors[1])
                ax1.set_xlabel('t')
                ax1.set_ylabel('value')
                ax1.legend()
                ax2 = fig.add_subplot(132)
                ax2.scatter(sol[:,0,0], sol[:,0,1], c=tspan, cmap=my_cmap, linewidths=0.01, s=6)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax3 = fig.add_subplot(133)
                ax3.plot(tspan, sol[:,0,0], label='x', c=colors[2])
                ax3.plot(tspan, sol[:,0,1], label='y', c=colors[3])
                ax3.set_xlabel('t')
                ax3.set_ylabel('value')
                ax1.legend()
                fig.tight_layout()
                plt.tight_layout(); plt.savefig(f'{data_dir}/{trace_id}_halfmoon.jpg', dpi=300)
                plt.close()
                
                fig = plt.figure(figsize=(6,6))
                ax1 = fig.add_subplot(111)
                ax1.plot(tspan, sol[:,0, 2], label='u', c=colors[0])
                ax1.plot(tspan, sol[:,0, 3], label='v', c=colors[1])
                ax1.set_xlabel('t / s')
                ax1.set_ylabel('value')
                ax1.legend()
                plt.tight_layout(); plt.savefig(f'{data_dir}/{trace_id}_uv.pdf', dpi=300)
                plt.close()
                fig = plt.figure(figsize=(6,6))
                ax1 = fig.add_subplot(111)
                ax1.scatter(sol[::10,0,0], sol[::10,0,1], c=tspan[::10], cmap=my_cmap, linewidths=0.01, s=11)
                ax1.set_xlabel('t / s')
                ax1.set_ylabel('value')
                plt.tight_layout(); plt.savefig(f'{data_dir}/{trace_id}_xy.pdf', dpi=300)
                plt.close()
            elif 'Coupled_Lorenz' in data_dir:
                # 3D trajectory
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.plot(xx[:,0], yy[:,0], zz[:,0], c=colors[0], linewidth=0.5)
                ax1.set_xlabel('x1')
                ax1.set_ylabel('y1')
                ax1.set_zlabel('z1')
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.plot(xx[:,1], yy[:,1], zz[:,1], c=colors[1], linewidth=0.5)
                ax2.set_xlabel('x2')
                ax2.set_ylabel('y2')
                ax2.set_zlabel('z2')
                plt.savefig(f'{data_dir}/{trace_id}_3D.jpg', dpi=300)
                
                # 2D trajectory
                fig = plt.figure(figsize=(10, 10))
                ax1 = fig.add_subplot(211)
                ax1.plot(tspan, xx[:,0], c=colors[0], label='x1')
                ax1.plot(tspan, yy[:,0], c=colors[1], label='y1')
                ax1.plot(tspan, zz[:,0], c=colors[2], label='z1')
                ax1.set_xlabel('t')
                ax1.legend(frameon=False)
                ax2 = fig.add_subplot(212)
                ax2.plot(tspan, xx[:,1], c=colors[3], label='x2')
                ax2.plot(tspan, yy[:,1], c=colors[4], label='y2')
                ax2.plot(tspan, zz[:,1], c=colors[5], label='z2')
                ax2.set_xlabel('t')
                ax2.legend(frameon=False)
                plt.savefig(f'{data_dir}/{trace_id}_2D.jpg', dpi=300)
        
        return sol
    
    if save and os.path.exists(f'{data_dir}/origin.npz'): return
    
    os.makedirs(data_dir, exist_ok=True)
    
    if 'FHN' in data_dir:
        if 'FHNv' in data_dir:
            trace = []
            assert trace_num<=5, 'Only 5 ICs for FHNv'
            for trace_id in tqdm(range(1, trace_num+1)):
                sol = solve_1_trace(trace_id, du, plot=trace_id<10)
                trace.append(sol)
        else:
            xdim = xdim + 2  # add two boundary points
            trace = []
            for trace_id in tqdm(range(1, trace_num+1)):
                sol = solve_1_trace(trace_id, du, plot=trace_id<10)
                sol = sol[:,:,1:-1]  # remove boundary points
                trace.append(sol)
            xdim = xdim - 2  # remove two boundary points
    elif 'HalfMoon' in data_dir or 'Coupled_Lorenz' in data_dir:
        trace = []
        for trace_id in tqdm(range(1, trace_num+1)):
            sol = solve_1_trace(trace_id, du, plot=trace_id<10)
            trace.append(sol)
    
    if save: 
        np.savez(f'{data_dir}/origin.npz', trace=trace, dt=dt, T=total_t)
        print(f'save origin data from seed 1 to {trace_num} at {data_dir}/')
    

def generate_dataset_slidingwindow(tmp, trace_num, tau, is_print=False, sequence_length=None, data_dir='Data/FHN_xdim1/data', start_t=0.0, end_t=None, sliding_length=None, stride_t=None, only_test=False, horizon=None):

    if sliding_length is not None:
        data_dir = f"{data_dir}/st{start_t}_et{end_t}/sliding_length-{sliding_length}/tau_{tau}"
    else:
        data_dir = f"{data_dir}/st{start_t}_et{end_t}/tau_{tau}"

    # Skip if existing
    if sequence_length is not None:
        if horizon is not None:
            tmp2 = f'{sequence_length}_{horizon}'
        else:
            tmp2 = f'{sequence_length}'
        if only_test and os.path.exists(f"{data_dir}/test_{tmp2}.npz"):
            return
        elif not only_test and os.path.exists(f"{data_dir}/train_{tmp2}.npz") and os.path.exists(f"{data_dir}/val_{tmp2}.npz") and os.path.exists(f"{data_dir}/test_{tmp2}.npz"):
            return
    elif sequence_length is None:
        if only_test and os.path.exists(f"{data_dir}/test.npz"):
            return
        elif os.path.exists(f"{data_dir}/train.npz") and os.path.exists(f"{data_dir}/val.npz") and os.path.exists(f"{data_dir}/test.npz"):
            return

    # load original data
    dt, data, total_t = tmp['dt'], tmp['trace'], tmp['T']
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    os.makedirs(data_dir, exist_ok=True)
    scale_path = data_dir.replace(f'/tau_{tau}', '/')
    if not os.path.exists(scale_path + "/tau.txt"):
        np.savetxt(scale_path + "/data_mean.txt", np.mean(data, axis=(0,1)).reshape(1,-1))
        np.savetxt(scale_path + "/data_std.txt", np.std(data, axis=(0,1)).reshape(1,-1))
        np.savetxt(scale_path + "/data_max.txt", np.max(data, axis=(0,1)).reshape(1,-1))
        np.savetxt(scale_path + "/data_min.txt", np.min(data, axis=(0,1)).reshape(1,-1))
        np.savetxt(scale_path + "/tau.txt", [tau]) # Save the timestep

    # single-sample time steps
    if sequence_length is None:
        sequence_length = 2 if tau != 0. else 1
        seq_none = True
    else:
        seq_none = False
    
    ##################################
    # Create [train,val,test] dataset
    ##################################
    train_num = int(0.8*trace_num)
    val_num = int(0.1*trace_num)
    test_num = int(0.1*trace_num)
    if 'FHNv' in data_dir:
        train_num, val_num, test_num = 3, 1, 1
    trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
    for item in ['train','val','test']:
        if only_test and item!='test': continue
                    
        # select trace num
        data_item = data[trace_list[item]]  # [trace_num, time_length, channel_num, feature_num]
        if is_print: print(f'tau[{tau}] {item} data shape: {data_item.shape}')
        
        # cut out time zone
        if end_t:
            assert total_t >= end_t, f"end_t({end_t}s) is longer than total_t({total_t}s)"
            data_item = data_item[:, int(start_t/dt):int(end_t/dt)]
            if is_print: print(f'tau[{tau}] {item} data shape: {data_item.shape} (cut out time zone)')
        else:
            end_t = total_t
            data_item = data_item[:, int(start_t/dt):int(end_t/dt)]

        # cut out sliding window
        if sliding_length:
            assert tau != 0, f"tau must be not 0 when sliding_length is not None"
            assert sliding_length+tau <= end_t, f"sliding_length({sliding_length}s) + tau({tau}s) is longer than end_t({end_t}s)"
            data_item = data_item[:, 0:int((sliding_length+tau)/dt)]
            if is_print: print(f'tau[{tau}] {item} data shape: {data_item.shape} (cut out sliding window)')
            end_t = start_t+sliding_length + tau

        # sampling for sliding window
        stride = int(stride_t/dt) if stride_t!=None else 1
        window_size = int(tau/dt) if tau!=0. else 1
        if horizon is not None:
            assert horizon+tau*(sequence_length-1) <= total_t, \
                f"Wrong: horizon+tau*(sequence_length-1)={horizon+tau*(sequence_length-1)}s is large than total_t={total_t}s!"
            horizon_points = int(horizon / dt)
            idxs = np.arange(0, horizon_points, stride)
            if is_print: print(f'trajectory_length={np.shape(data_item)[1]}, horizon={horizon}, stride={stride}, len(idxs)={len(idxs)}')
        else:
            idxs = np.arange(0, np.shape(data_item)[1]-window_size*(sequence_length-1), stride)
        
        # keep same sample num for FHN System
        if 'FHN' in data_dir and not 'FHNv' in data_dir:
            if sliding_length is not None and sliding_length>0.01: 
                idxs = idxs[np.random.choice(len(idxs), int(0.01/dt), replace=False)]
                idxs = np.sort(idxs)
        
        # sliding window
        sequences = [data_item[:, i:j:int(window_size)] for i, j in zip(idxs, idxs+int(window_size)*(sequence_length-1)+1)]
        sequences = np.transpose(sequences, (1,0,2,3,4)) # (trace_num, sample_num, sequence_length, channel_num, feature_num)
        sequences = np.concatenate(sequences, axis=0) # (trace_num*sample_num, sequence_length, channel_num, feature_num)
        
        if item=='test':
            if horizon is not None:
                start = start_t + tau + horizon
            else:
                start = start_t + tau
            
            if 'FHN' in data_dir:
                # plot heatmap
                fig, ax = plt.subplots(figsize=(6,6))
                im = ax.imshow(np.array(sequences)[int(sequences.shape[0]/test_num):0:-1,1,0,:], vmin=-3, vmax=3, cmap=my_cmap, aspect='auto', extent=[0, data_item.shape[-1], start, end_t])
                ax.set_xlabel('x')
                ax.set_ylabel('t')
                ax.set_title('u')
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(f'{data_dir}/target_u.png', dpi=150)
                plt.close()
                fig, ax = plt.subplots(figsize=(6,6))
                im = ax.imshow(np.array(sequences)[int(sequences.shape[0]/test_num):0:-1,1,1,:], vmin=-3, vmax=3, cmap=my_cmap, aspect='auto', extent=[0, data_item.shape[-1], start, end_t])
                ax.set_xlabel('x')
                ax.set_ylabel('t')
                ax.set_title('v')
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(f'{data_dir}/target_v.png', dpi=150)
                plt.close()
            elif 'HalfMoon_2D' in data_dir:
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121)
                ax1.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,0,2], label='u', c=colors[0])
                ax1.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,0,3], label='v', c=colors[-1])
                ax1.set_xlabel('t')
                ax1.set_ylabel('value')
                ax1.legend()
                ax3 = fig.add_subplot(122)
                ax3.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,0,0], label='x', c=colors[0])
                ax3.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,0,1], label='y', c=colors[-1])
                ax3.set_xlabel('t')
                ax3.set_ylabel('value')
                ax3.legend()
                plt.tight_layout(); plt.savefig(f'{data_dir}/target.jpg', dpi=300)
                plt.close()
            elif 'Coupled_Lorenz' in data_dir:
                # 2D trajectory
                fig = plt.figure(figsize=(10, 10))
                ax1 = fig.add_subplot(211)
                ax1.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,0,0], c=colors[0], label='x1')
                ax1.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,1,0], c=colors[1], label='y1')
                ax1.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,2,0], c=colors[2], label='z1')
                ax1.set_xlabel('t')
                ax1.legend(frameon=False)
                ax2 = fig.add_subplot(212)
                ax2.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,0,1], c=colors[3], label='x2')
                ax2.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,1,1], c=colors[4], label='y2')
                ax2.plot(np.array(sequences)[:int(sequences.shape[0]/test_num),1,2,1], c=colors[5], label='z2')
                ax2.set_xlabel('t')
                ax2.legend(frameon=False)
                plt.savefig(f'{data_dir}/target_2D.jpg', dpi=300)

        # sequences = np.array(sequences)
        if is_print: print(f'tau[{tau}]', f"{item} dataset (sequence_length={sequence_length}, window_size={window_size})", np.shape(sequences), '\n')

        # save
        if not seq_none:
            if horizon is not None:
                np.savez(data_dir+f'/{item}_{sequence_length}_{horizon}.npz', data=sequences)
            else:
                np.savez(data_dir+f'/{item}_{sequence_length}.npz', data=sequences)
        else:
            np.savez(data_dir+f'/{item}.npz', data=sequences)