<<<<<<< HEAD
import os
import argparse
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge


plt.style.use(['ieee'])
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 12
plt.rcParams['legend.fontsize'] = 16
colors = [(2/255,48/255,71/255), (255/255,202/255,95/255), (26/255,134/255,163/255), (253/255,152/255,2/255), (70/255,172/255,202/255), (14/255,91/255,118/255), (155/255,207/255,232/255), (251/255,132/255,2/255)]
markers = ['*', 'o', 's', 'D', 'v', '^', 'h']


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    np.random.seed(seed)


def rc_FHNv():
    
    data = np.load('Data/FHNv_xdim101_noise0.2_du0.5_trace_num5_t8000.0/data/st0.0_et8000.0/tau_10.0/test_500_3000.0.npz')
    data_x = data['data'] # (3000,500,2,101)
    total_t, dt = 5000.0, 10.0

    os.makedirs('rc/FHNv_xdim101_noise0.2_du0.5/', exist_ok=True)
    reservoir = Reservoir(units=1000, lr=1.0, sr=0.99)
    readout = Ridge(output_dim=202, ridge=1e-5)
    esn = reservoir >> readout

    nmse_list = [[] for _ in range(int(total_t/dt))]
    # for i in range(20):
    #     trace = data_x[i].reshape(-1,2*101)
    #     esn.fit(trace[:-2], trace[1:-1])

    for i in np.random.choice(len(data_x), 100, replace=False):

        trace = data_x[i].reshape(-1,2*101)
        
        esn.fit(trace[:49], trace[1:50])
        
        result = [trace[:1]]
        for t in range(500):
            result.append(esn.run(result[-1]))
        result = np.concatenate(result, axis=0)[1:]
            
        if i % 5 == 0:
            ax = plt.subplot(111)
            ax.plot(trace[:,50], label='true xdim=10')
            ax.plot(result[:,50], label='pred xdim=10')
            ax.set_xlabel('t / s')
            ax.legend()
            plt.tight_layout(); plt.savefig(f'rc/FHNv_xdim101_noise0.2_du0.5/rc_FHNv_{i}.jpg', dpi=300)
            plt.close()

        for k in range(500):
            nmse_list[k].append(np.mean((trace[k]-result[k])**2)/np.var(trace[k]))

    with open(f'results/FHNv_xdim101_noise0.2_du0.5/rc_evolve_test_0.5.txt','w') as f:
        for i in range(500):
            f.writelines(f'{(i+1)*dt:.1f}, 1, {nmse_list[i]}\n')
    
    
def rc_FHN():

    data = np.load('Data/FHN_xdim30_noise0.1_du0.5_trace_num500_t20.01-circle/data/st0.0_et20.01/tau_0.1/test_100_6.0.npz')
    data_x = data['data'] # (10000,100,2,30)
    total_t, dt = 10.0, 0.1

    os.makedirs('rc/FHN_xdim30_noise0.1_du0.5/', exist_ok=True)

    for seed in [1,2,3]:
        seed_everything(seed)
        
        reservoir = Reservoir(units=1000, lr=1.0, sr=0.99)
        readout = Ridge(output_dim=60, ridge=1e-5)
        esn = reservoir >> readout

        nmse_list = [[] for _ in range(100)]
        for i in np.random.choice(len(data_x), 200, replace=False):

            trace = data_x[i].reshape(-1,2*30)
            esn.fit(trace[:99], trace[1:100])
        
            result = [trace[:1]]
            for t in range(100):
                result.append(esn.run(result[-1]))
            result = np.concatenate(result, axis=0)[1:]
                
            if i % 50 == 0:
                ax = plt.subplot(111)
                ax.plot(trace[1:-1,10], label='true xdim=10')
                ax.plot(result[1:-1,10], label='pred xdim=10')
                ax.set_xlabel('t / s')
                ax.legend()
                plt.tight_layout(); plt.savefig(f'rc/FHN_xdim30_noise0.1_du0.5/rc_fhn_{i}.jpg', dpi=300)
                plt.close()

            for k in range(100):
                nmse_list[k].append(np.mean((trace[k]-result[k])**2)/np.var(trace[k]))

    with open(f'results/FHN_xdim30_noise0.1_du0.5/rc_evolve_test_1.0.txt','w') as f:
        for i in range(100):
            f.writelines(f'{(i+1)*0.1:.1f}, 1, {nmse_list[i]}\n')
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='lorenz')
    args = parser.parse_args()
    
    if args.system == 'FHNv':
        rc_FHNv()
    elif args.system == 'FHN':
=======
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge

from util import seed_everything


plt.style.use(['ieee'])
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 12
plt.rcParams['legend.fontsize'] = 16
colors = [(2/255,48/255,71/255), (255/255,202/255,95/255), (26/255,134/255,163/255), (253/255,152/255,2/255), (70/255,172/255,202/255), (14/255,91/255,118/255), (155/255,207/255,232/255), (251/255,132/255,2/255)]
markers = ['*', 'o', 's', 'D', 'v', '^', 'h']


def rc_lorenz():
    
    data = np.load('Data/Coupled_Lorenz_0.05/data/st0.0_et500.0/tau_0.01/test_5000_100.0.npz')
    data_x = data['data'] # (10000,5000,3,2)
    total_t, dt = 50.0, 0.01
    lenth = int(total_t/dt)
    data_x = data_x[:,:lenth]

    os.makedirs('rc/Coupled_Lorenz_0.05/', exist_ok=True)

    for seed in [1,2,3]:
        seed_everything(seed)
        
        reservoir = Reservoir(units=1000, lr=1.0)
        readout = Ridge(output_dim=6, ridge=1e-6)
        esn = reservoir >> readout

        nmse_list = [[] for _ in range(lenth)]
        for i in np.random.choice(len(data_x), 200, replace=False):
            trace = data_x[i].reshape(-1,2*3)
            esn.fit(trace[:lenth-1], trace[1:lenth])

            result = [trace[:1]]
            for t in range(lenth):
                result.append(esn.run(result[-1]))
            result = np.concatenate(result, axis=0)[1:]
                
            if i % 1 == 0:
                tmp1 = trace.reshape(-1,3,2)
                tmp2 = result.reshape(-1,3,2)
                plt.figure(figsize=(20,10))
                for idx in range(2):
                    for j, ylabel in enumerate(['x', 'y', 'z']):
                        ax = plt.subplot(2,3,idx*3+j+1)
                        ax.plot(tmp1[:2000,j,idx], label='true')
                        ax.plot(tmp2[:2000,j,idx], label='pred')
                        ax.set_ylabel(ylabel + str(idx+1))
                        if idx==0 and j==0: ax.legend(frameon=False)
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                plt.tight_layout(); plt.savefig(f'rc/Coupled_Lorenz_0.05/{i}.jpg', dpi=300)
                plt.close()

            for k in range(lenth):
                nmse_list[k].append(np.mean((trace[k]-result[k])**2)/np.var(trace[k]))

    with open(f'results/Coupled_Lorenz_0.05/rc_evolve_test_2.0.txt','w') as f:
        for i in range(lenth):
            f.writelines(f'{(i+1)*dt:.4f}, 1, {nmse_list[i]}\n')


def rc_FHNv():
    
    data = np.load('Data/FHNv_xdim101_noise0.2_du0.5_trace_num5_t8000.0/data/st0.0_et8000.0/tau_10.0/test_500_3000.0.npz')
    data_x = data['data'] # (3000,500,2,101)
    total_t, dt = 5000.0, 10.0

    os.makedirs('rc/FHNv_xdim101_noise0.2_du0.5/', exist_ok=True)
    reservoir = Reservoir(units=1000, lr=1.0, sr=0.99)
    readout = Ridge(output_dim=202, ridge=1e-5)
    esn = reservoir >> readout

    nmse_list = [[] for _ in range(int(total_t/dt))]
    # for i in range(20):
    #     trace = data_x[i].reshape(-1,2*101)
    #     esn.fit(trace[:-2], trace[1:-1])

    for i in np.random.choice(len(data_x), 100, replace=False):

        trace = data_x[i].reshape(-1,2*101)
        
        esn.fit(trace[:49], trace[1:50])
        
        result = [trace[:1]]
        for t in range(500):
            result.append(esn.run(result[-1]))
        result = np.concatenate(result, axis=0)[1:]
            
        if i % 5 == 0:
            ax = plt.subplot(111)
            ax.plot(trace[:,50], label='true xdim=10')
            ax.plot(result[:,50], label='pred xdim=10')
            ax.set_xlabel('t / s')
            ax.legend()
            plt.tight_layout(); plt.savefig(f'rc/FHNv_xdim101_noise0.2_du0.5/rc_FHNv_{i}.jpg', dpi=300)
            plt.close()

        for k in range(500):
            nmse_list[k].append(np.mean((trace[k]-result[k])**2)/np.var(trace[k]))

    with open(f'results/FHNv_xdim101_noise0.2_du0.5/rc_evolve_test_0.5.txt','w') as f:
        for i in range(500):
            f.writelines(f'{(i+1)*dt:.1f}, 1, {nmse_list[i]}\n')
    
    
def rc_FHN():

    data = np.load('Data/FHN_xdim30_noise0.1_du0.5_trace_num500_t20.01-circle/data/st0.0_et20.01/tau_0.1/test_100_6.0.npz')
    data_x = data['data'] # (10000,100,2,30)
    total_t, dt = 10.0, 0.1

    os.makedirs('rc/FHN_xdim30_noise0.1_du0.5/', exist_ok=True)

    for seed in [1,2,3]:
        seed_everything(seed)
        
        reservoir = Reservoir(units=1000, lr=1.0, sr=0.99)
        readout = Ridge(output_dim=60, ridge=1e-5)
        esn = reservoir >> readout

        nmse_list = [[] for _ in range(100)]
        for i in np.random.choice(len(data_x), 200, replace=False):

            trace = data_x[i].reshape(-1,2*30)
            esn.fit(trace[:99], trace[1:100])
        
            result = [trace[:1]]
            for t in range(100):
                result.append(esn.run(result[-1]))
            result = np.concatenate(result, axis=0)[1:]
                
            if i % 50 == 0:
                ax = plt.subplot(111)
                ax.plot(trace[1:-1,10], label='true xdim=10')
                ax.plot(result[1:-1,10], label='pred xdim=10')
                ax.set_xlabel('t / s')
                ax.legend()
                plt.tight_layout(); plt.savefig(f'rc/FHN_xdim30_noise0.1_du0.5/rc_fhn_{i}.jpg', dpi=300)
                plt.close()

            for k in range(100):
                nmse_list[k].append(np.mean((trace[k]-result[k])**2)/np.var(trace[k]))

    with open(f'results/FHN_xdim30_noise0.1_du0.5/rc_evolve_test_1.0.txt','w') as f:
        for i in range(100):
            f.writelines(f'{(i+1)*0.1:.1f}, 1, {nmse_list[i]}\n')
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='lorenz')
    args = parser.parse_args()
    
    if args.system == 'lorenz':
        rc_lorenz()
    elif args.system == 'FHNv':
        rc_FHNv()
    elif args.system == 'FHN':
>>>>>>> dc075487f2a58d03645bfca002881f561f1e93d0
        rc_FHN()