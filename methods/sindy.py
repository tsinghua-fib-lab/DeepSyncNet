import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import pysindy as ps


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    np.random.seed(seed)


class SINDy():

    def __init__(self, poly_order, threshold, max_iter, alpha, dim):

        feature_names = [f'z{i+1}' for i in range(dim)]
        poly_lib = ps.PolynomialLibrary(degree=poly_order)
        self.lib = ps.GeneralizedLibrary([poly_lib])
        self.optim = ps.STLSQ(threshold=threshold, max_iter=max_iter, alpha=alpha)
        
        self.model = ps.SINDy(
            optimizer=self.optim,
            feature_library=self.lib,
            feature_names=feature_names,
            discrete_time=False,
        )
    
    def fit(self, x_train, dt):

        self.model.fit(
            x_train, 
            t=dt
        )
        self.model.print()
    
    def simulate(self, x0, tspan):

        return self.model.simulate(
                    x0 = x0,
                    t = tspan,
                    integrator = 'odeint',
                )


def sindy_lorenz():
    
    data = np.load('Data/Coupled_Lorenz_0.05/origin/origin.npz')
    data = data['trace'] # (100,50000,3,2)
    x_train = data[:,:5000+1].copy()

    os.makedirs('sindy/Coupled_Lorenz_0.05/', exist_ok=True)
    trace_num, total_t, dt = 10, 50.0, 1e-2
    x_train = x_train.reshape(100,5000+1,6)

    for seed in [1,2,3]:
        
        seed_everything(seed)
        
        model = SINDy(poly_order=2, threshold=0.00001, max_iter=1000, alpha=0.5, dim=6)
        nmse_list = []
        for i, trace in enumerate(x_train):

            true_x = trace
            
            tspan = np.arange(0, total_t+dt, dt)
            model.fit(trace, tspan)
            pred_x = model.simulate(trace[0], tspan)
            
            if i % 10 == 0:
                tmp1 = trace.reshape(-1,3,2)
                tmp2 = pred_x.reshape(-1,3,2)
                plt.figure(figsize=(20,10))
                for idx in range(2):
                    for j, ylabel in enumerate(['x', 'y', 'z']):
                        ax = plt.subplot(2,3,idx*3+j+1)
                        ax.plot(tmp1[:,j,idx], label='true')
                        ax.plot(tmp2[:,j,idx], label='pred')
                        ax.set_ylabel(ylabel + str(idx+1))
                        if idx==0 and j==0: ax.legend(frameon=False)
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
                plt.tight_layout(); plt.savefig(f'sindy/Coupled_Lorenz_0.05/sindy_{i}.jpg', dpi=300)
                plt.close()

            nmse_list.append([])
            for k in range(1,5000+1):
                nmse_list[-1].append(np.mean((pred_x[k]-true_x[k])**2)/np.var(true_x[k]))

        nmse_list = np.array(nmse_list).T
        with open(f'results/Coupled_Lorenz_0.05/sindy_evolve_test_2.0.txt','a') as f:
            for i in range(5000):
                f.writelines(f'{(i+1)*dt:.4f}, {seed}, {nmse_list[i].tolist()}\n')


def sindy_FHN():
    
    data = np.load('Data/FHN_xdim30_noise0.1_du0.5_trace_num500_t20.01-circle/data/st0.0_et20.01/tau_0.1/train_10_10.0.npz')
    x_train = data['data'] # (1000采样点*300轨迹,100时间点,2,30)
    x_test = x_train[:1000*50,0,0].copy()

    os.makedirs('sindy', exist_ok=True)
    trace_num, total_t, dt = 50, 10.0, 1e-2
    x_test = x_test.reshape(50,1000,30)

    for seed in [1,2,3]:
        
        seed_everything(seed)
        
        model = SINDy(poly_order=1, threshold=0.01, max_iter=500, alpha=0.1, dim=30)
        nmse_list = []
        for i, trace in enumerate(x_test):

            true_x = trace
            
            tspan = np.arange(0, total_t, dt)
            model.fit(trace, dt)
            pred_x = model.simulate(trace[0], tspan)[1:]
            
            if i % 10 == 0:
                ax = plt.subplot(111)
                ax.plot(trace[:,15], label='true u15')
                ax.plot(pred_x[:,15], label='fit u15')
                ax.set_xlabel('t / s')
                ax.legend()
                plt.tight_layout(); plt.savefig(f'sindy/sindy_{i}.jpg', dpi=300)
                plt.close()

            nmse_list.append([])
            for k in range(100):
                nmse_list[-1].append(np.mean((pred_x[k*10]-true_x[k*10])**2)/np.var(true_x[k*10]))

        nmse_list = np.array(nmse_list).T
        with open(f'results/FHN_xdim30_noise0.1_du0.5/sindy_evolve_test_1.0.txt','a') as f:
            for i in range(100):
                f.writelines(f'{(i+1)*0.1:.1f}, {seed}, {nmse_list[i].tolist()}\n')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='lorenz')
    args = parser.parse_args()
    
    if args.system == 'lorenz':
        sindy_lorenz()
    elif args.system == 'FHN':
        sindy_FHN()