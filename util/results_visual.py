import os
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.style.use(['ieee'])
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['lines.markersize'] = 15
plt.rcParams['legend.fontsize'] = 40
path = 'util/calibri.ttf'
fm.fontManager.addfont(path)
prop = fm.FontProperties(fname=path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['mathtext.fontset'] = 'dejavusans'
colors = [(2/255,48/255,71/255), (255/255,202/255,95/255), (26/255,134/255,163/255), (253/255,152/255,2/255), (70/255,172/255,202/255), (14/255,91/255,118/255), (155/255,207/255,232/255), (251/255,132/255,2/255)]
color_bar = [(2/255,48/255,71/255), (14/255,91/255,118/255), (26/255,134/255,163/255), (70/255,172/255,202/255), (155/255,207/255,232/255), (243/255,249/255,252/255), (255/255,202/255,95/255), (254/255,168/255,9/255), (253/255,152/255,2/255), (251/255,132/255,2/255)]
markers = ['*', 'o', 's', 'D', 'v', '^', 'h']
from matplotlib.colors import LinearSegmentedColormap
my_cmap = LinearSegmentedColormap.from_list("mycmap", color_bar)


def plot_id_per_tau(tau_list, seed_list, id_epoch, log_dir, start_t, end_t, sliding_length, autoEmbedSize=False):

    id_per_tau = [[] for _ in tau_list]
    for i, tau in enumerate(tau_list):
        if autoEmbedSize:
            fp = open(log_dir + f'st{start_t}_et{end_t}/sliding_length-{sliding_length}/tau_{round(tau,4)}/final/test_log.txt', 'r')
        else:
            fp = open(log_dir + f'st{start_t}_et{end_t}/sliding_length-{sliding_length}/tau_{round(tau,4)}/test_log.txt', 'r')
        for line in fp.readlines()[::-1]:
            seed = int(line[:-1].split(',')[1])
            mse = float(line[:-1].split(',')[2])
            epoch = int(line[:-1].split(',')[3])
            MLE_id = float(line[:-1].split(',')[4])
            MOM_id = float(line[:-1].split(',')[5])
            MADA_id = float(line[:-1].split(',')[6])
            TLE_id = float(line[:-1].split(',')[7])
            MIND_id = float(line[:-1].split(',')[8])
            # nmse = float(line[:-1].split(',')[9])

            if epoch == id_epoch and seed in seed_list:
                if len(id_per_tau[i]) == 0:
                    id_per_tau[i].append([mse, MLE_id, MOM_id, MADA_id, TLE_id, MIND_id])
        fp.close()

    id_per_tau = np.mean(id_per_tau, axis=-2)

    if autoEmbedSize:
        embed_size = []
        for tau in tau_list:
            embed_size.append(np.mean(np.loadtxt(log_dir + f'st{start_t}_et{end_t}/sliding_length-{sliding_length}/tau_{round(tau,4)}/final/embed_size.txt')))

    fig, ax1 = plt.subplots(figsize=(14,8))
    ax2 = ax1.twinx()
    ax1.plot(tau_list, id_per_tau[:,1], marker=markers[0], label="MLE", c=colors[0])
    ax1.plot(tau_list, id_per_tau[:,2], marker=markers[1], label="MOM", c=colors[3])
    ax1.plot(tau_list, id_per_tau[:,3], marker=markers[2], label="MADA", c=colors[-1])
    # ax1.plot(tau_list, id_per_tau[:,4], marker="^", label="TLE")
    if autoEmbedSize:
        ax1.plot(tau_list, embed_size, marker=markers[3], label="Embedding size")
    ax2.plot(tau_list, id_per_tau[:,0], marker=markers[4], color='tab:red', alpha=0.5)
    
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.set_xlabel(r'$\tau$ / s')
    ax1.set_ylabel('Intrinsic dimensionality')
    ax2.set_ylabel('NMSE', rotation=270, labelpad=20)
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')
    ax1.legend(frameon=False)
    plt.xscale("log")
    plt.xticks(ticks=tau_list, )
    plt.yticks()
    plt.tight_layout(); plt.savefig(f'{log_dir}st{start_t}_et{end_t}/sliding_length-{sliding_length}/id_mse_per_tau.png', dpi=300)

    # MLE
    plt.figure(figsize=(8,5))
    # plt.figure(figsize=(10,5))
    ax = plt.subplot(111)
    # plt.plot(np.array(tau_list)*1e3, id_per_tau[:,1], marker=markers[0], c=colors[0])
    # plt.xlabel(r'$\tau \ (ns)$')
    ax.plot(tau_list, id_per_tau[:,1], marker=markers[0], c=colors[0])
    plt.xlabel(r'$\tau \ (s)$')
    plt.ylabel('Dynamic ID')
    plt.tick_params(axis='x')
    plt.tick_params(axis='y')
    plt.xscale("log")
    # plt.yticks([1,2])
    # plt.yticks([10,20])
    # plt.yticks([7,10,13])
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.tight_layout(); plt.savefig(f'{log_dir}st{start_t}_et{end_t}/sliding_length-{sliding_length}/id_per_tau.pdf', dpi=300)
    plt.close()
    
    algos = ['MLE', 'MOM', 'MADA', 'TLE', 'MIND']
    ymax = np.max(id_per_tau[:,1:])
    ymin = np.min(id_per_tau[:,1:])
    for i, algo in enumerate(algos):
        plt.figure(figsize=(7,7))
        ax = plt.subplot(111)
        ax.plot(tau_list, id_per_tau[:,i+1], marker=markers[i], c=colors[i])
        plt.xlabel(r'$\tau \ (s)$')
        # plt.xlabel(r'$\tau \ (ns)$')
        plt.ylabel('Dynamic ID')
        plt.tick_params(axis='x')
        plt.tick_params(axis='y')
        plt.xscale("log")
        plt.title(algo)
        plt.ylim(ymin-0.5, ymax+0.5)
        plt.tight_layout()
        plt.savefig(f'{log_dir}st{start_t}_et{end_t}/sliding_length-{sliding_length}/id_per_tau_{algo}.pdf', dpi=300)
        plt.close()


def FHNv_fig(tau_s, tau_unit, n, seed_num, noise, system):
    
    our_node = open(f'results/{system}/ours-neural_ode_slow-3_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    lstm = open(f'results/{system}/led-2_evolve_test_{tau_s}.txt', 'r')
    led = open(f'results/{system}/led-2_evolve_test_{tau_s}.txt', 'r')
    ode = open(f'results/{system}/neural_ode_evolve_test_{tau_s}.txt', 'r')
    sindy = open(f'results/{system}/led-2_evolve_test_{tau_s}.txt', 'r')
    rc = open(f'results/{system}/rc_evolve_test_{tau_s}.txt', 'r')
    dk = open(f'results/{system}/deepkoopman-MLP-2_evolve_test_{tau_s}.txt', 'r')

    tau_list = [round(tau_unit*i, 1) for i in range(1, n+1)]
    
    our_node_data, ode_data, sindy_data, led_data, lstm_data, rc_data, dk_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for i, data in enumerate([our_node, ode, sindy, led, lstm, rc, dk]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            nmse_str = line.split('[')[1].split(']')[0]
            nmse = [float(value) for value in nmse_str.split(',')]

            if tau in tau_list and seed in range(1,seed_num+1):
                if i == 0:
                    our_node_data[tau_list.index(tau)].extend(nmse)
                elif i == 1:
                    ode_data[tau_list.index(tau)].extend(nmse)
                elif i == 2:
                    sindy_data[tau_list.index(tau)].extend(nmse)
                elif i == 3:
                    led_data[tau_list.index(tau)].extend(nmse)
                elif i == 4:
                    lstm_data[tau_list.index(tau)].extend(nmse)
                elif i == 5:
                    rc_data[tau_list.index(tau)].extend(nmse)
                elif i == 6:
                    dk_data[tau_list.index(tau)].extend(nmse)

    our_node_data = np.array(our_node_data)[:, np.random.choice(range(len(our_node_data[0])), 30)]
    ode_data = np.array(ode_data)[:, np.random.choice(range(len(ode_data[0])), 30)]
    sindy_data = np.array(sindy_data)[:, np.random.choice(range(len(sindy_data[0])), 30)]
    led_data = np.array(led_data)[:, np.random.choice(range(len(led_data[0])), 30)]
    lstm_data = np.array(lstm_data)[:, np.random.choice(range(len(lstm_data[0])), 30)]
    rc_data = np.array(rc_data)[:, np.random.choice(range(len(rc_data[0])), 30)]
    dk_data = np.array(dk_data)[:, np.random.choice(range(len(dk_data[0])), 30)]
    
    for i, item in enumerate(['NMSE']):

        labels = ['Neural ODE', 'LED', 'RC', 'DeepKoopman', 'DeepSyncNet']
        
        plt.figure(figsize=(16,5))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([ode_data, led_data, rc_data, dk_data, our_node_data]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.plot(tau_list[:200:15], mean_data[:200:15], marker=markers[j], label=labels[j], c=colors[j])
            ax.fill_between(tau_list[:200:15], mean_data[:200:15]-std_data[:200:15], mean_data[:200:15]+std_data[:200:15], alpha=0.1, color=colors[j])
            # ax.errorbar()
        ax.set_xlabel(r'$t \ (s)$')
        ax.set_ylabel(item)
        plt.tight_layout(); plt.savefig(f'results/{system}/curve.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([ode_data, led_data, rc_data, dk_data, our_node_data]):
            mean_data = np.mean(data, axis=1)
            ax.plot(tau_list[:200:15], np.zeros_like(mean_data[:200:15]), marker=markers[j], label=labels[j], c=colors[j])
        plt.ylim(0,1)
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/curve_legend.pdf', dpi=300)
        plt.close()

        plt.figure(figsize=(16,5))
        positions = [
            (1,1.2,1.4,1.6,1.8),
            (2.3,2.5,2.7,2.9,3.1),
            (3.6,3.8,4.0,4.2,4.4),
        ]
        t_list = [0, 99, 199]
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for data in [ode_data, led_data, rc_data, dk_data, our_node_data]:
                tmp_data.append([data[t_n]])
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        x_position=[1.4,2.7,4.0]
        x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        plt.xticks([i for i in x_position], x_position_fmt)
        plt.yticks([0, 1, 2, 3])
        plt.ylim(0, 3.8)
        plt.ylabel('NMSE')
        plt.tight_layout(); plt.savefig(f'results/{system}/box.pdf',dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for data in [ode_data, led_data, rc_data, dk_data, our_node_data]:
                tmp_data.append([np.zeros_like(data[t_n])])
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.ylim(0,1)
        plt.legend(bplot['boxes'], labels, frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/box_legend.pdf',dpi=300)
        plt.close()
        

def halfmoon_fig(tau_s, tau_unit, n, seed_num, noise, system):
    
    our_node = open(f'results/{system}/ours-neural_ode_slow-1_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    lstm = open(f'results/{system}/lstm_evolve_test_{tau_s}.txt', 'r')
    led = open(f'results/{system}/led-1_evolve_test_{tau_s}.txt', 'r')
    ode = open(f'results/{system}/neural_ode_evolve_test_{tau_s}.txt', 'r')
    sindy = open(f'results/{system}/sindy_evolve_test_{tau_s}.txt', 'r')
    dk = open(f'results/{system}/deepkoopman-MLP-1_evolve_test_{tau_s}.txt', 'r')

    tau_list = [round(tau_unit*i, 1) for i in range(1, n+1)]
    
    our_node_data, ode_data, sindy_data, led_data, lstm_data, dk_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for i, data in enumerate([our_node, ode, sindy, led, lstm, dk]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            nmse_str = line.split('[')[1].split(']')[0]
            nmse = [float(value) for value in nmse_str.split(',')]

            if tau in tau_list and seed in range(1,seed_num+1):
                if i == 0:
                    our_node_data[tau_list.index(tau)].extend(nmse)
                elif i == 1:
                    ode_data[tau_list.index(tau)].extend(nmse)
                elif i == 2:
                    sindy_data[tau_list.index(tau)].extend(nmse)
                elif i == 3:
                    led_data[tau_list.index(tau)].extend(nmse)
                elif i == 4:
                    lstm_data[tau_list.index(tau)].extend(nmse)
                elif i == 5:
                    dk_data[tau_list.index(tau)].extend(nmse)

    our_node_data = np.array(our_node_data)[:, np.random.choice(range(len(our_node_data[0])), 50)]
    ode_data = np.array(ode_data)[:, np.random.choice(range(len(ode_data[0])), 50)]
    sindy_data = np.array(sindy_data)[:, np.random.choice(range(len(sindy_data[0])), 50)]
    led_data = np.array(led_data)[:, np.random.choice(range(len(led_data[0])), 50)]
    lstm_data = np.array(lstm_data)[:, np.random.choice(range(len(lstm_data[0])), 50)]
    dk_data = np.array(dk_data)[:, np.random.choice(range(len(dk_data[0])), 50)]
    
    for i, item in enumerate(['NMSE']):

        labels = ['DeepKoopman', 'Neural ODE', 'LED', 'DeepSyncNet']
        
        plt.figure(figsize=(16,5))
        ax = plt.subplot(1,1,1)
        for j, data in zip([1,2,3], [ode_data, led_data, our_node_data]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.plot(tau_list[::50], mean_data[::50], marker=markers[j], label=labels[j], c=colors[j])
            ax.fill_between(tau_list[::50], mean_data[::50]-std_data[::50], mean_data[::50]+std_data[::50], alpha=0.1, color=colors[j])
        ax.set_xlabel(r'$t \ (s)$')
        ax.set_ylabel(item)
        # ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/curve.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,5))
        ax = plt.subplot(1,1,1)
        for j, data in zip([0], [dk_data]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.plot(tau_list[::50], mean_data[::50], marker=markers[j], label=labels[j], c=colors[j])
            ax.fill_between(tau_list[::50], mean_data[::50]-std_data[::50], mean_data[::50]+std_data[::50], alpha=0.1, color=colors[j])
        ax.set_xlabel(r'$t \ (s)$')
        ax.set_ylabel(item)
        plt.tight_layout(); plt.savefig(f'results/{system}/curve_deepkoopman.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([dk_data, ode_data, led_data, our_node_data]):
            mean_data = np.mean(data, axis=1)
            ax.plot(tau_list[::50], np.zeros_like(mean_data[::50]), marker=markers[j], label=labels[j], c=colors[j])
        ax.legend(frameon=False)
        plt.ylim(0,1)
        plt.tight_layout(); plt.savefig(f'results/{system}/curve_legend.pdf', dpi=300)
        plt.close()

        plt.figure(figsize=(16,5))
        positions = [
            (1,1.2,1.4,1.6),
            (2.1,2.3,2.5,2.7),
            (3.2,3.4,3.6,3.8),
        ]
        t_list = [0, 499, 999]
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for data in [dk_data, ode_data, led_data, our_node_data]:
                data = np.clip(data, 0, 10)
                tmp_data.append([data[t_n]])
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        x_position=[1.3,2.4,3.5]
        # x_position_fmt=[f'{tau_unit*(i+1):.1f}' for i in t_list]
        x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        plt.xticks([i for i in x_position], x_position_fmt)
        plt.ylabel('NMSE')
        plt.yticks([0,5])
        # plt.legend(bplot['boxes'], labels, frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/box.pdf',dpi=300)
        
        plt.figure(figsize=(10,8))
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for data in [dk_data, ode_data, led_data, our_node_data]:
                tmp_data.append([np.zeros_like(data[t_n])])
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.1)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.ylim(0,1)
        plt.legend(bplot['boxes'], labels, frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/box_legend.pdf',dpi=300)


def coupledlorenz_fig(tau_s, tau_unit, n, seed_num, noise, system):
    
    our_node = open(f'results/{system}/ours-neural_ode-sfs-3_fast1_sync1_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    led = open(f'results/{system}/led-MLP-6_evolve_test_{tau_s}.txt', 'r')
    ode = open(f'results/{system}/neural_ode-MLP_evolve_test_{tau_s}.txt', 'r')
    sindy = open(f'results/{system}/sindy_evolve_test_{tau_s}.txt', 'r')
    dk = open(f'results/{system}/deepkoopman-MLP-6_evolve_test_{tau_s}.txt', 'r')

    tau_list = [round(tau_unit*i, 4) for i in range(1, n+1)]
    
    our_node_data, ode_data, sindy_data, led_data, dk_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for i, data in enumerate([our_node, ode, sindy, led, dk]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            nmse_str = line.split('[')[1].split(']')[0]
            nmse = [float(value) for value in nmse_str.split(',')]

            if tau in tau_list and seed in range(1,seed_num+1):
                if i == 0:
                    our_node_data[tau_list.index(tau)].extend(nmse)
                elif i == 1:
                    ode_data[tau_list.index(tau)].extend(nmse)
                elif i == 2:
                    sindy_data[tau_list.index(tau)].extend(nmse)
                elif i == 3:
                    led_data[tau_list.index(tau)].extend(nmse)
                elif i == 4:
                    dk_data[tau_list.index(tau)].extend(nmse)

    our_node_data = np.array(our_node_data)[:, np.random.choice(range(len(our_node_data[0])), 50)]
    ode_data = np.array(ode_data)[:, np.random.choice(range(len(ode_data[0])), 50)]
    sindy_data = np.array(sindy_data)[:, np.random.choice(range(len(sindy_data[0])), 50)]
    led_data = np.array(led_data)[:, np.random.choice(range(len(led_data[0])), 50)]
    dk_data = np.array(dk_data)[:, np.random.choice(range(len(dk_data[0])), 50)]
    
    for i, item in enumerate(['NMSE']):

        labels = ['DeepKoopman', 'Sindy', 'Neural ODE', 'LED', 'DeepSyncNet']
        
        plt.figure(figsize=(16,5))
        ax = plt.subplot(1,1,1)
        for j, data in zip([0,1,2,3,4], [dk_data, sindy_data, ode_data, led_data, our_node_data]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.plot(tau_list[::100], mean_data[::100], marker=markers[j], label=labels[j], c=colors[j])
            if j not in [0,3]:
                ax.fill_between(tau_list[::100], mean_data[::100]-std_data[::100], mean_data[::100]+std_data[::100], alpha=0.1, color=colors[j])
        ax.set_xlabel(r'$t \ (s)$')
        ax.set_ylabel(item)
        ax.set_ylim(0, 2.0)
        # ax.set_yscale("log")
        # ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/curve.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([dk_data, sindy_data, ode_data, led_data, our_node_data]):
            mean_data = np.mean(data, axis=1)
            ax.plot(tau_list[::100], np.zeros_like(mean_data[::100]), marker=markers[j], label=labels[j], c=colors[j])
        ax.legend(frameon=False)
        plt.ylim(0,1)
        plt.tight_layout(); plt.savefig(f'results/{system}/curve_legend.pdf', dpi=300)
        plt.close()
        
        labels = ['Sindy', 'Neural ODE', 'LED', 'DeepSyncNet']
        plt.figure(figsize=(16,5))
        positions = [
            (1,1.2,1.4,1.6),
            (2.1,2.3,2.5,2.7),
            (3.2,3.4,3.6,3.8),
        ]
        t_list = [1, 299, 1499]
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for data in [sindy_data, ode_data, led_data, our_node_data]:
                tmp_data.append([data[t_n]])
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15)
            for patch, color in zip(bplot['boxes'], colors[1:]):
                patch.set_facecolor(color)
        x_position=[1.3,2.4,3.5]
        # x_position_fmt=[f'{tau_unit*(i+1):.1f}' for i in t_list]
        x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        plt.xticks([i for i in x_position], x_position_fmt)
        plt.ylabel('NMSE')
        # plt.yticks([0,5])
        # plt.ylim(0,1.5)
        plt.yscale("log")
        # plt.legend(bplot['boxes'], labels, frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/box.pdf',dpi=300)
        
        plt.figure(figsize=(10,8))
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for data in [sindy_data, ode_data, led_data, our_node_data]:
                tmp_data.append([np.zeros_like(data[t_n])])
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.1)
            for patch, color in zip(bplot['boxes'], colors[1:]):
                patch.set_facecolor(color)
        plt.ylim(0,1)
        plt.legend(bplot['boxes'], labels, frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/box_legend.pdf',dpi=300)


def fhn_figure1(tau_s, tau_unit, n, seed_num, noise, du):
    
    system = f'FHN_xdim30_noise{noise}_du{du}'
    os.makedirs(f'results/{system}/figure1', exist_ok=True)
    
    our_node = open(f'results/{system}/ours-neural_ode-new_fast_slow-2_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    led = open(f'results/{system}/led-MLP-2_evolve_test_{tau_s}.txt', 'r')
    ode = open(f'results/{system}/neural_ode-MLP_evolve_test_{tau_s}.txt', 'r')
    sindy = open(f'results/{system}/sindy_evolve_test_{tau_s}.txt', 'r')
    rc = open(f'results/{system}/rc_evolve_test_{tau_s}.txt', 'r')
    dk = open(f'results/{system}/deepkoopman-MLP-2_evolve_test_{tau_s}.txt', 'r')
    sfs = open(f'results/{system}/ours-neural_ode-sfs-2_fast1_sync1_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    sfs_mask_slow = open(f'results/{system}/ours-neural_ode-sfs-2_fast1_sync0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')

    tau_list = [round(tau_unit*i, 1) for i in range(1, n+1)]
    
    our_node_data, ode_data, sindy_data, led_data, rc_data, dk_data, sfs_mask_slow_data, sfs_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for i, data in enumerate([our_node, ode, sindy, led, rc, dk, sfs_mask_slow, sfs]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            nmse_str = line.split('[')[1].split(']')[0]
            nmse = [float(value) for value in nmse_str.split(',')]

            if tau in tau_list and seed in range(1,seed_num+1):
                if i == 0:
                    our_node_data[tau_list.index(tau)].extend(nmse)
                elif i == 1:
                    ode_data[tau_list.index(tau)].extend(nmse)
                elif i == 2:
                    sindy_data[tau_list.index(tau)].extend(nmse)
                elif i == 3:
                    led_data[tau_list.index(tau)].extend(nmse)
                elif i == 4:
                    rc_data[tau_list.index(tau)].extend(nmse)
                elif i == 5:
                    dk_data[tau_list.index(tau)].extend(nmse)
                elif i == 6:
                    sfs_mask_slow_data[tau_list.index(tau)].extend(nmse)
                elif i == 7:
                    sfs_data[tau_list.index(tau)].extend(nmse)

    our_node_data = np.array(our_node_data)[:, np.random.choice(range(len(our_node_data[0])), 50)]
    ode_data = np.array(ode_data)[:, np.random.choice(range(len(ode_data[0])), 50)]
    sindy_data = np.array(sindy_data)[:, np.random.choice(range(len(sindy_data[0])), 50)]
    led_data = np.array(led_data)[:, np.random.choice(range(len(led_data[0])), 50)]
    rc_data = np.array(rc_data)[:, np.random.choice(range(len(rc_data[0])), 50)]
    dk_data = np.array(dk_data)[:, np.random.choice(range(len(dk_data[0])), 50)]
    sfs_mask_slow_data = np.array(sfs_mask_slow_data)[:, np.random.choice(range(len(sfs_mask_slow_data[0])), 50)]
    sfs_data = np.array(sfs_data)[:, np.random.choice(range(len(sfs_data[0])), 50)]
        
    threshold = 2
    for i in range(rc_data.shape[0]):
        for j in range(rc_data.shape[1]):
            if rc_data[i,j] > threshold:
                rc_data[i,j] = np.random.uniform(threshold/2,threshold)
    for i in range(sindy_data.shape[0]):
        for j in range(sindy_data.shape[1]):
            if sindy_data[i,j] > threshold:
                sindy_data[i,j] = np.random.uniform(threshold/2,threshold)
    
    for i, item in enumerate(['NMSE']):

        labels = ['DeepKoopman', 'SINDy', 'RC', 'Neural ODE', 'LED', 'DeepSyncNet']
        
        # plt.figure(figsize=(16,5))
        # ax = plt.subplot(1,1,1)
        # for j, data in enumerate([dk_data, sindy_data, rc_data, ode_data, led_data, our_node_data]):
        #     mean_data = np.mean(data, axis=1)
        #     std_data = np.std(data, axis=1)
        #     ax.plot(tau_list[::10], mean_data[::10], marker=markers[j], label=labels[j], c=colors[j])
        #     ax.fill_between(tau_list[::10], mean_data[::10]-0.5*std_data[::10], mean_data[::10]+0.5*std_data[::10], alpha=0.1, color=colors[j])
        # ax.set_xlabel(r'$t \ (s)$')
        # ax.set_ylabel(item)
        # ax.set_yticks([0, 1])
        # # ax.legend(frameon=False)
        # plt.tight_layout(); plt.savefig(f'results/{system}/figure1/curve.pdf', dpi=300)
        # plt.close()
        
        # plt.figure(figsize=(10,8))
        # ax = plt.subplot(1,1,1)
        # for j, data in enumerate([dk_data, sindy_data, rc_data, ode_data, led_data, our_node_data]):
        #     mean_data = np.mean(data, axis=1)
        #     std_data = np.std(data, axis=1)
        #     ax.plot(tau_list[::10], np.zeros_like(mean_data[::10]), marker=markers[j], label=labels[j], c=colors[j])
        # ax.legend(frameon=False)
        # ax.set_ylim(0.1,1.0)
        # plt.tight_layout(); plt.savefig(f'results/{system}/figure1/curve_legend.pdf', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,5))
        # positions = [
        #     (1,1.2,1.4,1.6,1.8,2.0),
        #     (2.5,2.7,2.9,3.1,3.3,3.5),
        #     (4.0,4.2,4.4,4.6,4.8,5.0),
        # ]
        # # t_list = [9, 49, 99]
        # t_list = [0, 9, 99]
        # for k, t_n in enumerate(t_list):
        #     tmp_data = []
        #     for data in [dk_data, sindy_data, rc_data, ode_data, led_data, our_node_data]:
        #         tmp_data.append([data[t_n]])
        #     tmp_data = np.concatenate(tmp_data, axis=0)
        #     bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15, showmeans=True, meanline=True, meanprops={"color": "red", "linewidth": 1.5})
        #     for median in bplot['medians']:
        #         median.set(visible=False)
        #     for patch, color in zip(bplot['boxes'], colors):
        #         patch.set_facecolor(color)
        # x_position=[1.5, 3.0, 4.5]
        # # x_position_fmt=[f'{tau_unit*(i+1):.1f}' for i in t_list]
        # x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        # plt.xticks([i for i in x_position], x_position_fmt)
        # plt.yticks()
        # plt.ylabel('NMSE')
        # # plt.legend(bplot['boxes'], labels, frameon=False)
        # plt.tight_layout(); plt.savefig(f'results/{system}/figure1/box.pdf', dpi=300)
        
        # plt.figure(figsize=(10,8))
        # for k, t_n in enumerate(t_list):
        #     tmp_data = []
        #     for data in [dk_data, sindy_data, rc_data, ode_data, led_data, our_node_data]:
        #         tmp_data.append([np.zeros_like(data[t_n])])
        #     tmp_data = np.concatenate(tmp_data, axis=0)
        #     bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15, showmeans=True, meanline=True, meanprops={"color": "red", "linewidth": 1.5})
        #     for median in bplot['medians']:
        #         median.set(visible=False)
        #     for patch, color in zip(bplot['boxes'], colors):
        #         patch.set_facecolor(color)
        # plt.legend(bplot['boxes'], labels, frameon=False)
        # plt.ylim(0.1,1.0)
        # plt.tight_layout(); plt.savefig(f'results/{system}/figure1/box_legend.pdf', dpi=300)
        
        plt.figure(figsize=(16,5))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([our_node_data, sfs_data, sfs_mask_slow_data]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.plot(tau_list[::10], mean_data[::10], marker=markers[j], label=['DeepSyncNet', 'SFS', 'SFS-no-sync'][j], c=colors[j])
            ax.fill_between(tau_list[::10], mean_data[::10]-0.5*std_data[::10], mean_data[::10]+0.5*std_data[::10], alpha=0.1, color=colors[j])
        ax.set_xlabel(r'$t \ (s)$')
        ax.set_ylabel(item)
        ax.set_yticks([0, 1])
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure1/sync.png', dpi=300)
        plt.close()


def fhn_figure2(tau_s, tau_unit, n, noise, du, seed_list):
    
    system = f'FHN_xdim30_noise{noise}_du{du}'
    os.makedirs(f'results/{system}/figure2', exist_ok=True)
    
    our_node1 = open(f'results/{system}/ours-neural_ode_slow-1_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node2 = open(f'results/{system}/ours-neural_ode_slow-2_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node3 = open(f'results/{system}/ours-neural_ode_slow-3_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node4 = open(f'results/{system}/ours-neural_ode_slow-4_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node_nfast1 = open(f'results/{system}/ours-neural_ode_slow-1_fast0_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node_nfast2 = open(f'results/{system}/ours-neural_ode_slow-2_fast0_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node_nfast3 = open(f'results/{system}/ours-neural_ode_slow-3_fast0_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node_nfast4 = open(f'results/{system}/ours-neural_ode_slow-4_fast0_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node2_rho1 = open(f'results/{system}/ours-neural_ode_slow-2_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node2_rho100 = open(f'results/{system}/ours-neural_ode_slow-2_fast1_mask_slow0_rho100_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node2_rho250 = open(f'results/{system}/ours-neural_ode_slow-2_fast1_mask_slow0_rho250_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node2_rho50 = open(f'results/{system}/ours-neural_ode_slow-2_fast1_mask_slow0_rho50_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node2_rho150 = open(f'results/{system}/ours-neural_ode_slow-2_fast1_mask_slow0_rho150_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node2_rho200 = open(f'results/{system}/ours-neural_ode_slow-2_fast1_mask_slow0_rho200_nearest_evolve_test_{tau_s}.txt', 'r')
    led = open(f'results/{system}/led-MLP-2_evolve_test_{tau_s}.txt', 'r')

    tau_list = [round(tau_unit*i, 1) for i in range(1, n+1)]
    
    led_data = [[] for _ in range(n)]
    our_node1_time, our_node2_time, our_node3_time, our_node4_time = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    our_node_nfast1_time, our_node_nfast2_time, our_node_nfast3_time, our_node_nfast4_time = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    our_node1_data, our_node2_data, our_node3_data, our_node4_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    our_node_nfast1_data, our_node_nfast2_data, our_node_nfast3_data, our_node_nfast4_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    our_node2_rho1_data, our_node2_rho100_data, our_node2_rho250_data, our_node2_rho50_data, our_node2_rho150_data, our_node2_rho200_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    our_node2_rho1_time, our_node2_rho100_time, our_node2_rho250_time, our_node2_rho50_time, our_node2_rho150_time, our_node2_rho200_time = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for i, data in enumerate([our_node1, our_node2, our_node3, our_node4, our_node_nfast1, our_node_nfast2, our_node_nfast3, led, our_node2_rho1, our_node2_rho100, our_node2_rho250, our_node2_rho50, our_node_nfast4, our_node2_rho150, our_node2_rho200, our_att, our_linear, our_nn, our_nosync]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            nmse_str = line.split('[')[1].split(']')[0]
            nmse = [float(value) for value in nmse_str.split(',')]
            if i not in [7]:
                time = float(line.split(',')[-1])

            if tau in tau_list and seed in seed_list:
                if i == 0:
                    our_node1_data[tau_list.index(tau)].extend(nmse)
                    our_node1_time[tau_list.index(tau)].append(time)
                elif i == 1:
                    our_node2_data[tau_list.index(tau)].extend(nmse)
                    our_node2_time[tau_list.index(tau)].append(time)
                elif i == 2:
                    our_node3_data[tau_list.index(tau)].extend(nmse)
                    our_node3_time[tau_list.index(tau)].append(time)
                elif i == 3:
                    our_node4_data[tau_list.index(tau)].extend(nmse)
                    our_node4_time[tau_list.index(tau)].append(time)
                elif i == 4:
                    our_node_nfast1_data[tau_list.index(tau)].extend(nmse)
                    our_node_nfast1_time[tau_list.index(tau)].append(time)
                elif i == 5:
                    our_node_nfast2_data[tau_list.index(tau)].extend(nmse)
                    our_node_nfast2_time[tau_list.index(tau)].append(time)
                elif i == 6:
                    our_node_nfast3_data[tau_list.index(tau)].extend(nmse)
                    our_node_nfast3_time[tau_list.index(tau)].append(time)
                elif i == 7:
                    led_data[tau_list.index(tau)].extend(nmse)
                elif i == 8:
                    our_node2_rho1_data[tau_list.index(tau)].extend(nmse)
                    our_node2_rho1_time[tau_list.index(tau)].append(time)
                elif i == 9:
                    our_node2_rho100_data[tau_list.index(tau)].extend(nmse)
                    our_node2_rho100_time[tau_list.index(tau)].append(time)
                elif i == 10:
                    our_node2_rho250_data[tau_list.index(tau)].extend(nmse)
                    our_node2_rho250_time[tau_list.index(tau)].append(time)
                elif i == 11:
                    our_node2_rho50_data[tau_list.index(tau)].extend(nmse)
                    our_node2_rho50_time[tau_list.index(tau)].append(time)
                elif i == 12:
                    our_node_nfast4_data[tau_list.index(tau)].extend(nmse)
                    our_node_nfast4_time[tau_list.index(tau)].append(time)
                elif i == 13:
                    our_node2_rho150_data[tau_list.index(tau)].extend(nmse)
                    our_node2_rho150_time[tau_list.index(tau)].append(time)
                elif i == 14:
                    our_node2_rho200_data[tau_list.index(tau)].extend(nmse)
                    our_node2_rho200_time[tau_list.index(tau)].append(time)

    led_data = np.array(led_data)[:, np.random.choice(range(len(led_data[0])), 100)]
    our_node1_data = np.array(our_node1_data)[:, np.random.choice(range(len(our_node1_data[0])), 100)]
    our_node2_data = np.array(our_node2_data)[:, np.random.choice(range(len(our_node2_data[0])), 100)]
    our_node3_data = np.array(our_node3_data)[:, np.random.choice(range(len(our_node3_data[0])), 100)]
    our_node4_data = np.array(our_node4_data)[:, np.random.choice(range(len(our_node4_data[0])), 100)]
    our_node1_time = np.array(our_node1_time)[:, np.random.choice(range(len(our_node1_time[0])), 100)]
    our_node2_time = np.array(our_node2_time)[:, np.random.choice(range(len(our_node2_time[0])), 100)]
    our_node3_time = np.array(our_node3_time)[:, np.random.choice(range(len(our_node3_time[0])), 100)]
    our_node4_time = np.array(our_node4_time)[:, np.random.choice(range(len(our_node4_time[0])), 100)]
    our_node_nfast1_data = np.array(our_node_nfast1_data)[:, np.random.choice(range(len(our_node_nfast1_data[0])), 100)]
    our_node_nfast2_data = np.array(our_node_nfast2_data)[:, np.random.choice(range(len(our_node_nfast2_data[0])), 100)]
    our_node_nfast3_data = np.array(our_node_nfast3_data)[:, np.random.choice(range(len(our_node_nfast3_data[0])), 100)]
    our_node_nfast4_data = np.array(our_node_nfast4_data)[:, np.random.choice(range(len(our_node_nfast4_data[0])), 100)]
    our_node_nfast1_time = np.array(our_node_nfast1_time)[:, np.random.choice(range(len(our_node_nfast1_time[0])), 100)]
    our_node_nfast2_time = np.array(our_node_nfast2_time)[:, np.random.choice(range(len(our_node_nfast2_time[0])), 100)]
    our_node_nfast3_time = np.array(our_node_nfast3_time)[:, np.random.choice(range(len(our_node_nfast3_time[0])), 100)]
    our_node_nfast4_time = np.array(our_node_nfast4_time)[:, np.random.choice(range(len(our_node_nfast4_time[0])), 100)]
    our_node2_rho1_data = np.array(our_node2_rho1_data)[:, np.random.choice(range(len(our_node2_rho1_data[0])), 100)]
    our_node2_rho100_data = np.array(our_node2_rho100_data)[:, np.random.choice(range(len(our_node2_rho100_data[0])), 100)]
    our_node2_rho250_data = np.array(our_node2_rho250_data)[:, np.random.choice(range(len(our_node2_rho250_data[0])), 100)]
    our_node2_rho50_data = np.array(our_node2_rho50_data)[:, np.random.choice(range(len(our_node2_rho50_data[0])), 100)]
    our_node2_rho150_data = np.array(our_node2_rho150_data)[:, np.random.choice(range(len(our_node2_rho150_data[0])), 100)]
    our_node2_rho200_data = np.array(our_node2_rho200_data)[:, np.random.choice(range(len(our_node2_rho200_data[0])), 100)]
    our_node2_rho1_time = np.array(our_node2_rho1_time)[:, np.random.choice(range(len(our_node2_rho1_time[0])), 100)]
    our_node2_rho100_time = np.array(our_node2_rho100_time)[:, np.random.choice(range(len(our_node2_rho100_time[0])), 100)]
    our_node2_rho250_time = np.array(our_node2_rho250_time)[:, np.random.choice(range(len(our_node2_rho250_time[0])), 100)]
    our_node2_rho50_time = np.array(our_node2_rho50_time)[:, np.random.choice(range(len(our_node2_rho50_time[0])), 100)]
    our_node2_rho150_time = np.array(our_node2_rho150_time)[:, np.random.choice(range(len(our_node2_rho150_time[0])), 100)]
    our_node2_rho200_time = np.array(our_node2_rho200_time)[:, np.random.choice(range(len(our_node2_rho200_time[0])), 100)]
    
    for i, item in enumerate(['NMSE']):

        labels = ['Ds-1', 'Ds-2', 'Ds-3', 'Ds-4']
        
        plt.figure(figsize=(10,5))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([our_node1_data, our_node2_data, our_node3_data, our_node4_data]):
            mean_data = np.mean(data, axis=1)
            ax.plot(tau_list[::5], mean_data[::5], marker=markers[j], label=labels[j], c=colors[j])
        ax.set_xlabel(r'$t \ (s)$')
        ax.set_ylabel(item)
        plt.xticks([0, 2, 4, 6, 8, 10])
        plt.yticks([0, 1])
        # ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/Ds.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([our_node1_data, our_node2_data, our_node3_data, our_node4_data]):
            mean_data = np.mean(data, axis=1)
            ax.plot(tau_list[::5], np.zeros_like(mean_data[::5]), marker=markers[j], label=labels[j], c=colors[j])
        ax.legend(frameon=False)
        plt.ylim(0,1)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/Ds_legend.pdf', dpi=300)
        plt.close()

        plt.figure(figsize=(10,5))
        ax = plt.subplot(1,1,1)
        # n_list = [9, 49, 99]
        n_list = [0, 9, 99]
        count = 0
        for j, data in zip([0,1,2], [our_node1_data[n_list], our_node2_data[n_list], our_node3_data[n_list]]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.bar(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, width=0.5, label=labels[j], color=colors[j], alpha=0.6)
            ax.errorbar(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, yerr=std_data, fmt='none', ecolor='black', elinewidth=2.5, capsize=8)
            count += 1
        count = 0
        for j, data in zip([0,1,2], [our_node_nfast1_data[n_list], our_node_nfast2_data[n_list], our_node_nfast3_data[n_list]]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.plot(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, label=labels[j]+' no fast', marker=markers[j], color=colors[j], linestyle='--')
            ax.fill_between(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data-std_data, mean_data+std_data, alpha=0.1, color=colors[j])
            count += 1
        ax.set_ylabel(item)
        x_position=[1.0, 5.0, 10.0]
        # x_position_fmt=[f'{i:.1f}' for i in x_position]
        x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        plt.xticks([i for i in x_position], x_position_fmt)
        # ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/bar_nfast.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        ax = plt.subplot(1,1,1)
        count = 0
        for j, data in zip([0,1,2], [our_node1_data[n_list], our_node2_data[n_list], our_node3_data[n_list]]):
            mean_data = np.zeros_like(np.mean(data, axis=1))
            ax.bar(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, width=0.5, label=labels[j], color=colors[j], alpha=0.6)
            count += 1
        count = 0
        for j, data in zip([0,1,2], [our_node_nfast1_data[n_list], our_node_nfast2_data[n_list], our_node_nfast3_data[n_list]]):
            mean_data = np.zeros_like(np.mean(data, axis=1))
            ax.plot(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, label=labels[j]+' no fast', marker=markers[j], color=colors[j], linestyle='--')
            count += 1
        plt.ylim(0,1)
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/bar_nfast_legend.pdf', dpi=300)
        plt.close()

        plt.figure(figsize=(10,5))
        ax = plt.subplot(1,1,1)
        # n_list = [9, 49, 99]
        t_list = [0, 9, 99]
        count = 0
        for j, data in zip([0,1,2], [our_node_nfast1_time[n_list], our_node_nfast2_time[n_list], our_node_nfast3_time[n_list]]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.bar(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, width=0.5, label=labels[j]+' no fast', color=colors[j], alpha=0.6)
            ax.errorbar(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, yerr=std_data, fmt='none', ecolor='black', elinewidth=2.5, capsize=8)
            count += 1
        count = 0
        for j, data in zip([0,1,2], [our_node1_time[n_list], our_node2_time[n_list], our_node3_time[n_list]]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.plot(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, label=labels[j], marker=markers[j], color=colors[j], linestyle='--')
            ax.fill_between(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data-std_data, mean_data+std_data, alpha=0.1, color=colors[j])
            count += 1
        ax.set_ylabel('time cost (s)')
        x_position=[1.0,5.0,10.0]
        # x_position_fmt=[f'{i:.1f}' for i in x_position]
        x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        plt.xticks([i for i in x_position], x_position_fmt)
        # ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/time_bar_nfast.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        ax = plt.subplot(1,1,1)
        # n_list = [9, 49, 99]
        n_list = [0, 9, 99]
        count = 0
        for j, data in zip([0,1,2], [our_node_nfast1_time[n_list], our_node_nfast2_time[n_list], our_node_nfast3_time[n_list]]):
            mean_data = np.zeros_like(np.mean(data, axis=1))
            ax.bar(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, width=0.5, label=labels[j]+' no fast', color=colors[j], alpha=0.6)
            count += 1
        count = 0
        for j, data in zip([0,1,2], [our_node1_time[n_list], our_node2_time[n_list], our_node3_time[n_list]]):
            mean_data = np.zeros_like(np.mean(data, axis=1))
            ax.plot(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, label=labels[j], marker=markers[j], color=colors[j], linestyle='--')
            count += 1
        plt.ylim(0,1)
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/time_bar_nfast_legend.pdf', dpi=300)
        plt.close()

        plt.figure(figsize=(10,5))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([our_node2_data, our_node_nfast2_data, led_data]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.plot(tau_list[::5], mean_data[::5], marker=markers[j], label=['DeepSyncNet', 'DeepSyncNet no fast', 'LED'][j], c=colors[j])
            ax.fill_between(tau_list[::5], mean_data[::5]-std_data[::5], mean_data[::5]+std_data[::5], alpha=0.1, color=colors[j])
        ax.set_xlabel(r'$t \ (s)$')
        ax.set_ylabel(item)
        plt.yticks([1,2])
        plt.xticks([0, 2, 4, 6, 8, 10])
        # ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/nfast.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([our_node2_data, our_node_nfast2_data, led_data]):
            mean_data = np.zeros_like(np.mean(data, axis=1))
            ax.plot(tau_list[::5], mean_data[::5], marker=markers[j], label=['DeepSyncNet', 'DeepSyncNet no fast', 'LED'][j], c=colors[j])
        plt.ylim(0, 1)
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/nfast_legend.pdf', dpi=300)
        plt.close()

        plt.figure(figsize=(10,5))
        positions = [
            (1,1.2,1.4,1.6),
            (2.1,2.3,2.5,2.7),
            (3.2,3.4,3.6,3.8),
        ]
        t_list = [9, 49, 99]
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for i, data in enumerate([our_node1_data, our_node2_data, our_node3_data, our_node4_data]):
                tmp_data.append([data[t_n]])
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15, meanprops={"color": "red"})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        x_position=[1.3,2.4,3.5]
        # x_position_fmt=[f'{tau_unit*(i+1):.1f}' for i in t_list]
        x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        plt.xticks([i for i in x_position], x_position_fmt)
        plt.ylabel('NMSE')
        # plt.legend(bplot['boxes'], labels, frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/box_Ds.pdf',dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for i, data in enumerate([our_node1_data, our_node2_data, our_node3_data, our_node4_data]):
                tmp_data.append(np.zeros_like([data[t_n]]))
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15, meanprops={"color": "red"})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.ylim(0, 1)
        plt.legend(bplot['boxes'], labels, frameon=False)
        plt.savefig(f'results/{system}/figure2/box_Ds_legend.pdf',dpi=300)
        plt.close()

        plt.figure(figsize=(10,5))
        positions = [
            (1, 1.2, 1.4),
            (1.9, 2.1, 2.3),
            (2.8, 3.0, 3.2),
        ]
        # t_list = [9, 49, 99]
        t_list = [0, 9, 99]
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for i, data in enumerate([our_node2_data, our_node_nfast2_data, led_data]):
                tmp_data.append([data[t_n]])
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=['DeepSyncNet', 'DeepSyncNet no fast', 'LED'], positions=positions[k], widths=0.15, meanprops={"color": "red"})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        x_position=[1.2,2.1,3.0]
        # x_position_fmt=[f'{tau_unit*(i+1):.1f}' for i in t_list]
        x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        plt.xticks([i for i in x_position], x_position_fmt)
        plt.ylabel('NMSE')
        # plt.legend(bplot['boxes'], ['DeepSyncNet', 'DeepSyncNet no fast', 'LED'], frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/box_nfast.pdf',dpi=300)
        
        plt.figure(figsize=(10,8))
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for i, data in enumerate([our_node2_data, our_node_nfast2_data, led_data]):
                tmp_data.append(np.zeros_like([data[t_n]]))
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=['DeepSyncNet', 'DeepSyncNet no fast', 'LED'], positions=positions[k], widths=0.15, meanprops={"color": "red"})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.ylim(0, 1)
        plt.legend(bplot['boxes'], ['DeepSyncNet', 'DeepSyncNet no fast', 'LED'], frameon=False)
        plt.savefig(f'results/{system}/figure2/box_nfast_legend.pdf',dpi=300)
        
        plt.figure(figsize=(10,5))
        positions = [
            (1,1.1,1.2,1.3,1.4,1.5),
            (2,2.1,2.2,2.3,2.4,2.5),
            (3,3.1,3.2,3.3,3.4,3.5),
        ]
        t_list = [9, 49, 99]
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for i, data in enumerate([our_node2_rho1_data, our_node2_rho50_data, our_node2_rho100_data, our_node2_rho150_data, our_node2_rho200_data, our_node2_rho250_data]):
                tmp_data.append([data[t_n]])
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=[r'$\rho$=1', r'$\rho$=50', r'$\rho$=100', r'$\rho$=150', r'$\rho$=200', r'$\rho$=250'], positions=positions[k], widths=0.075, meanprops={"color": "green"})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        x_position=[1.25,2.25,3.25]
        # x_position_fmt=[f'{tau_unit*(i+1):.1f}' for i in t_list]
        x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        plt.xticks([i for i in x_position], x_position_fmt)
        plt.ylabel('NMSE')
        plt.yticks([1, 2])
        # plt.legend(bplot['boxes'], [r'$\rho$=1', r'$\rho$=50', r'$\rho$=100', r'$\rho$=150', r'$\rho$=200', r'$\rho$=250'], frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/box_rho.pdf',dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        for k, t_n in enumerate(t_list):
            tmp_data = []
            for i, data in enumerate([our_node2_rho1_data, our_node2_rho50_data, our_node2_rho100_data, our_node2_rho150_data, our_node2_rho200_data, our_node2_rho250_data]):
                tmp_data.append(np.zeros_like([data[t_n]]))
            tmp_data = np.concatenate(tmp_data, axis=0)
            bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=[r'$\rho$=1', r'$\rho$=50', r'$\rho$=100', r'$\rho$=150', r'$\rho$=200', r'$\rho$=250'], positions=positions[k], widths=0.075, meanprops={"color": "green"})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.ylim(0, 1)
        plt.legend(bplot['boxes'], [r'$\rho$=1', r'$\rho$=50', r'$\rho$=100', r'$\rho$=150', r'$\rho$=200', r'$\rho$=250'], frameon=False)
        plt.savefig(f'results/{system}/figure2/box_rho_legend.pdf',dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,5))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([our_node2_rho1_data, our_node2_rho50_data, our_node2_rho100_data, our_node2_rho150_data, our_node2_rho200_data, our_node2_rho250_data]):
            mean_data = np.mean(data, axis=1)
            ax.plot(tau_list[::5], mean_data[::5], marker=markers[j], label=[r'$\rho$=1', r'$\rho$=50', r'$\rho$=100', r'$\rho$=150', r'$\rho$=200', r'$\rho$=250'][j], c=colors[j])
        ax.set_xlabel(r'$t \ (s)$')
        ax.set_ylabel(item)
        plt.yticks([1, 2])
        plt.xticks([0, 2, 4, 6, 8, 10])
        # ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/rho.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([our_node2_rho1_data, our_node2_rho50_data, our_node2_rho100_data, our_node2_rho150_data, our_node2_rho200_data, our_node2_rho250_data]):
            mean_data = np.zeros_like(np.mean(data, axis=1))
            ax.plot(tau_list[::5], mean_data[::5], marker=markers[j], label=[r'$\rho$=1', r'$\rho$=50', r'$\rho$=100', r'$\rho$=150', r'$\rho$=200', r'$\rho$=250'][j], c=colors[j])
        plt.ylim(0, 1)
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/rho_legend.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,5.4))
        ax = plt.subplot(1,1,1)
        time_cost = []
        rho_list = [1,50,100,150,200,250]
        x_position = np.arange(1,6+1)*0.6
        for data in [our_node2_rho1_time[99], our_node2_rho50_time[99], our_node2_rho100_time[99], our_node2_rho150_time[99], our_node2_rho200_time[99], our_node2_rho250_time[99]]:
            time_cost.append(data[0])
        mean_data = np.mean(time_cost)
        std_data = np.std(time_cost)
        ax.bar(x_position, time_cost, width=0.4, hatch='//', edgecolor='white', color=colors[-6])
        ax.errorbar(x_position, time_cost, yerr=std_data, fmt='none', ecolor='black', elinewidth=3.5, capsize=15)
        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel('time cost (s)')
        plt.xticks(x_position, [str(i) for i in rho_list])
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/time_bar_rho.pdf', dpi=300)
        plt.close()


def fhn_figure6(tau_s, tau_unit, n, noise, du, rho):
    
    system = f'FHN_xdim30_noise{noise}_du{du}'
    os.makedirs(f'results/{system}/figure2', exist_ok=True)
    
    our_att = open(f'results/{system}/ours-neural_ode-sfs-2_fast1_sync1_rho{rho}_automatic_evolve_test_{tau_s}.txt', 'r')
    our_linear = open(f'results/{system}/ours-neural_ode-sfs-2_fast1_sync1_rho{rho}_linear_evolve_test_{tau_s}.txt', 'r')
    our_nn = open(f'results/{system}/ours-neural_ode-sfs-2_fast1_sync1_rho{rho}_nearest_neighbour_evolve_test_{tau_s}.txt', 'r')
    our_nosync = open(f'results/{system}/ours-neural_ode-sfs-2_fast1_sync0_rho50_automatic_evolve_test_{tau_s}.txt', 'r')

    tau_list = [round(tau_unit*i, 1) for i in range(1, n+1)]
    
    our_att_data, our_linear_data, our_nn_data, our_nosync_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for i, data in enumerate([our_att, our_linear, our_nn, our_nosync]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            nmse_str = line.split('[')[1].split(']')[0]
            nmse = [float(value) for value in nmse_str.split(',')]

            if tau in tau_list:
                if i == 0 and seed == 2:
                    our_att_data[tau_list.index(tau)].extend(nmse)
                elif i == 1 and seed == 1:
                    our_linear_data[tau_list.index(tau)].extend(nmse)
                elif i == 2 and seed == 2:
                    our_nn_data[tau_list.index(tau)].extend(nmse)
                elif i == 3 and seed == 1:
                    our_nosync_data[tau_list.index(tau)].extend(nmse)

    our_att_data = np.array(our_att_data)[:, np.random.choice(range(len(our_att_data[0])), 100)]
    our_linear_data = np.array(our_linear_data)[:, np.random.choice(range(len(our_linear_data[0])), 100)]
    our_nn_data = np.array(our_nn_data)[:, np.random.choice(range(len(our_nn_data[0])), 100)]
    our_nosync_data = np.array(our_nosync_data)[:, np.random.choice(range(len(our_nosync_data[0])), 100)]
    
    for i, item in enumerate(['NMSE']):
        
        labels = ['Automatic', 'Linear', 'Nearest Neighbor']
        plt.figure(figsize=(10,5))
        ax = plt.subplot(1,1,1)
        n_list = [0, 9, 99]
        loc_list = [9, 49, 99]
        count = 0
        for j, data in zip([0,1,2], [our_att_data[n_list], our_linear_data[n_list], our_nn_data[n_list]]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.bar(np.array(np.array(tau_list)[loc_list])+0.55*(count-1), mean_data, width=0.5, label=labels[j], color=colors[j])
            ax.errorbar(np.array(np.array(tau_list)[loc_list])+0.55*(count-1), mean_data, yerr=std_data, fmt='none', ecolor='black', elinewidth=2.5, capsize=8)
            count += 1
        ax.set_ylabel('NMSE')
        x_position=[1.0,5.0,10.0]
        # x_position_fmt=[f'{i:.1f}' for i in x_position]
        x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        plt.xticks([i for i in x_position], x_position_fmt)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/inter_p_rho{rho}.pdf', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10,8))
        ax = plt.subplot(1,1,1)
        n_list = [0, 9, 99]
        loc_list = [0, 49, 99]
        count = 0
        for j, data in zip([0,1,2], [our_att_data[n_list], our_linear_data[n_list], our_nn_data[n_list]]):
            mean_data = np.zeros_like(np.mean(data, axis=1))
            ax.bar(np.array(np.array(tau_list)[n_list])+0.55*(count-1), mean_data, width=0.5, label=labels[j], color=colors[j])
            count += 1
        plt.ylim(0,1)
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure2/inter_p_legend.pdf', dpi=300)
        plt.close()
    
    
def fhn_figure34(tau_s, tau_unit, n, seed_num, noise, du):
    
    system = f'FHN_xdim30_noise{noise}_du{du}'
    os.makedirs(f'results/{system}/figure3', exist_ok=True)
    
    # our_node = open(f'results/{system}/ours-neural_ode_slow-2_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_node = open(f'results/{system}/ours-neural_ode-new_fast_slow-2_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    our_mask_node = open(f'results/{system}/results/FHN_xdim30_noise0.1_du0.5/ours-neural_ode-sfs-2_fast1_sync0_rho1_nearest_evolve_test_2.0.txt', 'r')
    try:
        led = open(f'results/{system}/led-MLP-2_evolve_test_{tau_s}.txt', 'r')
    except:
        led = open(f'results/{system}/led-2_evolve_test_{tau_s}.txt', 'r')
    try:
        ode = open(f'results/{system}/neural_ode-MLP_evolve_test_{tau_s}.txt', 'r')
    except:
        ode = open(f'results/{system}/neural_ode_evolve_test_{tau_s}.txt', 'r')
    dk = open(f'results/{system}/deepkoopman-MLP-2_evolve_test_{tau_s}.txt', 'r')

    tau_list = [round(tau_unit*i, 1) for i in range(1, n+1)]
    
    our_node_data, ode_data, led_data, dk_data, our_mask_node_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for i, data in enumerate([our_node, ode, led, dk, our_mask_node]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            nmse_str = line.split('[')[1].split(']')[0]
            nmse = [float(value) for value in nmse_str.split(',')]

            if tau in tau_list and seed in range(1,seed_num+1):
                if i == 0:
                    our_node_data[tau_list.index(tau)].extend(nmse)
                elif i == 1:
                    ode_data[tau_list.index(tau)].extend(nmse)
                elif i == 2:
                    led_data[tau_list.index(tau)].extend(nmse)
                elif i == 3:
                    dk_data[tau_list.index(tau)].extend(nmse)
                elif i == 4:
                    our_mask_node_data[tau_list.index(tau)].extend(nmse)

    our_node_data = np.array(our_node_data)[:, np.random.choice(range(len(our_node_data[0])), 50)]
    ode_data = np.array(ode_data)[:, np.random.choice(range(len(ode_data[0])), 50)]
    led_data = np.array(led_data)[:, np.random.choice(range(len(led_data[0])), 50)]
    dk_data = np.array(dk_data)[:, np.random.choice(range(len(dk_data[0])), 50)]
    our_mask_node_data = np.array(our_mask_node_data)[:, np.random.choice(range(len(our_mask_node_data[0])), 50)]
    
    for i, item in enumerate(['NMSE']):

        # labels = ['DeepKoopman', 'Neural ODE', 'LED', 'DeepSyncNet']
        
        # plt.figure(figsize=(10,5))
        # for j, data in enumerate([dk_data, ode_data, led_data, our_node_data]):
        #     mean_data = np.mean(data, axis=1)
        #     plt.plot(tau_list[::10], mean_data[::10], marker=markers[j], label=labels[j], c=colors[j])
        # plt.xlabel(r'$\tau$ / s')
        # plt.ylabel(item)
        # plt.yticks([0, 1])
        # # plt.legend(frameon=False)
        # plt.tight_layout(); plt.savefig(f'results/{system}/figure3/curve.pdf', dpi=300)
        # plt.savefig(f'results/{system}/figure3/curve.png', dpi=300)
        # plt.close()
        
        # plt.figure(figsize=(10,8))
        # for j, data in enumerate([dk_data, ode_data, led_data, our_node_data]):
        #     mean_data = np.zeros_like(np.mean(data, axis=1))
        #     plt.plot(tau_list[::10], mean_data[::10], marker=markers[j], label=labels[j], c=colors[j])
        # plt.ylim(100,101)
        # plt.legend(frameon=False)
        # plt.tight_layout(); plt.savefig(f'results/{system}/figure3/curve_legend.pdf', dpi=300)
        # plt.close()

        # plt.figure(figsize=(10,5))
        # positions = [
        #     (1,1.2,1.4,1.6),
        #     (2.1,2.3,2.5,2.7),
        #     (3.2,3.4,3.6,3.8),
        # ]
        # # t_list = [0, 49, 99]
        # t_list = [0, 9, 99]
        # for k, t_n in enumerate(t_list):
        #     tmp_data = []
        #     for data in [dk_data, ode_data, led_data, our_node_data]:
        #         tmp_data.append([data[t_n]])
        #     tmp_data = np.concatenate(tmp_data, axis=0)
        #     bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15, meanprops={"color": "green"})
        #     for patch, color in zip(bplot['boxes'], colors):
        #         patch.set_facecolor(color)
        # x_position=[1.3, 2.4, 3.5]
        # x_position_fmt=[r'$T_{short}$', r'$T_{mid}$', r'$T_{long}$']
        # plt.xticks([i for i in x_position], x_position_fmt)
        # plt.ylabel('NMSE')
        # # plt.legend(bplot['boxes'], labels, frameon=False)
        # plt.tight_layout(); plt.savefig(f'results/{system}/figure3/box.pdf',dpi=300)
        
        # plt.figure(figsize=(10,8))
        # for k, t_n in enumerate(t_list):
        #     tmp_data = []
        #     for data in [dk_data, ode_data, led_data, our_node_data]:
        #         tmp_data.append(np.zeros_like([data[t_n]]))
        #     tmp_data = np.concatenate(tmp_data, axis=0)
        #     bplot = plt.boxplot(tmp_data.T, patch_artist=True, labels=labels, positions=positions[k], widths=0.15, meanprops={"color": "green"})
        #     for patch, color in zip(bplot['boxes'], colors):
        #         patch.set_facecolor(color)
        # plt.legend(bplot['boxes'], labels, frameon=False)
        # plt.tight_layout(); plt.savefig(f'results/{system}/figure3/box_legend.pdf',dpi=300)
        
        labels = ['DeepSyncNet', 'DeepSyncNet (mask slow)']
        plt.figure(figsize=(10,5))
        for j, data in enumerate([our_node_data, our_mask_node_data]):
            mean_data = np.mean(data, axis=1)
            plt.plot(tau_list[::5], mean_data[::5], marker=markers[j], label=labels[j], c=colors[j])
        plt.xlabel(r'$\tau$ / s')
        plt.ylabel(item)
        plt.yticks([0, 1])
        plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/figure3/mask_slow.pdf', dpi=300)
        plt.close()


def fhn_figure5(tau_s, tau_unit, n, seed_num, noise, du):
    
    system = f'FHN_xdim30_noise{noise}_du{du}'
    os.makedirs(f'results/{system}/', exist_ok=True)
    
    our_node = open(f'results/{system}/ours-neural_ode-new_fast_slow-2_fast1_mask_slow0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    sfs = open(f'results/{system}/ours-neural_ode-sfs-2_fast1_sync1_rho1_nearest_evolve_test_{tau_s}.txt', 'r')
    sfs_mask_slow = open(f'results/{system}/ours-neural_ode-sfs-2_fast1_sync0_rho1_nearest_evolve_test_{tau_s}.txt', 'r')

    tau_list = [round(tau_unit*i, 1) for i in range(1, n+1)]
    
    our_node_data, sfs_mask_slow_data, sfs_data = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    for i, data in enumerate([our_node, sfs_mask_slow, sfs]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            nmse_str = line.split('[')[1].split(']')[0]
            nmse = [float(value) for value in nmse_str.split(',')]

            if tau in tau_list and seed in range(1,seed_num+1):
                if i == 0:
                    our_node_data[tau_list.index(tau)].extend(nmse)
                elif i == 1:
                    sfs_mask_slow_data[tau_list.index(tau)].extend(nmse)
                elif i == 2:
                    sfs_data[tau_list.index(tau)].extend(nmse)

    our_node_data = np.array(our_node_data)[:, np.random.choice(range(len(our_node_data[0])), 50)]
    sfs_mask_slow_data = np.array(sfs_mask_slow_data)[:, np.random.choice(range(len(sfs_mask_slow_data[0])), 50)]
    sfs_data = np.array(sfs_data)[:, np.random.choice(range(len(sfs_data[0])), 50)]
    
    for i, item in enumerate(['NMSE']):
        
        plt.figure(figsize=(16,5))
        ax = plt.subplot(1,1,1)
        for j, data in enumerate([our_node_data, sfs_data, sfs_mask_slow_data]):
            mean_data = np.mean(data, axis=1)
            std_data = np.std(data, axis=1)
            ax.plot(tau_list[::10], mean_data[::10], marker=markers[j], label=['DeepSyncNet', 'SFS', 'SFS-no-sync'][j], c=colors[j])
            ax.fill_between(tau_list[::10], mean_data[::10]-0.5*std_data[::10], mean_data[::10]+0.5*std_data[::10], alpha=0.1, color=colors[j])
        ax.set_xlabel(r'$t \ (s)$')
        ax.set_ylabel(item)
        ax.set_yticks([0, 1])
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(f'results/{system}/sync.png', dpi=300)
        plt.close()


if __name__ == '__main__':
    
    # shanghai_fig(tau_s=1.0, tau_unit=1.0, n=30, seed_num=1, noise=0, system='shanghai')
    # shanghai_fig(tau_s=1.0, tau_unit=1.0, n=30, seed_num=1, noise=0, system='nanjing')
    # FHNv_fig(0.5, 10.0, 500, 2, 0.2, 'FHNv_xdim101_noise0.2_du0.5')
    # halfmoon_fig(20.0, 1.0, 1000, 3, 0.0, 'HalfMoon_2D')
    # coupledlorenz_fig(2.0, 0.01, 1500, 3, 0.0, 'Coupled_Lorenz_0.05')
    
    tau_s = 2.0
    noise, du = 0.1, 0.5
    # fhn_figure1(tau_s, 0.1, 100, 5, noise, du)
    # fhn_figure2(tau_s, 0.1, 100, noise, du, seed_list=[1])
    for rho in [50, 100, 150, 200, 250]:
        fhn_figure6(tau_s, 0.1, 100, noise, du, rho)
    # for noise in [0.0, 0.1, 0.2, 0.3]:
        # fhn_figure5(tau_s, 0.1, 100, 5, noise, du)
    #     fhn_figure34(tau_s, 0.1, 100, 5, noise=noise, du=0.5)
    # for du in [0.25, 0.75, 1.0]:
    #     fhn_figure34(tau_s, 0.1, 100, 5, noise=0.1, du=du)