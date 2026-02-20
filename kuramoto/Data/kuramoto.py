import os
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from scipy.integrate import odeint
from scipy.fft import rfft, rfftfreq

from util import *


def generate_network(graph_type, N, **kwargs):
    """
    生成指定的复杂网络拓扑 (ER/BA/WS)
    返回: 邻接矩阵 A (numpy array)
    """
    if graph_type == 'ER':
        # Erdos-Renyi: p=连接概率
        G = nx.erdos_renyi_graph(N, p=kwargs.get('p', 0.2))
    elif graph_type == 'BA':
        # Barabasi-Albert: m=新节点连接数
        G = nx.barabasi_albert_graph(N, m=kwargs.get('m', 3))
    elif graph_type == 'WS':
        # Watts-Strogatz: k=近邻数, p=重连概率
        G = nx.watts_strogatz_graph(N, k=kwargs.get('k', 4), p=kwargs.get('p', 0.1))
    else:
        raise ValueError("Unknown graph type")
    
    return nx.to_numpy_array(G)

def ode_func(y, t, N, A, omega, K, epsilon):
    """
    网络动力学方程 (Adaptive Kuramoto)
    y: 状态向量 [theta_1...theta_N, w_11...w_NN] (扁平化)
    """
    # 1. 解包状态变量
    theta = y[:N]              # 快变量: 相位
    W_flat = y[N:]             # 慢变量: 权重 (扁平化)
    W = W_flat.reshape(N, N)   # 还原为矩阵
    
    # 2. 计算相位差矩阵 (利用广播机制加速)
    diff_mat = theta[None, :] - theta[:, None]
    
    # 3. 快动力学: 相位演化 (Phase dynamics)
    coupling = np.sum(A * W * np.sin(diff_mat), axis=1)
    dtheta = omega + (K / N) * coupling
    
    # 4. 慢动力学: 权重自适应 (Hebbian learning)
    # 仅在有物理连接(A_ij=1)的地方演化
    dW = epsilon * (np.cos(diff_mat) - W)
    dW = dW * A  # Apply topology mask
    
    return np.concatenate([dtheta, dW.flatten()])



def generate_kuramoto_data(trace_num, total_t=200, dt=0.05, save=True, plot=False, xdim=50, K=50.0, epsilon=0.05, data_dir=None):
    # 生成网络结构
    np.random.seed(1)
    random.seed(1)
    N = xdim
    A = generate_network('BA', N, m=3, k=6, p=0.1)
    
    def solve_1_trace(trace_id, A, dt, plot=False):
        seed_everything(trace_id)
        
        # 1. 系统参数
        omega = np.random.normal(0, 1.0, N) # 自然频率分布

        # 3. 初始化状态
        theta0 = np.random.uniform(0, 2*np.pi, N) # 初始相位 (0 ~ 2pi)
        W0 = A.copy() * 0.5                       # 初始权重 (设为0.5)
        y0 = np.concatenate([theta0, W0.flatten()])

        # 4. 积分演化
        t = np.arange(0, total_t, dt)
        steps = len(t)
        sol = odeint(ode_func, y0, t, args=(N, A, omega, K, epsilon))

        # 提取结果
        theta_t = sol[:, :N]

        # --- 宏观序参量 R(t) ---
        # R(t) = |(1/N) * sum(exp(i * theta_j))|
        complex_order_param = np.mean(np.exp(1j * theta_t), axis=1)
        R_t = np.abs(complex_order_param)

        # 5. 数据后处理与可视化
        sin_theta = np.sin(theta_t)

        # --- 排序逻辑：基于FFT主频 ---
        dt = t[1] - t[0]
        node_freqs = []

        for i in range(N):
            y_signal = sin_theta[steps//2:, i] # 取后半段
            current_n = len(y_signal)
            yf = np.abs(rfft(y_signal))
            xf = rfftfreq(current_n, dt)
            peak_idx = np.argmax(yf[1:]) + 1 
            dom_freq = xf[peak_idx]
            node_freqs.append(dom_freq)

        node_freqs = np.array(node_freqs)
        sort_idx = np.argsort(node_freqs) # 按频率排序

        sorted_sin_theta = sin_theta[:, sort_idx]
        sorted_freqs = node_freqs[sort_idx]

        if plot:
            # --- 绘制节点动力学轨迹 ---
            fig = plt.figure(figsize=(40, 16))
            # 定义网格布局: 2行2列
            gs = fig.add_gridspec(2, 2, width_ratios=[60, 1], height_ratios=[1, 0.6], wspace=0.02, hspace=0.2)
            # 创建子图对象
            ax1 = fig.add_subplot(gs[0, 0])
            cax = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1) # 共享x轴，进一步保证对齐
            # --- Subplot 1: 节点轨迹热图 ---
            im = ax1.imshow(sorted_sin_theta.T, aspect='auto', cmap='twilight', extent=[0, total_t, 0, N], origin='lower')
            plt.colorbar(im, cax=cax, label=r'$\sin(\theta)$')
            ax1.set_ylabel('Node Index (Sorted by Frequency)')
            ax1.set_title(f'Node Dynamics ($N={N}, K={K}, \epsilon={epsilon}$)')
            plt.setp(ax1.get_xticklabels(), visible=False)
            # --- Subplot 2: 宏观序参量 R(t) ---
            ax2.plot(t, R_t, color='darkorange', linewidth=2, label=r'$R(t)$')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Order Parameter $R(t)$')
            ax2.set_ylim(-0.05, 1.05) # 固定Y轴范围 [0, 1]
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.set_title('Macroscopic Synchronization Level')
            ax2.legend(loc='lower right', frameon=False)
            # 强制设置 x 轴范围一致
            ax2.set_xlim(0, total_t)
            # --- 保存 ---
            plt.savefig(f'{data_dir}/{trace_id}_traj.png', dpi=200, bbox_inches='tight')
        
        return sorted_sin_theta
    
    if save and os.path.exists(f'{data_dir}/origin.npz'): 
        return
    
    os.makedirs(data_dir, exist_ok=True)
    
    trace = []
    for trace_id in tqdm(range(1, trace_num+1)):
        sol = solve_1_trace(trace_id, A, dt, plot=trace_id<10)
        trace.append(sol)
    
    if save: 
        np.savez(f'{data_dir}/origin.npz', trace=trace, dt=dt, T=total_t)





def generate_dataset_slidingwindow(tmp, trace_num, tau, is_print=False, sequence_length=None, data_dir=None, start_t=0.0, end_t=None, sliding_length=None, stride_t=None, only_test=False, horizon=None):
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
    data = data[..., np.newaxis].transpose(0,1,3,2) # add channel dim
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
        
        # sliding window
        sequences = [data_item[:, i:j:int(window_size)] for i, j in zip(idxs, idxs+int(window_size)*(sequence_length-1)+1)]
        sequences = np.transpose(sequences, (1,0,2,3,4)) # (trace_num, sample_num, sequence_length, channel_num, feature_num)
        sequences = np.concatenate(sequences, axis=0) # (trace_num*sample_num, sequence_length, channel_num, feature_num)

        if is_print: 
            print(f'tau[{tau}]', f"{item} dataset (sequence_length={sequence_length}, window_size={window_size})", np.shape(sequences), '\n')

        # save
        if not seq_none:
            if horizon is not None:
                np.savez(data_dir+f'/{item}_{sequence_length}_{horizon}.npz', data=sequences)
            else:
                np.savez(data_dir+f'/{item}_{sequence_length}.npz', data=sequences)
        else:
            np.savez(data_dir+f'/{item}.npz', data=sequences)






if __name__ == "__main__":
    # 1. 系统参数
    N = 50              # 节点数量
    T_max = 200         # 演化时长
    steps = 20 * T_max  # 时间步数
    epsilon = 0.05      # 权重演化速率(慢变量)
    K = 50.0            # 耦合强度
    omega = np.random.normal(0, 1.0, N) # 自然频率分布
    net_type = 'BA'
    SEED = 5



    # 2. 生成网络结构
    np.random.seed(1)
    random.seed(1)
    A = generate_network(net_type, N, m=3, k=6, p=0.1)

    # 3. 初始化状态
    np.random.seed(SEED)
    random.seed(SEED)
    theta0 = np.random.uniform(0, 2*np.pi, N) # 初始相位 (0 ~ 2pi)
    W0 = A.copy() * 0.5                       # 初始权重 (设为0.5)
    y0 = np.concatenate([theta0, W0.flatten()])

    # 4. 积分演化
    t = np.linspace(0, T_max, steps)
    sol = odeint(ode_func, y0, t, args=(N, A, omega, K, epsilon))

    # 提取结果
    theta_t = sol[:, :N]

    # --- 宏观序参量 R(t) ---
    # R(t) = |(1/N) * sum(exp(i * theta_j))|
    complex_order_param = np.mean(np.exp(1j * theta_t), axis=1)
    R_t = np.abs(complex_order_param)

    # 5. 数据后处理与可视化
    sin_theta = np.sin(theta_t)

    # --- 排序逻辑：基于FFT主频 ---
    dt = t[1] - t[0]
    node_freqs = []

    for i in range(N):
        y_signal = sin_theta[steps//2:, i] # 取后半段
        current_n = len(y_signal)
        yf = np.abs(rfft(y_signal))
        xf = rfftfreq(current_n, dt)
        peak_idx = np.argmax(yf[1:]) + 1 
        dom_freq = xf[peak_idx]
        node_freqs.append(dom_freq)

    node_freqs = np.array(node_freqs)
    sort_idx = np.argsort(node_freqs) # 按频率排序

    sorted_sin_theta = sin_theta[:, sort_idx]
    sorted_freqs = node_freqs[sort_idx]

    print(f"频率范围: {sorted_freqs.min():.4f} - {sorted_freqs.max():.4f} Hz")










    # --- 绘制节点动力学轨迹 ---
    fig = plt.figure(figsize=(20, 8))
    # 定义网格布局: 2行2列
    gs = fig.add_gridspec(2, 2, width_ratios=[60, 1], height_ratios=[1, 0.6], wspace=0.02, hspace=0.2)
    # 创建子图对象
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1) # 共享x轴，进一步保证对齐
    # --- Subplot 1: 节点轨迹热图 ---
    im = ax1.imshow(sorted_sin_theta.T, aspect='auto', cmap='twilight', 
            extent=[0, T_max, 0, N], origin='lower')
    # 关键: 将 colorbar 画在指定的 cax 上，而不是偷 ax1 的空间
    plt.colorbar(im, cax=cax, label=r'$\sin(\theta)$')
    ax1.set_ylabel('Node Index (Sorted by Frequency)')
    ax1.set_title(f'Node Dynamics ($N={N}, K={K}, \epsilon={epsilon}$)')
    # # 对数x轴
    # ax1.set_xscale('log')
    # 隐藏 ax1 的 x 轴刻度标签 (因为下面有 ax2 了)
    plt.setp(ax1.get_xticklabels(), visible=False)
    # --- Subplot 2: 宏观序参量 R(t) ---
    ax2.plot(t, R_t, color='darkorange', linewidth=2, label=r'$R(t)$')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Order Parameter $R(t)$')
    ax2.set_ylim(-0.05, 1.05) # 固定Y轴范围 [0, 1]
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('Macroscopic Synchronization Level')
    ax2.legend(loc='lower right', frameon=False)
    # 强制设置 x 轴范围一致
    ax2.set_xlim(0, T_max)
    # # 对数x轴
    # ax2.set_xscale('log')
    # --- 保存 ---
    plt.savefig(f'adaptive_kuramoto_{net_type}.png', dpi=300, bbox_inches='tight')






    # --- 频率分布图 ---
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_freqs, 'o-', markersize=3, label='synchronized frequencies')
    plt.plot(omega[sort_idx], 'x', markersize=5, label='natural frequencies')
    plt.legend(frameon=False)
    plt.xlabel('Node Index (Sorted)')
    plt.ylabel('Dominant Frequency (Hz)')
    plt.title('Frequency Profile')
    plt.grid(True)
    plt.savefig(f'frequency_profile_{net_type}.png', dpi=300, bbox_inches='tight')






    # --- 网络拓扑可视化 ---
    # 1. 获取最终时刻的权重矩阵 W_final
    W_final_flat = sol[-1, N:]         # 取出最后时刻的 W 部分
    W_final = W_final_flat.reshape(N, N)
    # 2. 为了让圆环上的节点按频率排列（视觉上形成连续的色块），我们需要重排矩阵
    # 使用上一段代码计算出的 sort_idx (按主频排序的索引)
    def reorder_matrix(Mat, idx):
        return Mat[idx, :][:, idx]
    # 重排邻接矩阵和权重矩阵，使得圆环上相邻的节点频率也相近
    A_sorted = reorder_matrix(A, sort_idx)
    W0_sorted = reorder_matrix(W0, sort_idx)
    W_final_sorted = reorder_matrix(W_final, sort_idx)
    freqs_sorted = node_freqs[sort_idx]
    # --- 绘图函数 ---
    def plot_circular_network(ax, Adj, Weights, Freqs, title):
        # 创建图对象
        G = nx.from_numpy_array(Adj) # 仅基于物理连接创建图
        # 设置布局: 圆环布局
        pos = nx.circular_layout(G)
        # --- 1. 绘制连边 (Edge) ---
        edges = G.edges()
        weights = []
        colors = []
        for u, v in edges:
            w = Weights[u, v]
            # 宽度映射: 绝对值越大越粗。乘以一个系数让线条明显点
            # 初始权重是0.5, 最终权重接近1.0或-1.0
            weights.append(3.0 * np.abs(w)) 
            # 颜色/透明度映射: 
            # 如果权重接近0，说明断开了，设为极高透明度
            # 权重越大，越不透明
            alpha_val = np.clip(np.abs(w), 0.05, 1.0) 
            colors.append((0.5, 0.5, 0.5, alpha_val)) # 灰色，带透明度
        nx.draw_networkx_edges(G, pos, ax=ax, width=weights, edge_color=colors)
        # --- 2. 绘制节点 (Node) ---
        # 颜色映射: 根据频率
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100, 
                            node_color=Freqs, cmap='plasma', 
                            edgecolors='black', linewidths=0.5)
        # 移除坐标轴
        ax.axis('off')
        ax.set_title(title, fontsize=14, pad=10)
    # --- 主绘图过程 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # 1. 绘制初始网络 (t=0)
    plot_circular_network(ax1, A_sorted, W0_sorted, freqs_sorted, f"Initial Network ($t=0$)\nUniform Weights ($w_{{ij}}={W0[A!=0][0]:.1f}$)")
    # 2. 绘制最终网络 (t=end)
    plot_circular_network(ax2, A_sorted, W_final_sorted, freqs_sorted, f"Final Network ($t={T_max}$)\nAdaptive Rewiring ($\epsilon={epsilon}$)")
    # 添加颜色条 (Colorbar) 用于说明节点频率
    sm = cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=freqs_sorted.min(), vmax=freqs_sorted.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label('Node Dominant Frequency (Hz)')
    plt.savefig(f'adaptive_kuramoto_topology_{net_type}.png', dpi=300, bbox_inches='tight')