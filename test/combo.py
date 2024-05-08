import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt


def conn_combo_finder(prev_comb, conn1_comb):
    # len_prev = len(prev_comb[0])
    combinations = []
    for combo in prev_comb:
        for i in conn1_comb:
            if i[0] not in combo:
                combinations.append(combo + i)

    return list(set(tuple(sorted(l)) for l in combinations))


def list_all_combinations(n_nodes):
    max_n_conn = n_nodes * (n_nodes - 1)
    nodes = list(np.arange(1, n_nodes + 1).astype(str))

    # set up dictionary for combinations
    all_combs = {'0': ['None']}
    for i in range(1, 7):
        all_combs[str(i)] = []

    # find all pairs: n_connection = 1
    for node in nodes:
        combs = [tuple([node + i]) for i in nodes if i != node]
        all_combs['1'].extend(combs)

    # n_connection = ...
    for i in range(2, max_n_conn + 1):
        all_combs[str(i)] = conn_combo_finder(all_combs[str(i - 1)], all_combs['1'])

    return all_combs


def convert_to_matrix(n_node, comb_dict):
    """
    :param n_node: number of nodes
    :param comb_dict: a dictionary that contains tuples of all possible combinations, ordered
    by the number of connections.
    :return: mat (n_comb, n_node, n_node)
    """

    n_combs = []
    for key, grp in comb_dict.items():
        n_combs.append(len(grp))
    n_comb = np.sum(n_combs)

    mat = np.zeros((n_comb, n_node, n_node)
                   )
    for j, (key, combo) in enumerate(comb_dict.items()):
        if j == 0:
            pass
        else:
            # print (key)
            start_idx = int(np.sum(n_combs[:j]))
            for i, l in enumerate(combo):
                pre_idx, post_idx = [int(l[0][0]), int(l[0][1])]
                mat[start_idx + i, pre_idx - 1, post_idx - 1] = 1

    return mat


def input_combo_finder(n_input, n_node):
    combination_list = np.array(list(itertools.product([0, 1], repeat=n_input * n_node)))
    mat = combination_list.reshape(combination_list.shape[0], n_node, n_input)

    return mat


def ReLu(x, theta=1):
    new_x = np.subtract(x, theta)

    return np.maximum(new_x, 0.0)


def jorge(x, d=10.0, theta=1):
    return (x - theta) / (1 - np.exp(-d * (x - theta)))


def sigmoid(x):
    return 100 / (1 + np.exp(-(x - 10)))


def drdt(r, Isyn):
    return -r + jorge(Isyn)
    # return -r + ReLu(Isyn)
    # return -r + sigmoid(Isyn)


def simulation(T, dt, Ibg, Iv, Im, inp_mat, w_mat, tau):
    # initialize firing rates: [PC, PV, SST, VIP]
    frs = np.zeros((len(inp_mat), 4, int(T / dt) + 1))

    # scale_apical = np.array([[1,1], [1, 1], [1,0.5], [1,1]])

    for i, inp_i in enumerate(tqdm.tqdm(inp_mat, position=0, desc='input type', leave=False, colour='green')):
        for t in tqdm.tqdm(range(int(T / dt)), position=1, desc='sim time', leave=False, colour='red'):
            Isyn = inp_i @ np.array([Iv[t], Im[t]]) + w_mat @ frs[i, :, t] + Ibg
            frs[i, :, t + 1] = frs[i, :, t] + (dt / tau) * drdt(r=frs[i, :, t], Isyn=Isyn)

        print(f'{i + 1}/{len(inp_mat)} done!')

    return frs


def main(windows, thresholds, Iv, Im, input_mat=None):
    # sim parameters
    dt = 1e-4  # 0.1 ms
    T = 4.0  # s

    tau_exc = 60e-3
    tau_inh = 2e-3

    # background input
    bg_exc = 4e-3
    bg_inh = 3e-3
    Ibg = np.array([bg_exc, bg_inh, bg_inh, bg_inh])
    tau = np.array([tau_exc, tau_inh, tau_inh, tau_inh])

    if input_mat:
        inp_mat = input_mat
    else:
        # input combination matrix
        inp_mat = input_combo_finder(2, 4)

    # synaptic weights
    w_str = 1.0
    npe_mat = np.array([
        [0, -w_str, -w_str, 0],
        [w_str, 0, 0, 0],
        [w_str, 0, 0, -w_str],
        [w_str, 0, -w_str, 0]
    ])

    firing_rates = simulation(T=T, dt=dt,
                              Ibg=Ibg, Iv=Iv, Im=Im,
                              inp_mat=inp_mat, w_mat=npe_mat,
                              tau=tau)
    syn_combs = search_idcs(frs=firing_rates,
                            windows=windows,
                            thresholds=thresholds)

    return firing_rates[:, :, 1:], syn_combs, npe_mat, inp_mat, Iv, Im, Ibg, tau


def search_idcs(frs, windows, thresholds):
    # time window for three conditions:
    match_window = frs[:, 0, range(*windows[0])].mean(axis=1)
    mismatch_window = frs[:, 0, range(*windows[1])].mean(axis=1)
    playback_window = frs[:, 0, range(*windows[2])].mean(axis=1)

    # indices for three conditions - nPE
    nPE_match = np.argwhere((match_window >= thresholds[0][0]) & (match_window < thresholds[0][1])).flatten()
    nPE_mismatch = np.argwhere(mismatch_window > thresholds[1]).flatten()
    nPE_playback = np.argwhere((playback_window >= thresholds[0][0]) & (playback_window < thresholds[0][1])).flatten()
    nPE_idcs = [nPE_match, nPE_mismatch, nPE_playback]

    # indices for three conditions - pPE
    pPE_match = np.argwhere((match_window >= thresholds[0][0]) & (match_window < thresholds[0][1])).flatten()
    pPE_mismatch = np.argwhere((mismatch_window >= thresholds[0][0]) & (mismatch_window < thresholds[0][1])).flatten()
    pPE_playback = np.argwhere(playback_window > thresholds[1]).flatten()
    pPE_idcs = [pPE_match, pPE_mismatch, pPE_playback]

    # list all synaptic combinations that show nPE
    syn_combs = [
        list(set.intersection(*map(set, nPE_idcs))),
        list(set.intersection(*map(set, pPE_idcs)))
    ]

    return syn_combs


def plot_PE_combo(frs, npe_idcs, inp_mat, ncol, nrow):
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(3 * ncol, 3 * nrow))
    axs = axs.flatten()
    for i, npe_idx in enumerate(npe_idcs):
        axs[i * 2].plot(frs[npe_idx, 0])
        axs[i * 2].set_ylim([0, 1])
        axs[i * 2].set_title(f'syn_combo # = {npe_idx}')
        axs[i * 2].set_axis_off()
        axs[i * 2 + 1].imshow(inp_mat[npe_idx], cmap='gray')
        axs[i * 2 + 1].set_xticks(np.arange(2))
        axs[i * 2 + 1].set_yticks(np.arange(4))
        axs[i * 2 + 1].set_xticklabels(['V', 'M'], fontsize=10)
        axs[i * 2 + 1].set_yticklabels(['PC', 'PV', 'SST', 'VIP'], fontsize=10)
        axs[i * 2 + 1].xaxis.tick_top()
    fig.tight_layout()

    return fig

# # sim parameters
# dt = 1e-4  # 0.1 ms
# T = 4.0  # s
#
# tau_exc = 60e-3
# tau_inh = 2e-3
#
# # background input
# bg_exc = 4e-3
# bg_inh = 3e-3
# Ibg = np.array([bg_exc, bg_inh, bg_inh, bg_inh])
# tau = np.array([tau_exc, tau_inh, tau_inh, tau_inh])
#
# # external input
# v_input = 2
# Iv = np.zeros(int(T / dt))
# Iv[int(1 / dt):int(1.5 / dt)] = v_input
# Iv[int(3 / dt):int(3.5 / dt)] = v_input
#
# m_input = 2
# Im = np.zeros(int(T / dt))
# Im[int(1 / dt):int(1.5 / dt)] = m_input
# Im[int(2 / dt):int(2.5 / dt)] = m_input
#
# # input combination matrix
# inp_mat = np.array([
#     [1,1],
#     [1,0],
#     [1,1],
#     [0,1]
# ])
#
# # synaptic weights
# w_str = 1.0
# npe_mat = np.array([
#     [0, -w_str, -w_str, 0],
#     [w_str, 0, 0, 0],
#     [w_str, 0, 0, -w_str],
#     [w_str, 0, 0, 0]
# ])
#
#
# # initialize firing rates: [PC, PV, SST, VIP]
# frs = np.zeros((len(inp_mat), 4, int(T / dt) + 1))
#
# frs = np.zeros((4, int(T / dt) + 1))
# Isyn = np.zeros(4)
# for t in tqdm.tqdm(range(int(T / dt)), position=1, desc='sim time', leave=False, colour='red'):
#     Isyn = inp_mat @ np.array([Iv[t], Im[t]]) + npe_mat @ frs[:, t] + Ibg
#     frs[:, t + 1] = frs[:, t] + (dt / tau) * drdt(r=frs[:, t], Isyn=Isyn)
#
# plt.plot(frs[0])
# plt.show()

# if __name__ == '__main__':
#     rates = main()
#     np.save('rates2.npy', rates)
