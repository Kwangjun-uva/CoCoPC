import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt
from tools import ReLu


inp_template = np.array([
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 1]
    ])

syn_template = np.array([
        [1,-1,-1, 0],
        [1, 0, 0, 0],
        [1, 0, 0,-1],
        [1, 0, 0, 0]
    ])

def conn_combo_finder(prev_comb, conn1_comb):
    # len_prev = len(prev_comb[0])
    combinations = []
    for combo in prev_comb:
        for i in conn1_comb:
            if i[0] not in combo:
                combinations.append(combo + i)

    return list(set(tuple(sorted(l)) for l in combinations))


# def list_all_combinations(n_nodes):
#     max_n_conn = n_nodes * (n_nodes - 1)
#     nodes = list(np.arange(1, n_nodes + 1).astype(str))
#
#     # set up dictionary for combinations
#     all_combs = {'0': ['None']}
#     for i in range(1, 7):
#         all_combs[str(i)] = []
#
#     # find all pairs: n_connection = 1
#     for node in nodes:
#         combs = [tuple([node + i]) for i in nodes if i != node]
#         all_combs['1'].extend(combs)
#
#     # n_connection = ...
#     for i in range(2, max_n_conn + 1):
#         all_combs[str(i)] = conn_combo_finder(all_combs[str(i - 1)], all_combs['1'])
#
#     return all_combs
#
#
# def convert_to_matrix(n_node, comb_dict):
#     """
#     :param n_node: number of nodes
#     :param comb_dict: a dictionary that contains tuples of all possible combinations, ordered
#     by the number of connections.
#     :return: mat (n_comb, n_node, n_node)
#     """
#
#     n_combs = []
#     for key, grp in comb_dict.items():
#         n_combs.append(len(grp))
#     n_comb = np.sum(n_combs)
#
#     mat = np.zeros((n_comb, n_node, n_node)
#                    )
#     for j, (key, combo) in enumerate(comb_dict.items()):
#         if j == 0:
#             pass
#         else:
#             # print (key)
#             start_idx = int(np.sum(n_combs[:j]))
#             for i, l in enumerate(combo):
#                 pre_idx, post_idx = [int(l[0][0]), int(l[0][1])]
#                 mat[start_idx + i, pre_idx - 1, post_idx - 1] = 1
#
#     return mat
#
# #
# def input_combo_finder(n_input, n_node):
#     # all possible combinations
#     combination_list = np.array(list(itertools.product([0, 1], repeat=n_input * n_node)))
#     # reshape to (n_combination, n_node, n_input)
#     mat = combination_list.reshape(combination_list.shape[0], n_node, n_input)
#
#     return mat
def drdt(r, Isyn):
    # return -r + jorge(Isyn)
    return -r + ReLu(Isyn)
    # return -r + sigmoid(Isyn)


def simulation(T, dt, Ibg, Iv, Im, inp_mat, w_mat, tau, progress_num, total_num):
    # initialize firing rates: [PC, PV, SST, VIP]
    frs = np.zeros((len(inp_mat), 4, int(T / dt) + 1))

    # scale_apical = np.array([[1,1], [1, 1], [1,0.5], [1,1]])

    for i, inp_i in enumerate(tqdm.tqdm(inp_mat, position=0, desc=f'#{progress_num + 1} / {total_num}', leave=False, colour='green')):
        # for t in tqdm.tqdm(range(int(T / dt)), position=1, desc='sim time', leave=False, colour='red'):
        for t in range(int(T / dt)):
            Isyn = inp_i @ np.array([Iv[t], Im[t]]) + w_mat @ frs[i, :, t] + Ibg
            frs[i, :, t + 1] = frs[i, :, t] + (dt / tau) * drdt(r=frs[i, :, t], Isyn=Isyn)

        # print(f'{i + 1}/{len(inp_mat)} done!')

    return frs


def main(windows, thresholds, Iv, Im, T=6, dt=1e-3, tau_exc=20e-3, tau_inh=20e-3, bg_exc=0.0, bg_inh=0.0):

    # background input
    Ibg = np.array([bg_exc, bg_inh, bg_inh, bg_inh])
    tau = np.array([tau_exc, tau_inh, tau_inh, tau_inh])

    # set input pattern
    inp_mat = get_conn_combo(constraint=inp_template, change_val=1)

    # synaptic weights
    # get all possible synaptic combinations
    pe_mats = get_conn_combo(constraint=syn_template, change_val=-1)

    # syn_comb_number = []
    # inp_comb_number = []
    frs = []
    combo_dict = {}

    # loop over synaptic combos
    for mat_i, pe_mat in enumerate(pe_mats):
        # simulate on all input combination patterns
        firing_rates = simulation(T=T, dt=dt,
                                  Ibg=Ibg, Iv=Iv, Im=Im,
                                  inp_mat=inp_mat, w_mat=pe_mat,
                                  tau=tau, progress_num=mat_i, total_num=len(pe_mats))

        npe_comb, ppe_comb = search_idcs(frs=firing_rates,
                                         windows=windows,
                                         thresholds=thresholds)

        if (len(npe_comb) >= 1) and (len(ppe_comb) >= 1):

            combo_dict[mat_i] = list(itertools.product(npe_comb, ppe_comb))
            # inp_comb_number.append(inp_combs)
            # syn_comb_number.append(mat_i)
            frs.append(firing_rates[:, :, 1:])

    return combo_dict, frs, inp_mat, pe_mats

    # return firing_rates[:, :, 1:], syn_combs, npe_mat, inp_mat, Iv, Im, Ibg, tau


def search_idcs(frs, windows, thresholds):

    # time window for three conditions:
    # take only pyramidal cell FR
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


    # # input - synapse combinations that give rise to nPE and pPE: {syn_idx: [(in1, ip1), ..., (inN, ipN)]}
    # combo_dict = {}
    # for i, syn_idx in enumerate(syn_combos):
    #     x1, x2 = inp_combos[i]
    #     combo_dict[syn_idx] = list(itertools.product(x1, x2))

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


def get_conn_combo(constraint, change_val=1):

    conn_change_idcs = np.argwhere(constraint == 0)
    conn_combination_list = np.array(list(itertools.product([0, change_val], repeat=len(conn_change_idcs))))

    conn_mat = np.stack([constraint] * len(conn_combination_list))
    for i, conn_combo in enumerate(conn_combination_list):
        for j, (x_idx, y_idx) in enumerate(conn_change_idcs):
            conn_mat[i][x_idx, y_idx] = conn_combo[j]

    return conn_mat

# combo_frs = (nConn_mat, nInp_mat, nNeurons)
# nConn_mat : connection matrices that passed the combinatorial search criteria (i.e., generate pPE and nPE)
# nInp_mat : ALL possible input matrices
# nNeurons: 4 cortical neurons (pyr, PV, SST, VIP)
#
# combos = {conn_mat_idx : list of inp_mat pairs}
# e.g., 166: [(54, 10), (54, 15), (54, 7)]
# 166 is the connectivity matrix index, which can be retrieved from pe_mats
# 54, 10, 15, 7 are input matrix indices, which can be retrieved from inp_mat
#
# pe_mats : ALL possible connectivity matrices
# inp_mat : ALL possible input matrices