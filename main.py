from test import combo
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse

T = 2.0  # s
dt = 1e-3
steps = int(T / dt)

# external input
v_input = 1
Iv = np.zeros(int(T / dt))
Iv[int(0.1 * steps):int(0.3 * steps)] = v_input
Iv[int(0.4 * steps):int(0.6 * steps)] = v_input * 0.5
Iv[int(0.7 * steps):int(0.9 * steps)] = v_input

m_input = 1
Im = np.zeros(int(T / dt))
Im[int(0.1 * steps):int(0.3 * steps)] = m_input
Im[int(0.4 * steps):int(0.6 * steps)] = m_input
Im[int(0.7 * steps):int(0.9 * steps)] = m_input * 0.5

windows = [(int(0.1 * steps), int(0.3 * steps)),
           (int(0.4 * steps), int(0.6 * steps)),
           (int(0.7 * steps), int(0.9 * steps))
           ]
thresholds = [(0.0, 0.1), 0.4]

# frs, npe_idcs, npe_mat, inp_mat, Iv, Im, Ibg, tau = (
combo_dict, frs, inp_mat, pe_mats = combo.main(
    windows=windows,
    thresholds=thresholds,
    Iv=Iv, Im=Im, T=T, dt=dt
)

# # plot nPE combi<nations
# nPE_fig = combo.plot_PE_combo(frs=frs, npe_idcs=npe_idcs[0], inp_mat=inp_mat, ncol=6, nrow=5)
# plt.show()
# # plot pPE combinations
# pPE_fig = combo.plot_PE_combo(frs=frs, npe_idcs=npe_idcs[1], inp_mat=inp_mat, ncol=6, nrow=5)
# plt.show()
#
# steps = 10
# min_converge_time = 1.5
# T = steps * min_converge_time
# dt = 1e-4
# Ibgs = np.array([4e-3, 3e-3, 3e-3, 3e-3])
# taus = np.array([60e-3, 2e-3, 2e-3, 2e-3])
# # external input
# m_input = 2
# Im = np.zeros(int(T / dt))
# for i in range(steps):
#     Im[i * int(min_converge_time / dt):(i + 1) * int(min_converge_time / dt)] = i
# Iv = np.ones(int(T / dt)) * m_input
#
# nfrs = combo.simulation(T, dt,
#                         Ibg=Ibgs,
#                         Iv=Iv, Im=Im,
#                         inp_mat=inp_mat[npe_idcs[0]], w_mat=npe_mat,
#                         tau=taus)
#
# fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(3 * 5, 3 * 6))
# axs = axs.flatten()
# for i, npe_idx in enumerate(inp_mat[npe_idcs[0]]):
#     mean_nfrs = [np.mean(nfrs[i, 0, j * int(min_converge_time / dt):(j + 1) * int(min_converge_time / dt)]) for j in
#                  range(10)]
#     axs[i * 2].plot(mean_nfrs)
#     axs[i * 2].axvline(x=2, c='r', ls='--')
#     # axs[i * 2].plot(Im, nfrs[i, 0, 1:])
#     axs[i * 2].set_ylim([0, 4])
#     axs[i * 2].set_title(f'syn# = {npe_idcs[0][i]} \nmse ={np.round(mse(mean_nfrs, np.arange(10)), 3)}')
#     axs[i * 2 + 1].imshow(npe_idx, cmap='gray')
#     axs[i * 2 + 1].set_xticks(np.arange(2))
#     axs[i * 2 + 1].set_yticks(np.arange(4))
#     axs[i * 2 + 1].set_xticklabels(['V', 'M'], fontsize=10)
#     axs[i * 2 + 1].set_yticklabels(['PC', 'PV', 'SST', 'VIP'], fontsize=10)
#     axs[i * 2 + 1].xaxis.tick_top()
# fig.tight_layout()
# plt.show()
#
# # external input
# v_input = 2
# Iv = np.zeros(int(T / dt))
# for i in range(steps):
#     Iv[i * int(min_converge_time / dt):(i + 1) * int(min_converge_time / dt)] = i
# Im = np.ones(int(T / dt)) * v_input
#
# pfrs = combo.simulation(T, dt,
#                         Ibg=Ibgs,
#                         Iv=Iv, Im=Im,
#                         inp_mat=inp_mat[npe_idcs[1]], w_mat=npe_mat,
#                         tau=taus)
#
# fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(3 * 5, 3 * 6))
# axs = axs.flatten()
# for i, npe_idx in enumerate(inp_mat[npe_idcs[1]]):
#     mean_pfrs = [np.mean(pfrs[i, 0, j * int(min_converge_time / dt):(j + 1) * int(min_converge_time / dt)])
#                  for j in range(10)]
#     axs[i * 2].plot(mean_pfrs)
#     axs[i * 2].axvline(x=2, c='r', ls='--')
#     axs[i * 2].set_ylim([0, 4])
#     axs[i * 2].set_title(f'syn# = {npe_idcs[1][i]} \nmse ={np.round(mse(mean_pfrs, np.arange(10)), 3)}')
#     axs[i * 2 + 1].imshow(npe_idx, cmap='gray')
#     axs[i * 2 + 1].set_xticks(np.arange(2))
#     axs[i * 2 + 1].set_yticks(np.arange(4))
#     axs[i * 2 + 1].set_xticklabels(['V', 'M'], fontsize=10)
#     axs[i * 2 + 1].set_yticklabels(['PC', 'PV', 'SST', 'VIP'], fontsize=10)
#     axs[i * 2 + 1].xaxis.tick_top()
# fig.tight_layout()
# fig.show()


# # input patterns in matrix format: (nI, 4, 2)
# inp_mats = get_conn_combo(constraint=inp_template, change_val=1)
# # synapse patterns in matrix format: (nS, 4, 4)
# syn_mats = get_conn_combo(constraint=syn_template, change_val=-1)
# # input - synapse combinations that give rise to nPE and pPE: {syn_idx: [(in1, ip1), ..., (inN, ipN)]}
# combo_dict = {}
# for i, syn_idx in enumerate(syn_combos):
#     x1, x2 = inp_combos[i]
#     combo_dict[syn_idx] = list(itertools.product(x1, x2))
#
# T = 4
# dt = 1e-3
# # tau = np.array([2e-3] * 4)
# tau = np.array([20e-3, 20e-3, 20e-3, 20e-3])
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
# windows = [(int(1 / dt), int(1.5 / dt)),
#            (int(2 / dt), int(2.5 / dt)),
#            (int(3 / dt), int(3.5 / dt))
#            ]
# thresholds = [(0.0, 0.1), 0.5]
# #
#
# # select random synapse patterns
# syn_choices = np.random.choice(syn_combos, 1)[4]
# # select a random synapse pattern
# syn_choice = np.random.choice(syn_combos, 1)[0]
# # find the index of the selected random synapse pattern
# syn_idx = np.argwhere(syn_combos==syn_choice)[0][0]
# # select random input patterns for nPE and pPE
# inp_idx = np.random.choice(np.arange(len(combo_dict[syn_choice])), 1)[0]
# npe_idx, ppe_idx = combo_dict[syn_choice][inp_idx]
#
# frs_npe = np.zeros((4, int(T / dt) +1))
# frs_ppe = np.zeros((4, int(T / dt) +1))
# for t in range(int(T/dt)):
#     Isyn_npe = inp_mats[npe_idx] @ np.array([Iv[t], Im[t]]) + syn_mats[syn_choice] @ frs_npe[:, t]
#     frs_npe[:, t+1] = frs_npe[:, t] + (dt / tau) * drdt(r=frs_npe[:, t], Isyn=Isyn_npe)
#     Isyn_ppe = inp_mats[ppe_idx] @ np.array([Iv[t], Im[t]]) + syn_mats[syn_choice] @ frs_ppe[:, t]
#     frs_ppe[:, t + 1] = frs_ppe[:, t] + (dt / tau) * drdt(r=frs_ppe[:, t], Isyn=Isyn_ppe)
#
# fig, axs = plt.subplots(2,2)
# axs = axs.flat
# axs[0].plot(frs[syn_idx][ppe_idx][0], c='r')
# axs[0].set_title('pPE')
# axs[1].plot(frs[syn_idx][npe_idx][0], c='b')
# axs[1].set_title('nPE')
#
# axs[2].plot(frs_ppe[0], c='r')
# axs[2].set_title('pPE')
# axs[3].plot(frs_npe[0], c='b')
# axs[3].set_title('nPE')
#
# fig.suptitle(f'syn#{syn_choice}, inp#{ppe_idx}, {npe_idx}')
# fig.show()

# mean firing rate plots
def plot_mean_firing_rates(frs, windows):

    ppe_frs, npe_frs = frs
    window_equal, window_bu_greater, window_bu_less = windows

    ppe_stats, npe_stats = np.zeros((2, len(windows), 3))

    ppe_equal_frs = ppe_frs[slice(*window_equal)]
    ppe_stats[0] = get_pe_response_stats(ppe_equal_frs)

    ppe_BUgreater_frs = ppe_frs[slice(*window_bu_greater)]
    ppe_stats[1] = get_pe_response_stats(ppe_BUgreater_frs)

    ppe_BUless_frs = ppe_frs[slice(*window_bu_less)]
    ppe_stats[2] = get_pe_response_stats(ppe_BUless_frs)

    npe_equal_frs = npe_frs[slice(*window_equal)]
    npe_stats[0] = get_pe_response_stats(npe_equal_frs)

    npe_BUgreater_frs = npe_frs[slice(*window_bu_greater)]
    npe_stats[1] = get_pe_response_stats(npe_BUgreater_frs)

    npe_BUless_frs = npe_frs[slice(*window_bu_less)]
    npe_stats[2] = get_pe_response_stats(npe_BUless_frs)

    stats = [ppe_stats, npe_stats]
    pe_colors = ['#CA181D', '#2070B4']
    pe_labels = ['pPE circuit', 'nPE circuit']

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all', figsize=(10, 10))
    for i, ax in enumerate(axs):
        ax.bar(x=[0, 1, 2], height=stats[i][:, 0], yerr=stats[i][:, 1], color=pe_colors[i])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, 1])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['BU = TD', 'BU > TD', 'BU < TD'])
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel('Firing rate (a.u.)')
        ax.set_title(pe_labels[i])

        # ax.tick_params(width=2)
        # for axis in ['bottom', 'left']:
        #     ax.spines[axis].set_linewidth(2)

    return ppe_stats, npe_stats, fig


def get_pe_response_stats(fr):
    return np.mean(fr), np.std(fr), np.std(fr) / len(fr)


# frs = np.vstack(
#     frs[list(combo_dict.keys()).index(151)][20,0],
#     frs[list(combo_dict.keys()).index(151)][18,0]
# )
# plt.close('all')
# ppe_stats, npe_stats, fig = plot_mean_firing_rates(frs=fr1, windows=windows)
# fig.show()
# fig.savefig('conn151_inp20_inp18.png', dpi=600, bbox_inches='tight')