import numpy as np
import itertools
import tqdm
import matplotlib.pyplot as plt
from tools import ReLu

#
# # define template matrices for input and conectivity patterns
#
# # synaptic input matrix
# # col1: bottom-up, col2: top-down
# # row1: to Pyr, row2: to PV, row3: to SST, row4: to VIP
# # e.g., the first row can be interpreted as pyramidal cell receiving bottom-up input but not top-down
# inp_template = np.array([
#         [1, 0],
#         [0, 0],
#         [0, 0],
#         [0, 1]
#     ])
#
# # PE microcircuit configurations
# # 1: excitatory, -1: inhibitory
# # synaptic connections between neuron types (Pyr, PV, SST, VIP) within a microcircuit
# # row: receiving input from, col: sending input to
# # e.g., the third row can be interpreted as SST cell receiving Pyr excitatory and VIP inhibitory inputs
# syn_template = np.array([
#         [1,-1,-1, 0],
#         [1, 0, 0, 0],
#         [1, 0, 0,-1],
#         [1, 0, 0, 0]
#     ])
#
# def drdt(r, Isyn):
#     """
#     :param r: firing rate
#     :param Isyn: synatpic input
#     :return gradient for the activity update
#     """
#     return -r + ReLu(Isyn)
#
#
# def simulation(T, dt, Ibg, Iv, Im, inp_mat, w_mat, tau, progress_num, total_num):
#     """
#     Given bottom-up (Iv) and top-down (Im) inputs, microcircuit activities are simulated over T/dt steps
#
#     :param T: simulation duration
#     :param dt: step size (or time resolution) of simulation
#     :param Ibg: background input
#     :param Iv: bottom-up input
#     :param Im: top-down input
#     :param inp_mat: synaptic input pattern to PE microcircuit
#     :param w_mat: connectivity pattern within PE microcircuit
#     :param tau: membrane constant; [tau_exc, tau_inh, tau_inh, tau_inh]
#     :param progress_num: i-th connectivity pattern (only for keeping track of search progress)
#     :param total_num: total number of connectivity patterns (only for keeping track of search progress)
#     :return: firing rates of
#     """
#     # initialize firing rates: [PC, PV, SST, VIP]
#     frs = np.zeros((len(inp_mat), 4, int(T / dt) + 1))
#     for i, inp_i in enumerate(
#             tqdm.tqdm(
#                 inp_mat, position=0,
#                 desc=f'Searching for combinations: connectivity pattern #{progress_num + 1} / {total_num}',
#                 leave=False, colour='green'
#             )
#     ):
#         for t in range(int(T / dt)):
#             Isyn = inp_i @ np.array([Iv[t], Im[t]]) + w_mat @ frs[i, :, t] + Ibg
#             frs[i, :, t + 1] = frs[i, :, t] + (dt / tau) * drdt(r=frs[i, :, t], Isyn=Isyn)
#
#     return frs
#
# def main(windows, thresholds, Iv, Im, T=6, dt=1e-3, tau_exc=20e-3, tau_inh=20e-3, bg_exc=0.0, bg_inh=0.0):
#     """
#     :param windows: time windows during which match (bu=td), mismatch (bu>td), or playback (bu<td) occurs
#     :param thresholds: threshold conditions for pPE or nPE responses
#     :param Iv: bottom-up input
#     :param Im: top-down input
#     :param T: simulation duration
#     :param dt: step size (or time resolution) of simulation
#     :param tau_exc: membrane constant for excitatory neurons
#     :param tau_inh: membrane constant for inhibitory neurons
#     :param bg_exc: background noise for excitatory neurons
#     :param bg_inh: background noise for inhibitory neurons
#     :return: a dictionary containing input and synaptic pattern combinations that give rise to
#     pPE or nPE response, all possible input patterns, all possible synaptic patterns
#     """
#
#     # background input
#     Ibg = np.array([bg_exc, bg_inh, bg_inh, bg_inh])
#     tau = np.array([tau_exc, tau_inh, tau_inh, tau_inh])
#
#     # set input pattern
#     inp_mat = get_conn_combo(constraint=inp_template, change_val=1)
#
#     # synaptic weights
#     # get all possible synaptic combinations
#     syn_mats = get_conn_combo(constraint=syn_template, change_val=-1)
#
#     # create a dictionary for synaptic-input pattern combinations
#     combo_dict = {}
#
#     # loop over synaptic combos
#     for syn_mat_i, syn_mat in enumerate(syn_mats):
#         # loop over all input combination patterns (loop inside the simulation function)
#         firing_rates = simulation(T=T, dt=dt,
#                                   Ibg=Ibg, Iv=Iv, Im=Im,
#                                   inp_mat=inp_mat, w_mat=syn_mat,
#                                   tau=tau, progress_num=syn_mat_i, total_num=len(syn_mats))
#         # find input patterns that satisfy pPE or nPE response patterns
#         npe_comb, ppe_comb = search_idcs(frs=firing_rates,
#                                          windows=windows,
#                                          thresholds=thresholds)
#         # if such pattern exists, add to the combination dictionary
#         if (len(npe_comb) >= 1) and (len(ppe_comb) >= 1):
#
#             # combo_dict[mat_i] = list(itertools.product(npe_comb, ppe_comb))
#             # frs.append(firing_rates[:, :, 1:])
#             pe_idcs = list(itertools.product(npe_comb, ppe_comb))
#             for iter_idx, (ppe_inp_idx, npe_inp_idx) in enumerate(pe_idcs):
#                 combo_dict[syn_mat_i, ppe_inp_idx, npe_inp_idx] = np.vstack(
#                     [firing_rates[ppe_inp_idx, :, 1:], firing_rates[npe_inp_idx, :, 1:]]
#                 )
#
#     return combo_dict, inp_mat, syn_mats
#     # return combo_dict, frs, inp_mat, syn_mats
#
#
# def search_idcs(frs, windows, thresholds):
#
#     # time window for three conditions:
#     # take only pyramidal cell FR
#     match_window = frs[:, 0, range(*windows[0])].mean(axis=1)
#     mismatch_window = frs[:, 0, range(*windows[1])].mean(axis=1)
#     playback_window = frs[:, 0, range(*windows[2])].mean(axis=1)
#
#     # thresholds
#     (nr_min, nr_max), r_thres = thresholds
#
#     # indices for three conditions - nPE
#     nPE_match = np.argwhere((match_window >= nr_min) & (match_window < nr_max)).flatten()
#     nPE_mismatch = np.argwhere(mismatch_window > r_thres).flatten()
#     nPE_playback = np.argwhere((playback_window >= nr_min) & (playback_window < nr_max)).flatten()
#     nPE_idcs = [nPE_match, nPE_mismatch, nPE_playback]
#
#     # indices for three conditions - pPE
#     pPE_match = np.argwhere((match_window >= nr_min) & (match_window < nr_max)).flatten()
#     pPE_mismatch = np.argwhere((mismatch_window >= nr_min) & (mismatch_window < nr_max)).flatten()
#     pPE_playback = np.argwhere(playback_window > r_thres).flatten()
#     pPE_idcs = [pPE_match, pPE_mismatch, pPE_playback]
#
#     # list all synaptic combinations that show nPE
#     syn_combs = [
#         list(set.intersection(*map(set, nPE_idcs))),
#         list(set.intersection(*map(set, pPE_idcs)))
#     ]
#
#     return syn_combs


# def plot_PE_combo(frs, npe_idcs, inp_mat, ncol, nrow):
#
#     fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(3 * ncol, 3 * nrow))
#     axs = axs.flatten()
#     for i, npe_idx in enumerate(npe_idcs):
#         axs[i * 2].plot(frs[npe_idx, 0])
#         axs[i * 2].set_ylim([0, 1])
#         axs[i * 2].set_title(f'syn_combo # = {npe_idx}')
#         axs[i * 2].set_axis_off()
#         axs[i * 2 + 1].imshow(inp_mat[npe_idx], cmap='gray')
#         axs[i * 2 + 1].set_xticks(np.arange(2))
#         axs[i * 2 + 1].set_yticks(np.arange(4))
#         axs[i * 2 + 1].set_xticklabels(['V', 'M'], fontsize=10)
#         axs[i * 2 + 1].set_yticklabels(['PC', 'PV', 'SST', 'VIP'], fontsize=10)
#         axs[i * 2 + 1].xaxis.tick_top()
#     fig.tight_layout()
#
#     return fig


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

class run_combinatorial_search(object):

    def __init__(self, total_sim_time=6.0, sim_resolution=1e-3, tau_e=20e-3, tau_i=20e-3, bg_e=0.0, bg_i=0.0):

        self.T = total_sim_time
        self.dt = sim_resolution
        self.steps = int(total_sim_time / sim_resolution)

        self.tau = np.array([tau_e] + [tau_i] * 3)
        self.Ibg = np.array([bg_e] + [bg_i] * 3)

        # define template matrices for input and conectivity patterns
        # Synaptic input matrix
        # col1: bottom-up, col2: top-down
        # row1: to Pyr, row2: to PV, row3: to SST, row4: to VIP
        # e.g., the first row can be interpreted as pyramidal cell receiving bottom-up input but not top-down
        inp_template = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])

        # PE microcircuit configurations
        # 1: excitatory, -1: inhibitory
        # synaptic connections between neuron types (Pyr, PV, SST, VIP) within a microcircuit
        # row: receiving input from, col: sending input to
        # e.g., the third row can be interpreted as SST cell receiving Pyr excitatory and VIP inhibitory inputs
        syn_template = np.array([
            [1, -1, -1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, -1],
            [1, 0, 0, 0]
        ])

        # create input and synaptic combinations
        self.input_patterns = get_conn_combo(constraint=inp_template, change_val=1)
        self.syn_patterns = get_conn_combo(constraint=syn_template, change_val=-1)

        # create input
        self.bu_input, self.td_input, self.windows, self.thresholds = self.create_input_patterns()

    def create_input_patterns(self, input_str=1.0, offset_str=0.5):

        # define intervals
        interval_start = np.arange(0.1 * self.steps, 1.0 * self.steps, 0.3 * self.steps).astype(int)
        input_duration = int(0.1 * self.steps)
        windows = [(interval_i, interval_i + input_duration) for interval_i in interval_start]

        # define under and overprediction interval order (at 0, bu = td)
        underprediction_interval = 1
        overprediction_interval = 2

        # initialize bu and td inputs
        bu_input, td_input = np.zeros((2, self.steps))

        # create bu and td input patterns
        for interval_idx, (t_start, t_end) in enumerate(windows):

            bu_input[slice(t_start, t_end)] = input_str - (offset_str * (interval_idx == overprediction_interval))
            td_input[slice(t_start, t_end)] = input_str - (offset_str * (interval_idx == underprediction_interval))

        # define thresholds for mean activity within the interval
        # if mean activity within the interval falls between the first tuple, consider no activity
        # elif mean activity within the interval falls above the second scalar, consider activity
        thresholds = [(0.0, 0.1), 0.4]

        return bu_input, td_input, windows, thresholds

    def plot_input_pattern(self):

        input_fig, input_axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
        input_axs[0].plot(self.bu_input, c='k')
        input_axs[1].plot(self.td_input, c='k', ls='--')

        input_axs[0].set_title('Bottom-up input')
        input_axs[1].set_title('Top-down input')

        for ax in input_axs:
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim([0.0, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_xticks([np.mean(ai).astype(int) for ai in self.windows])
            ax.set_xticklabels(['BU=TD', 'BU>TD', 'BU<TD'])
            ax.set_ylabel('Firing rate (a.u.)')

        input_fig.tight_layout()

        return input_fig


    def run(self):

        """
        :return: combo_dict

        key: tuple
            (
                index for synaptic connection pattern,
                index for input pattern generating pPE,
                index for input pattern generating nPE
            )

        value: array (8, T/dt)
            the first 4 rows are firing rates of 4 neuron types within microcircuit generating pPE
            the next 4 rows are firing rates of 4 neuron types within microcircuit generating nPE

        """

        # create a dictionary for synaptic-input pattern combinations
        combo_dict = {}

        # loop over synaptic combos (n=512)
        for syn_mat_i, syn_mat in enumerate(self.syn_patterns):
            # initialize firing rates: [PC, PV, SST, VIP]
            firing_rates = np.zeros((len(self.input_patterns), 4, self.steps + 1))
            # loop over all input combination patterns (n=64)
            for i, inp_i in enumerate(
                    tqdm.tqdm(
                        self.input_patterns,
                        position=0,
                        desc=f'Searching for combinations: connectivity pattern #{syn_mat_i + 1} / {len(self.syn_patterns)}',
                        leave=False
                    )
            ):
                for t in range(self.steps):
                    Isyn = (
                            inp_i @ np.array([self.bu_input[t], self.td_input[t]])
                            + syn_mat @ firing_rates[i, :, t]
                            + self.Ibg
                    )
                    firing_rates[i, :, t + 1] = (
                            firing_rates[i, :, t] + (self.dt / self.tau) * drdt(r=firing_rates[i, :, t], Isyn=Isyn)
                    )

            # find input patterns that satisfy pPE or nPE response patterns
            npe_comb, ppe_comb = self.search_idcs(frs=firing_rates)

            # if such pattern exists, add to the combination dictionary
            if (len(npe_comb) >= 1) and (len(ppe_comb) >= 1):

                # combo_dict[mat_i] = list(itertools.product(npe_comb, ppe_comb))
                # frs.append(firing_rates[:, :, 1:])
                pe_idcs = list(itertools.product(npe_comb, ppe_comb))
                for iter_idx, (ppe_inp_idx, npe_inp_idx) in enumerate(pe_idcs):
                    combo_dict[syn_mat_i, ppe_inp_idx, npe_inp_idx] = np.vstack(
                        [firing_rates[ppe_inp_idx, :, 1:], firing_rates[npe_inp_idx, :, 1:]]
                    )

        return combo_dict

    def search_idcs(self, frs):

        # time window for three conditions:
        # take only pyramidal cell FR
        match_window = frs[:, 0, range(*self.windows[0])].mean(axis=1)
        mismatch_window = frs[:, 0, range(*self.windows[1])].mean(axis=1)
        playback_window = frs[:, 0, range(*self.windows[2])].mean(axis=1)

        # thresholds
        (nr_min, nr_max), r_thres = self.thresholds

        # indices for three conditions - nPE
        nPE_match = np.argwhere((match_window >= nr_min) & (match_window < nr_max)).flatten()
        nPE_mismatch = np.argwhere(mismatch_window > r_thres).flatten()
        nPE_playback = np.argwhere((playback_window >= nr_min) & (playback_window < nr_max)).flatten()
        nPE_idcs = [nPE_match, nPE_mismatch, nPE_playback]

        # indices for three conditions - pPE
        pPE_match = np.argwhere((match_window >= nr_min) & (match_window < nr_max)).flatten()
        pPE_mismatch = np.argwhere((mismatch_window >= nr_min) & (mismatch_window < nr_max)).flatten()
        pPE_playback = np.argwhere(playback_window > r_thres).flatten()
        pPE_idcs = [pPE_match, pPE_mismatch, pPE_playback]

        # list all synaptic combinations that show nPE
        syn_combs = [
            list(set.intersection(*map(set, nPE_idcs))),
            list(set.intersection(*map(set, pPE_idcs)))
        ]

        return syn_combs

def plot_mean_firing_rates(windows, frs):

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

def get_conn_combo(constraint, change_val=1):

    conn_change_idcs = np.argwhere(constraint == 0)
    conn_combination_list = np.array(list(itertools.product([0, change_val], repeat=len(conn_change_idcs))))

    test_idcs = sorted(np.random.choice(len(conn_combination_list), 64, False))
    conn_combination_list = conn_combination_list[test_idcs]

    conn_mat = np.stack([constraint] * len(conn_combination_list))
    for i, conn_combo in enumerate(conn_combination_list):
        for j, (x_idx, y_idx) in enumerate(conn_change_idcs):
            conn_mat[i][x_idx, y_idx] = conn_combo[j]

    return conn_mat

def drdt(r, Isyn):
    """
    :param r: firing rate
    :param Isyn: synatpic input
    :return gradient for the activity update
    """
    return -r + ReLu(Isyn)


if __name__ == '__main__':

    searcher = run_combinatorial_search()
    input_pattern_fig = searcher.plot_input_pattern()
    input_pattern_fig.show()

    combi_dict = searcher.run()

    combinations = list(combi_dict.keys())
    ppe_stats, npe_stats, fig = plot_mean_firing_rates(
        windows=searcher.windows, frs=combi_dict[combinations[0]][[0, 4]]
    )
    fig.show()