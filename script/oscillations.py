import copy

import numpy as np
from scipy.signal import correlate, correlation_lags, find_peaks
from script.pc_network import network
import matplotlib.pyplot as plt
from tqdm import trange


def get_neuron_keys(activity_dict):
    interneuron_keys = [ng for ng in list(activity_dict.keys()) if ('0' in ng[0]) and ('pyr' not in ng[1])]
    pyr_keys = [[ng for ng in list(activity_dict.keys()) if (ng not in interneuron_keys)][i] for i in [2, 0, 1]]

    return pyr_keys, interneuron_keys


def get_pyr_data(activity_dict, nColors=16, cut=-1):
    pyr_keys, _ = get_neuron_keys(activity_dict)

    xs = [activity_dict[pyr_key].mean(axis=1).mean(axis=0)[:cut] for pyr_key in pyr_keys]
    xs_labels = ['Representation L5 pyramidal activity', r'PE$+$ pyramidal activity', r'PE$-$ pyramidal activity']
    cmaps = ['Purples', 'Reds', 'Blues']
    xs_colors = [plt.colormaps.get_cmap(cmap_i)(np.linspace(start=0, stop=1, num=nColors)) for cmap_i in cmaps]

    return xs, xs_labels, xs_colors, pyr_keys


def plot_phasePortraits(activity_dict, cut=-1):
    pyr_data, pyr_labels, pyr_colors, pyr_keys = get_pyr_data(activity_dict=activity_dict, cut=cut)
    velocity = {}

    plt.close('all')
    phase_fig, phase_axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    for phase_ax_i, phase_ax in enumerate(phase_axs):
        pyr_velocity = compute_phase_velocity(x=pyr_data[phase_ax_i])
        velocity[pyr_keys[phase_ax_i][1]] = pyr_velocity
        phase_ax.plot(pyr_data[phase_ax_i], pyr_velocity, c=pyr_colors[phase_ax_i][-1], lw=2)
        phase_ax.tick_params(axis='both', which='major', labelsize=15)
        phase_ax.set_title(pyr_labels[phase_ax_i], c=pyr_colors[phase_ax_i][-1])
        if phase_ax_i == 1:
            phase_ax.set_ylabel('Velocity (v)', fontsize=15)
        elif phase_ax_i == 2:
            phase_ax.set_xlabel('Position (x)', fontsize=15)
        for axis in ['top', 'bottom', 'left', 'right']:
            phase_ax.spines[axis].set_linewidth(1.5)
    phase_fig.tight_layout()

    return phase_fig, velocity


def compute_phase_velocity(x, dt=1):
    # Compute velocity as the numerical derivative of position (x)
    return np.gradient(x, dt)


def plot_autocorrelation(activity_dict, sim_params, cut=-1):
    sampling_frequency = int(1 / sim_params['dt'])  # Hz

    pyr_data, pyr_labels, pyr_colors, pyr_keys = get_pyr_data(activity_dict=activity_dict, cut=cut)
    freqs = np.zeros(len(pyr_data))

    ac_fig, ac_axis = plt.subplots(3, 1, figsize=(5, 8), sharex=True, sharey=True)
    for ac_ax_idx, ac_ax in enumerate(ac_axis):
        lags, autocorrelation, freq = compute_autocorrelation_and_estimate_frequency(pyr_data[ac_ax_idx],
                                                                                     sampling_frequency)
        freqs[ac_ax_idx] = freq
        # Plot autocorrelation
        ac_ax.plot(lags, autocorrelation, c=pyr_colors[ac_ax_idx][-1], label=f'$f$ = {freq:.2f} Hz')
        ac_ax.legend(loc='upper right', labelcolor=pyr_colors[ac_ax_idx][-1])
        if ac_ax_idx == 1:
            ac_ax.set_ylabel("Autocorrelation", fontsize=15)
        elif ac_ax_idx == 2:
            ac_ax.set_xlabel("Lags (ms)", fontsize=15)

        ac_ax.set_title(pyr_labels[ac_ax_idx], fontsize=15, color=pyr_colors[ac_ax_idx][-1])

        ac_ax.tick_params(axis='both', which='major', labelsize=15)
        for axis in ['top', 'bottom', 'left', 'right']:
            ac_ax.spines[axis].set_linewidth(1.5)
        ac_fig.tight_layout()

    return ac_fig, freqs


def compute_autocorrelation_and_estimate_frequency(signal, sampling_frequency):  # , signal_title, plot_color='b'):
    """
    This function plots the autocorrelation of a signal, identifies peaks (including lag 1), and estimates the fundamental frequency.

    Args:
      signal: A NumPy array containing the signal.
      sampling_frequency: The sampling frequency of the signal in Hz.

    Returns:
      dominant_frequency: The estimated dominant frequency of the signal in Hz (or None if not found).
    """

    # Remove the mean to avoid bias in autocorrelation
    signal_demeaned = signal - np.mean(signal)

    # Calculate autocorrelation with proper normalization
    autocorrelation = correlate(signal_demeaned, signal_demeaned, mode='full') / np.sum(signal_demeaned ** 2)

    # Calculate lags using scipy.signal.correlation_lags
    lags = correlation_lags(len(signal), len(signal))

    # find peaks
    peaks, _ = find_peaks(autocorrelation)
    # identify lag 1
    lag1_idx = peaks[len(peaks) // 2:][1]
    lag1_val = lags[lag1_idx]
    # compute the dominant frequency
    freq = 1 / lag1_val * sampling_frequency

    return lags, autocorrelation, freq


def get_activity_log(ntwork, steps_isi, steps_sim, dataset_select):
    # get all neuron groups in layer 0
    pe_circuit_neuron_groups = list(ntwork.network['layer_0'].keys())[3:]

    # add them to the output dictionary
    activity_dict = {
        ('layer_0', neuron_group): np.zeros((*ntwork.network['layer_0'][neuron_group].shape, steps_sim))
        for neuron_group in pe_circuit_neuron_groups
    }
    # add layer 1, rep_r neuron to the dictionary
    activity_dict.update({('layer_1', 'rep_r'): np.zeros((*ntwork.network['layer_1']['rep_r'].shape, steps_sim))})

    # simulate ISI
    for _ in trange(steps_isi):
        ntwork.compute(inputs=np.zeros(dataset_select.T.shape), record=None)

    # simulate
    for t_step in trange(steps_sim):
        ntwork.compute(inputs=dataset_select.T, record=None)
        # save activities
        for (pc_layer, neuron_group), activity in activity_dict.items():
            activity[:, :, t_step] += ntwork.network[pc_layer][neuron_group]

    return activity_dict


def plot_pyr_activity(activity_dict, take_mean=True):
    # neuron_types = [('layer_1', 'rep_r'), ('layer_0', 'ppe_pyr'), ('layer_0', 'npe_pyr')]
    # color_maps = ['Purples', 'Reds', 'Blues']
    # plot_labels = ['Representation L5 pyramidal activity', 'pPE pyramidal activity', 'nPE pyramidal activity']
    # plot_colors = [plt.get_cmap(cmap_i)(np.linspace(start=0, stop=1, num=nSample)) for cmap_i in color_maps]

    plot_data, plot_labels, plot_colors, pyr_keys = get_pyr_data(activity_dict=activity_dict)
    nSample = activity_dict[pyr_keys[0]].shape[1]

    plt.close('all')
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    for ax_i, ax in enumerate(axs.flat):
        if take_mean:
            ax.plot(
                activity_dict[pyr_keys[ax_i]].mean(axis=0).mean(axis=0),
                linewidth=2, c=plot_colors[ax_i][-1]
            )
        else:
            for sample_i in range(nSample):
                ax.plot(
                    activity_dict[pyr_keys[ax_i]][:, sample_i, :].mean(axis=0),
                    linewidth=2, c=plot_colors[ax_i][sample_i]
                )

        ax.set_title(plot_labels[ax_i], c=plot_colors[ax_i][-1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axs[1].set_ylabel('Firing rates (a.u.)')
    axs[2].set_xlabel('Time (ms)')

    fig.tight_layout()

    return fig


def plot_interneuron_activity(activity_dict, take_mean=True):
    nSample = activity_dict[('layer_0', 'ppe_pyr')].shape[1]

    _, interneuron_keys = get_neuron_keys(activity_dict=activity_dict)
    cmap_variables = ['Blues', 'Greens', 'Wistia']

    plt.close('all')
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    ax_counter = 0
    for (pc_layer, neuron_group), activity in activity_dict.items():
        if (pc_layer, neuron_group) in interneuron_keys:
            line_colors = plt.get_cmap(cmap_variables[ax_counter % 3])(
                np.linspace(start=0, stop=1, num=activity.shape[1]))
            if take_mean:
                axs.flat[ax_counter].plot(activity.mean(axis=1).mean(axis=0), c=line_colors[-1])
            else:
                for sample_i in range(nSample):
                    axs.flat[ax_counter].plot(activity[:, sample_i, :].mean(axis=0), c=line_colors[sample_i])
            axs.flat[ax_counter].set_title(neuron_group[0] + " ".join(neuron_group.split('_'))[1:].upper())
            axs.flat[ax_counter].spines['right'].set_visible(False)
            axs.flat[ax_counter].spines['top'].set_visible(False)
            ax_counter += 1
    fig.tight_layout()

    return fig


def plot_cutoff_pyr(activity_dict, first_range=(0, 1000), second_range=(1500, 2000)):
    # cut off plot for Fig1A
    # color_maps = ['Purples', 'Reds', 'Blues']
    # plot_labels = ['Representation L5 pyramidal activity', 'pPE pyramidal activity', 'nPE pyramidal activity']
    # plot_colors = [plt.get_cmap(cmap_i)(np.linspace(start=0, stop=1, num=nSample)) for cmap_i in color_maps]

    plot_data, plot_labels, plot_colors, pyr_keys = get_pyr_data(activity_dict=activity_dict)
    nSample = activity_dict[pyr_keys[0]].shape[1]

    plt.close('all')
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 4), gridspec_kw={'width_ratios': [2, 1]})
    for ax_i, ax in enumerate(axs.flat):
        # i // 2 % 3
        for i in range(nSample):
            if ax_i < 2:
                ax.plot(activity_dict[('layer_1', 'rep_r')][:, i, :].mean(axis=0), linewidth=2, c=plot_colors[0][i])
            elif (ax_i >= 2) and (ax_i < 4):
                ax.plot(activity_dict[('layer_0', 'ppe_pyr')][:, i, :].mean(axis=0), linewidth=2, c=plot_colors[1][i])
            else:
                ax.plot(activity_dict[('layer_0', 'npe_pyr')][:, i, :].mean(axis=0), linewidth=2, c=plot_colors[2][i])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if (ax_i % 2) == 0:
            ax.set_xlim(*first_range)
        else:
            ax.set_xlim(*second_range)
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])

    # axs[1].set_ylabel('firing rates (a.u.)')
    # axs[2].set_xlabel('time (ms)')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)

    return fig


def plot_cutoff_interneurons(activity_dict, first_range=(0, 1000), second_range=(1500, 2000)):
    # suppFig: cut-off plot for interneurons
    interneuron_keys = [ng[1] for ng in list(activity_dict.keys()) if ('0' in ng[0]) and ('pyr' not in ng[1])]
    cmap_variables = ['Blues', 'Greens', 'Wistia']
    nSample = activity_dict[('layer_0', 'ppe_pyr')].shape[1]

    plt.close('all')
    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1, 2, 1, 2, 1]})

    plot_counter = 0
    for (layer, ng), activity in activity_dict.items():
        if ng in interneuron_keys:
            c_idx = int(plot_counter / 2 % 3)
            cmap_i = plt.colormaps.get(cmap_variables[c_idx])(np.linspace(start=0, stop=1, num=nSample))
            for sample_i in range(nSample):
                axs.flat[plot_counter].plot(activity[:, sample_i, :].mean(axis=0), c=cmap_i[sample_i])
                axs.flat[plot_counter + 1].plot(activity[:, sample_i, :].mean(axis=0), c=cmap_i[sample_i])
            axs.flat[plot_counter].set_xlim(*first_range)
            axs.flat[plot_counter + 1].set_xlim(*second_range)
            axs.flat[plot_counter + 1].spines['left'].set_visible(False)
            axs.flat[plot_counter + 1].yaxis.set_ticks([])

            plot_counter += 2

    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()

    return fig


def plot_phase_autocorr(activity_dict, sim_params, cut=-1):
    sampling_frequency = int(1 / sim_params['dt'])

    plot_data, plot_labels, plot_colors, pyr_keys = get_pyr_data(activity_dict=activity_dict, cut=cut)
    # nSample = activity_dict[pyr_keys[0]].shape[1]

    # xs = [
    #     activity_dict[('layer_1', 'rep_r')].mean(axis=1).mean(axis=0),
    #     activity_dict[('layer_0', 'ppe_pyr')].mean(axis=1).mean(axis=0),
    #     activity_dict[('layer_0', 'npe_pyr')].mean(axis=1).mean(axis=0)
    # ]
    # xs_colors = [plt.cm.Purples(0.75), plt.cm.Reds(0.75), plt.cm.Blues(0.75)]
    plot_colors = ['#6950A3', '#CA181D', '#2070B4']
    # phase autocorr together: horizontal
    phase_fig, phase_axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for phase_ax_i, phase_ax in enumerate(phase_axs.flat):
        # p_ax = phase_fig.add_subplot(3, 1, phase_ax_i + 1)

        data_idx = phase_ax_i % 3  # 012012
        if phase_ax_i >= 3:
            phase_axs[1, phase_ax_i - 3].plot(plot_data[data_idx], compute_phase_velocity(x=plot_data[data_idx]),
                                              c=plot_colors[data_idx],
                                              lw=2)
            # phase_ax.yaxis.tick_right()
        else:
            lags, autocorrelation, freq = compute_autocorrelation_and_estimate_frequency(plot_data[data_idx][:2000],
                                                                                         sampling_frequency)
            phase_ax.plot(lags, autocorrelation, c=plot_colors[data_idx], label=f'$f$ = {freq:.2f} Hz')
            # phase_ax.legend(loc='upper right', labelcolor=cs[data_idx], fontsize=15)

        phase_ax.tick_params(axis='both', which='major', labelsize=15)
        # if phase_ax_i == 2:
        #     phase_ax.set_ylabel('Velocity (v)')
        # elif phase_ax_i == 2:
        #     phase_ax.set_xlabel('Position (x)')
        for axis in ['top', 'bottom', 'left', 'right']:
            phase_ax.spines[axis].set_linewidth(1.5)
    phase_fig.subplots_adjust(wspace=0.4, hspace=0.6)

    return phase_fig


def get_taus(tau_multiples, sim_params, weights, data_select, steps_isi, steps_sim):

    freq_per_tau = np.zeros((len(tau_multiples), 3))

    for i, tau_i in enumerate(tau_multiples):
        # deepcopy simParams
        sim_parameters = copy.deepcopy(sim_params)
        sim_parameters['tau_exc'] *= tau_i
        sim_parameters['tau_inh'] *= tau_i

        # initialize network
        test_net = network(simParams=sim_parameters, pretrained_weights=weights)
        test_net.initialize_network(batch_size=data_select.shape[0])
        test_net.initialize_error()

        # simulate and save activity of all neurons at every time step
        activity_dict = get_activity_log(
            ntwork=test_net,
            steps_isi=steps_isi, steps_sim=steps_sim,
            dataset_select=data_select
        )

        # get frequency
        _, freq_per_tau[i] = plot_autocorrelation(activity_dict=activity_dict, sim_params=sim_parameters)

    plt.close('all')
    x = [sim_params['tau_exc'] * tau for tau in tau_multiples]
    fig, axs = plt.subplots(1, 1)

    # 0 = rep, 1 = PE+, 2=PE-
    axs.scatter(x, freq_per_tau[:, 0], c='r', marker='o')
    axs.plot(x, freq_per_tau[:, 0], c='k')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_xlabel(r'$\tau$ (ms)', fontsize=15)
    axs.set_ylabel(r'$f$ (Hz)', fontsize=15)
    axs.tick_params(axis='both', which='major', labelsize=15)
    for axis in ['top', 'bottom', 'left', 'right']:
        axs.spines[axis].set_linewidth(1.5)
    fig.show()

    return freq_per_tau, fig
