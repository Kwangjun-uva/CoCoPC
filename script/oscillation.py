import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

from scipy.signal import correlate, correlation_lags, find_peaks
from script.pc_network import network
from script.tools import remove_top_right_spines


class oscillation_simulator(object):

    def __init__(self, sim_params, weights, dataset, n_sample=16):

        # pick random samples from the datset
        ran_idcs = np.random.choice(a=len(dataset['test_x']), size=n_sample, replace=False)
        self.dataset = dataset['test_x'][ran_idcs]
        self.n_images = len(self.dataset)

        # create network
        self.network = network(simParams=sim_params, pretrained_weights=weights)
        # initialize network
        self._initialize_network()

        # set sim steps
        self.dt = sim_params['dt']
        self.steps_isi = int(sim_params['isi_time'] / self.dt)
        self.steps_sim = int(sim_params['sim_time'] / self.dt)

        # initialize activity log (dictionary)
        self._initialize_activity_dict()
        # get dictionary keys for pyramidal and interneuron keys (for easier access)
        self.pyr_keys, self.interneuron_keys = self._get_neuron_keys()

        # for reset
        self._tau = sim_params['tau_exc']

    def _initialize_network(self):
        """
        initialize the network
        """

        self.network.initialize_network(batch_size=self.n_images)
        self.network.initialize_activity_log()

    def _initialize_activity_dict(self):
        """
        initializes all neural activities in the network to baseline (default = 0)
        """

        # get all neuron groups in layer 0
        pe_circuit_neuron_groups = list(self.network.network['layer_0'].keys())[3:]

        # add them to the output dictionary
        self.activity_dict = {
            ('layer_0', neuron_group): np.zeros((*self.network.network['layer_0'][neuron_group].shape, self.steps_sim))
            for neuron_group in pe_circuit_neuron_groups
        }
        # add layer 1, rep_r neuron to the dictionary
        self.activity_dict.update(
            {('layer_1', 'rep_r'): np.zeros((*self.network.network['layer_1']['rep_r'].shape, self.steps_sim))})

    def _reset_tau(self):
        self.network.tau_exc = self._tau
        self.network.tau_inh = self._tau

    def simulate(self, simT=10):
        """
        simulates neural activities in response to randomly selected images
        """

        # define simulation duration
        self.steps_sim = int(simT / self.dt)

        # initialize network
        self._initialize_network()
        # initialize activity log
        self._initialize_activity_dict()

        # simulate ISI (for baseline activity)
        for _ in trange(self.steps_isi):
            self.network.compute(inputs=np.zeros(self.dataset.T.shape), record=None)

        # simulate
        for t_step in trange(self.steps_sim):
            self.network.compute(inputs=self.dataset.T, record=None)
            # save activities
            for (pc_layer, neuron_group), activity in self.activity_dict.items():
                activity[:, :, t_step] += self.network.network[pc_layer][neuron_group]

    def plot_pyr_activity_across_images(self, take_mean=True, t_range=None):

        """
        :param take_mean: take mean of pyramidal neuron population activities (default=True).
                            if set to False, all individual neuron activities are plotted.
        :param t_range: magnify into the range (time steps in tuple)
        :return:
            a figure that shows pyramidal neuron population (or individual) activities
            averaged across randomly selected images.
                subplot1: L5 pyramidal neurons in Rep microcircuit
                subplot2: L2/3 pyramidal neurons in PE+ microcircuit
                subplot3: L2/3 pyramidal neurons in PE- microcircuit
        """
        # neuron_types = [('layer_1', 'rep_r'), ('layer_0', 'ppe_pyr'), ('layer_0', 'npe_pyr')]
        # color_maps = ['Purples', 'Reds', 'Blues']
        # plot_labels = ['Representation L5 pyramidal activity', 'pPE pyramidal activity', 'nPE pyramidal activity']
        # plot_colors = [plt.get_cmap(cmap_i)(np.linspace(start=0, stop=1, num=nSample)) for cmap_i in color_maps]

        _, plot_labels, plot_colors = self._get_pyramidal_neuron_activity()
        if t_range is None:
            t_range = (0, -1)
        else:
            t_range = t_range

        plt.close('all')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        for i in range(len(plot_labels)):
            # Get the data for the current curve
            if take_mean:
                y_data = self.activity_dict[self.pyr_keys[i]].mean(axis=0).mean(axis=0)[slice(*t_range)]
                ax.plot(y_data, linewidth=2, c=plot_colors[i][-1], label=plot_labels[i])
                # Find peaks
                peaks, _ = find_peaks(y_data)

                # Plot vertical lines at each peak
                for peak_x_coord in peaks:
                    ax.axvline(x=peak_x_coord, c=plot_colors[i][-1], linestyle='--')
            else:
                for sample_i in t_range(self.n_images):
                    ax.plot(
                        self.activity_dict[self.pyr_keys[i]][:, sample_i, :].mean(axis=0)[slice(*t_range)],
                                linewidth=2, c=plot_colors[i][sample_i]
                            )

        ax.set_title('Phase shifted oscillations')
        ax.set_ylabel('Firing rates (a.u.)')
        ax.set_xlabel('Time (ms)')
        remove_top_right_spines(ax)
        ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')

        fig.tight_layout()
        fig.show()

        return fig

    def _get_pyramidal_neuron_activity(self, cut=-1):
        """
        :param cut:
        :return:
            xs: list of array. L2/3 PE+ and PE- and L5 Rep pyramidal neuron firing rates.
            xs_labels: list of str. For plotting purpose.
            xs_colors: list of colors. For plotting purpose.
            pyr_keys: list of str. For indexing purpose.
        """

        # get mean firing rates of each pyramidal neuron populations
        xs = [self.activity_dict[pyr_key].mean(axis=1).mean(axis=0)[:cut] for pyr_key in self.pyr_keys]
        # set labels for each pyramidal neuron populations
        xs_labels = ['Representation L5 pyramidal activity', r'PE$+$ L2/3 pyramidal activity', r'PE$-$ L2/3 pyramidal activity']
        # set colors for each pyramidal neuron populations
        cmaps = ['Purples', 'Reds', 'Blues']
        # generate shades of color, each of which represents the firing rate of pyramidal neurons
        # in response to different images (e.g., different shades of red in the middle subplot show
        # firing rates of L2/3 pyramidal cells PE+ microcircuits in response to different images
        xs_colors = [plt.colormaps.get_cmap(cmap_i)(np.linspace(start=0, stop=1, num=self.n_images)) for cmap_i in cmaps]

        return xs, xs_labels, xs_colors

    def _get_neuron_keys(self):
        """
        :return: pyramidal neuron keys, interneuron keys

        Get pyramidal and interneuron keys in the activity log dictionary.
        ng stands for neuron group (e.g., ppe_pyr, ppe_pv, npe_sst, etc.)
        ng is a tuple of layer number and neuron type (e.g., 'layer_0', 'pyr')
        [2,0,1] is to order pyramidal cells
            as L5 pyr in Rep microcircuit, L2/3 pyr in PE+ microcircuit, and L2/3 pyr in PE- microcircuit
        """

        # get activity_dict keys for interneurons (pv, sst, and vip)
        interneuron_keys = [ng for ng in list(self.activity_dict.keys()) if ('0' in ng[0]) and ('pyr' not in ng[1])]
        # get activity_dict keys for pyramidal cells
        pyr_keys = [[ng for ng in list(self.activity_dict.keys()) if (ng not in interneuron_keys)][i] for i in [2, 0, 1]]

        return pyr_keys, interneuron_keys

    def plot_cutoffPlot_pyr(self, first_range=(0, 1000), second_range=(1500, 2000)):
        """
        :param first_range: tuple. the first range to plot (in ms)
        :param second_range: tuple. the second range to plot (in ms)
        :return:
            (Fig 4A) a figure that shows pyramidal neuron activities in the first and second ranges
            defined as arguments, with cutoff in the middle. Note that, unlike plot_pyr_activity_across_images,
            this function plots neural activities in response to individual images as different shades of the same color
        """
        # cut off plot for Fig1A
        # color_maps = ['Purples', 'Reds', 'Blues']
        # plot_labels = ['Representation L5 pyramidal activity', 'pPE pyramidal activity', 'nPE pyramidal activity']
        # plot_colors = [plt.get_cmap(cmap_i)(np.linspace(start=0, stop=1, num=nSample)) for cmap_i in color_maps]

        _, plot_labels, plot_colors = self._get_pyramidal_neuron_activity()

        plt.close('all')
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 4), gridspec_kw={'width_ratios': [2, 1]})
        for ax_i, ax in enumerate(axs.flat):
            # i // 2 % 3
            for i in range(self.n_images):
                if ax_i < 2:
                    ax.plot(self.activity_dict[('layer_1', 'rep_r')][:, i, :].mean(axis=0), linewidth=2, c=plot_colors[0][i])
                elif (ax_i >= 2) and (ax_i < 4):
                    ax.plot(self.activity_dict[('layer_0', 'ppe_pyr')][:, i, :].mean(axis=0), linewidth=2,
                            c=plot_colors[1][i])
                else:
                    ax.plot(self.activity_dict[('layer_0', 'npe_pyr')][:, i, :].mean(axis=0), linewidth=2,
                            c=plot_colors[2][i])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if (ax_i % 2) == 0:
                ax.set_xlim(*first_range)
            else:
                ax.set_xlim(*second_range)
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks([])

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)

        return fig

    def plot_interneuron_activity_across_images(self, take_mean=True):

        """
        :param take_mean: take mean of pyramidal neuron population activities (default=True).
                            if set to False, all individual neuron activities are plotted.
        :return:
            a figure that shows pyramidal neuron population (or individual) activities
            averaged across randomly selected images.
                subplot1: L2/3 PV cells in PE+ microcircuit
                subplot2: L2/3 SST cells in PE+ microcircuit
                subplot3: L2/3 VIP cells in PE+ microcircuit
                subplot4: L2/3 PV cells in PE- microcircuit
                subplot5: L2/3 SST cells in PE- microcircuit
                subplot6: L2/3 VIP cells in PE- microcircuit
        """

        # define base colors for PV, SST, and VIP
        cmap_variables = ['Blues', 'Greens', 'Wistia']

        # plot
        plt.close('all')
        fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        ax_counter = 0
        for (pc_layer, neuron_group), activity in self.activity_dict.items():
            if (pc_layer, neuron_group) in self.interneuron_keys:
                line_colors = plt.get_cmap(cmap_variables[ax_counter % 3])(
                    np.linspace(start=0, stop=1, num=activity.shape[1]))
                if take_mean:
                    axs.flat[ax_counter].plot(activity.mean(axis=1).mean(axis=0), c=line_colors[-1])
                else:
                    for sample_i in range(self.n_images):
                        axs.flat[ax_counter].plot(activity[:, sample_i, :].mean(axis=0), c=line_colors[sample_i])
                axs.flat[ax_counter].set_title(neuron_group[0] + " ".join(neuron_group.split('_'))[1:].upper())
                axs.flat[ax_counter].spines['right'].set_visible(False)
                axs.flat[ax_counter].spines['top'].set_visible(False)
                ax_counter += 1
        fig.tight_layout()

        return fig

    def plot_cutoffPlot_interneurons(self, first_range=(0, 1000), second_range=(1500, 2000)):

        # suppFig: cut-off plot for interneurons
        interneuron_keys = [ng[1] for ng in list(self.activity_dict.keys()) if ('0' in ng[0]) and ('pyr' not in ng[1])]
        cmap_dict = {
            'ppe_pv': '#FF2F92', 'npe_pv': '#00B0F0',
            'ppe_sst': '#F90000', 'npe_sst': '#0432FF',
            'ppe_vip': '#FF6666', 'npe_vip': '#76ADEE'
        }
        line_alpha = np.linspace(start=0.2, stop=0.7, num=self.n_images)
        # cmap_variables = ['Blues', 'Greens', 'Wistia']

        plt.close('all')
        fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1, 2, 1, 2, 1]})

        plot_counter = 0
        for (layer, ng), activity in self.activity_dict.items():
            if ng in interneuron_keys:
                # c_idx = int(plot_counter / 2 % 3)
                # cmap_i = plt.colormaps.get(cmap_dict[c_idx])(np.linspace(start=0, stop=1, num=self.n_images))
                base_color = cmap_dict[ng]
                for sample_i in range(self.n_images):
                    # axs.flat[plot_counter].plot(activity[:, sample_i, :].mean(axis=0), c=cmap_i[sample_i])
                    # axs.flat[plot_counter + 1].plot(activity[:, sample_i, :].mean(axis=0), c=cmap_i[sample_i])
                    axs.flat[plot_counter].plot(
                        activity[:, sample_i, :].mean(axis=0), c=base_color, alpha=line_alpha[sample_i]
                    )
                    axs.flat[plot_counter + 1].plot(
                        activity[:, sample_i, :].mean(axis=0), c=base_color, alpha=line_alpha[sample_i]
                    )
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

    def plot_phase_autocorr(self):
        """
        (Fig 4C). plots autocorrelation in the first row and phase in the second row.
                  Col 1: L5 pyr in Rep microcircuit (Purple)
                  Col 2: L2/3 pyr in PE+ microcircuit (Red)
                  Col 3: L2/3 pyr in PE- microcircuit (Blue)
        """
        sampling_frequency = int(1 / self.dt)

        # get pyramidal cell activities
        pyr_data, _, _ = self._get_pyramidal_neuron_activity()
        # define colors
        pyr_colors = ['#6950A3', '#CA181D', '#2070B4']

        # phase plots in the first row and autocorrleation plots in the second row
        phase_fig, phase_axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
        for phase_ax_i, phase_ax in enumerate(phase_axs.flat):

            # set an index to loop pyramidal cell data (len=3) once for each row
            data_idx = phase_ax_i % 3  # 012012
            # phase plot in the second row
            if phase_ax_i >= 3:
                phase_axs[1, phase_ax_i - 3].plot(
                    pyr_data[data_idx], compute_phase_velocity(x=pyr_data[data_idx]),
                    c=pyr_colors[data_idx], lw=2
                )
            # autocorrelation plot in the first row
            else:
                lags, autocorrelation, freq = compute_autocorrelation_and_estimate_frequency(
                    pyr_data[data_idx][:2000], sampling_frequency
                )
                phase_ax.plot(lags, autocorrelation, c=pyr_colors[data_idx], label=f'$f$ = {freq:.2f} Hz')
            # set tick label size
            phase_ax.tick_params(axis='both', which='major', labelsize=15)
            # set spine linewidth
            for axis in ['top', 'bottom', 'left', 'right']:
                phase_ax.spines[axis].set_linewidth(1.5)
        # adjust figure margin
        phase_fig.subplots_adjust(wspace=0.4, hspace=0.6)
        phase_fig.show()

        return phase_fig

    def plot_freq_across_taus(self, simT, tau_multiples):

        freq_per_tau = np.zeros((len(tau_multiples), 3))

        for i, tau_i in enumerate(tau_multiples):

            # reset tau to the value used during training
            self._reset_tau()
            # set tau to a different value
            self.network.tau_exc *= tau_i
            self.network.tau_inh *= tau_i

            # initialize network
            self._initialize_network()
            # initialize activity log
            self._initialize_activity_dict()
            # simulate and save activity of all neurons at every time step
            self.simulate(simT)

            # ge pyramidal neuron activities
            pyr_activity, _, _ = self._get_pyramidal_neuron_activity()

            # get frequency
            for pyr_idx, pyr_activity in enumerate(pyr_activity):
                _, _, freq_per_tau[i, pyr_idx] = compute_autocorrelation_and_estimate_frequency(
                    signal=pyr_activity[:2000], sampling_frequency=int(1 / self.dt)
            )
            print(f'tau = {tau_i} Done!')

        # plot
        plt.close('all')
        # rescale x-axis (ms)
        x = [self._tau * tau_j * 1000 for tau_j in tau_multiples]
        _, pyr_labels, pyr_colors = self._get_pyramidal_neuron_activity()
        # fig, axs = plt.subplots(nrows=1, ncols=1)
        #
        # # 0 = rep, 1 = PE+, 2=PE-
        # axs.scatter(x, freq_per_tau[:, 0], c='r', marker='o')
        # axs.plot(x, freq_per_tau[:, 0], c='k')
        # axs.spines['top'].set_visible(False)
        # axs.spines['right'].set_visible(False)
        # axs.set_xlabel(r'$\tau$ (ms)', fontsize=15)
        # axs.set_ylabel(r'$f$ (Hz)', fontsize=15)

        plt.close('all')
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex='all', sharey='all', figsize=(8, 8))

        # 0 = rep, 1 = PE+, 2=PE-
        for i, ax in enumerate(axs.flat):
            # plot
            ax.scatter(x, freq_per_tau[:, i], c=pyr_colors[i][-1], marker='o')
            ax.plot(x, freq_per_tau[:, i], c=pyr_colors[i][-5])

            # label
            ax.set_title(pyr_labels[i], color=pyr_colors[i][-2])

            # remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # add axis labels
            if i == 1:
                ax.set_ylabel(r'$f$ (Hz)', fontsize=15)
            elif i == 2:
                ax.set_xlabel(r'$\tau$ (ms)', fontsize=15)

            # set axis linewidth
            ax.tick_params(axis='both', which='major', labelsize=15)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.5)

        fig.tight_layout()
        fig.show()

        return freq_per_tau, fig

def compute_phase_velocity(x, dt=1):
    # Compute velocity as the numerical derivative of position (x)
    return np.gradient(x, dt)


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
