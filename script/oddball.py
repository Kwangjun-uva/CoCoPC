import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

from script.pc_network import network
from script.data_loader import generate_input
from script.tools import remove_top_right_spines, create_vlines


class oddball_simulator(object):

    def __init__(self, len_seq, pos_dev, sim_params, pretrained_weights, dataset, dev_stim_type='testSet_differentClass'):
        """
        :param len_seq: int. length of oddball sequence
        :param pos_dev: int. position of deviant stimulus
        :param sim_params: dict. simulation parameters
        :param pretrained_weights: dict. pretrained weights
        :param dataset: dict. dataset used during training
        :param dev_stim_type: str. deviant stimulus info. select from
                ['trainingSet_sameClass', 'trainingSet_differentClass', 'testSet_sameClass',
                'testSet_differentClass', 'fmnist', 'mnist']
        """
        # number of images in the sequence (length of sequence)
        self.num_images = len_seq
        # position of deviant in the sequence
        self.pos_deviant = pos_dev
        # simulation parameters
        self.simParams = sim_params
        # trained weights for the PC network
        self.weights = pretrained_weights
        # dataset from which images are selected
        self.dataset = dataset

        # simulation parameters
        self.steps_isi = int(self.simParams['isi_time'] / self.simParams['dt'])
        self.steps_sim = int(self.simParams['sim_time'] / self.simParams['dt'])

        # pick reference and deviant images
        self.devStim_type = dev_stim_type
        self.ref_img, self.odd_img = self.pick_ref_and_dev_images()

        # initialize network and sequence
        self.net = network(simParams=self.simParams, pretrained_weights=self.weights)
        self.sequence = None

    def generate_oddball_inputs(self, img_s, img_d):
        """
        :param img_s: array (h, w). reference image
        :param img_d: array (h, w). deviant image
        :return: oddball sequence (self.num_images, h, w) and odball sequence figure

        Given reference and deviant images, it generates an oddball sequence
        with length=self.num_images and dev_pos=self.pos_deviant
        """
        # create a sequence with a length (n_seq) and a deviant stimulation location (dev_loc)
        oddball_seq = np.repeat(img_s[None, ...], self.num_images, axis=0)
        oddball_seq[self.pos_deviant - 1] = img_d

        # show the sequence
        xy_dim = np.sqrt(oddball_seq[0].shape[0]).astype(int)
        oddball_seq_fig, oddball_seq_axs = plt.subplots(
            nrows=1, ncols=self.num_images, sharex='all', sharey='all', figsize=(self.num_images * 3, 3)
        )
        for i, ax in enumerate(oddball_seq_axs.flat):
            ax.imshow(oddball_seq[i].reshape(xy_dim, xy_dim), cmap='gray')
            ax.axis('off')

            if i == self.pos_deviant - 1:
                ax.set_title('deviant')
            else:
                ax.set_title('standard')

        oddball_seq_fig.suptitle('Oddball stimulus sequence')
        oddball_seq_fig.tight_layout()  # rect=[0, 0.03, 1, 0.95])

        return oddball_seq, oddball_seq_fig

    @staticmethod
    def pick_idx(idcs_list):
        return np.random.choice(a=idcs_list, size=1, replace=False)[0]


    def pick_ref_and_dev_images(self):

        dev_types = ['trainingSet_sameClass', 'trainingSet_differentClass', 'testSet_sameClass',
                     'testSet_differentClass', 'fmnist', 'mnist']

        # set reference image and class
        ref_class = np.random.choice(a=self.dataset['test_y'], size=1, replace=False)[0]
        ref_img_idcs = np.where(self.dataset['test_y'] == ref_class)[0]
        ref_img_idx = np.random.choice(a=ref_img_idcs, size=1, replace=False)[0]
        ref_img = self.dataset['test_x'][ref_img_idx]
        # ref_img = self.dataset['test_x'][0]
        # ref_class = self.dataset['test_y'][0]

        # randomly select an image of the same class from the training set (seen during the training)
        if self.devStim_type == 'trainingSet_sameClass':
            idx_same_class = np.argwhere(self.dataset['train_y'] == ref_class).flatten()
            idx_random = np.random.choice(a=idx_same_class)
            odd_img = self.dataset['train_x'][idx_random]

        # randomly select an image of a different class from the training set (seen during the training)
        elif self.devStim_type == 'trainingSet_differentClass':
            idx_diff_class = np.argwhere(self.dataset['train_y'] != ref_class).flatten()
            idx_random = self.pick_idx(idcs_list=idx_diff_class)
            odd_img = self.dataset['train_x'][idx_random]

        # randomly select an image of the same class from the test set (not seen during the training)
        elif self.devStim_type == 'testSet_sameClass':
            idx_same_class = np.argwhere(self.dataset['test_y'] == ref_class).flatten()
            idx_random = self.pick_idx(idcs_list=idx_same_class)
            odd_img = self.dataset['test_x'][idx_random]

        # randomly select an image of a different class from the test set (not seen during the training)
        elif self.devStim_type == 'testSet_differentClass':
            idx_diff_class = np.argwhere(self.dataset['test_y'] != ref_class).flatten()
            idx_random = self.pick_idx(idcs_list=idx_diff_class)
            odd_img = self.dataset['test_x'][idx_random]

        # randomly select an image from MNIST or FMNIST dataset (not seen during the training + different statistics)
        elif self.devStim_type in ['mnist', 'fmnist']:
            # edit sim_params to FMNIST
            self.simParams['dataset'] = 'fmnist'
            # edit sim_params to FMNIST
            self.simParams['n_class'] = 10
            # edit sim_params to FMNIST
            self.simParams['n_sample'] = 2
            # fmnist input
            fmnist_x, fmnist_y, _, _, _ = generate_input(
                dataset_type=self.simParams['dataset'],
                num_class=self.simParams['n_class'],
                num_sample=self.simParams['n_sample'],
                max_fr=self.simParams['max_fr'],
                class_choice=None, shuffle=True
            )
            idx_random = 1
            train_img_dim = np.sqrt(ref_img.shape[0]).astype(int)
            odd_img_dim = np.sqrt(fmnist_x[idx_random].shape[0]).astype(int)
            odd_img = np.zeros((train_img_dim, train_img_dim))
            odd_img[2:30, 2:30] = fmnist_x[idx_random].reshape(odd_img_dim, odd_img_dim)
            odd_img = odd_img.flatten()
        else:
            raise ValueError(f'Deviant type must be one of: {dev_types}')

        return ref_img, odd_img

    def inject_propofol(self, str):
        if isinstance(str, float) and 0 <= str <= 1:
            self.net.propofol_inh = 1.0 - str
        else:
            raise ValueError(f'Propofol strength must be a float between 0 and 1')

    def simulate(self, pfp_str=0.0, omit=False, record='all', show_sequence=False):

        # if omit, replace deviant image with zeros
        dev_img = np.zeros_like(self.odd_img) if omit else self.odd_img
        # generate sequence
        self.sequence, seq_fig = self.generate_oddball_inputs(img_s=self.ref_img, img_d=dev_img)
        # show sequence
        if show_sequence:
            seq_fig.show()

        # propofol
        if pfp_str > 0.0:
            self.inject_propofol(str=pfp_str)

        # initialize network
        self.net.initialize_network(batch_size=1)
        self.net.initialize_activity_log()

        for i in trange(self.num_images):
            # isi
            for t_step in range(self.steps_isi):
                self.net.compute(inputs=np.zeros(self.sequence[i].reshape(-1, 1).shape), record=record)
            # stimulus presentation
            for t_step in range(self.steps_sim):
                self.net.compute(inputs=self.sequence[i].reshape(-1, 1), record=record)

    def plot_pop_responses(self, target='pyramidal'):
        """
        Plot population responses for pyramidal cells or interneurons.

        Args:
            target: str, either 'pyramidal' or 'interneurons'
        """
        if target == 'pyramidal':
            keys = [
                ('layer_1', 'rep_r'),
                ('layer_0', 'ppe_pyr'),
                ('layer_0', 'npe_pyr')
            ]
            colors = ['#6950A3', '#CA181D', '#2070B4']
            labels = ['Rep', 'PE+', 'PE-']
            subplot_shape = (3, 1)
            figsize = (12, 8)

        elif target == 'interneurons':
            colors = {
                'ppe_pv': '#FF2F92', 'npe_pv': '#00B0F0',
                'ppe_sst': '#F90000', 'npe_sst': '#0432FF',
                'ppe_vip': '#FF6666', 'npe_vip': '#76ADEE'
            }
            labels = ['PV+', 'SST+', 'VIP+', 'PV-', 'SST-', 'VIP-']
            keys = [
                ('layer_0', int_type) for int_type in self.net.activity_log['layer_0']
                if ('pyr' not in int_type) and (int_type != 'rep_r')
            ]
            subplot_shape = (2, 3)
            figsize = (12, 4)

        else:
            raise ValueError(f"Unknown target: {target}. Must be 'pyramidal' or 'interneurons'.")

        # === shared plotting code ===
        nrows, ncols = subplot_shape
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', figsize=figsize)
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        total_t = self.num_images * (self.steps_isi + self.steps_sim)
        lines = []

        for i, ax in enumerate(axs[:len(keys)]):
            key1, key2 = keys[i]
            color = colors[i] if isinstance(colors, list) else colors[key2]
            line, = ax.plot(self.net.activity_log[key1][key2], c=color)
            lines.append(line)

            # axis labels: automatic placement
            if i // ncols == nrows - 1:  # bottom row
                ax.set_xlabel('Time (ms)')
            if i % ncols == 0:  # first column
                ax.set_ylabel('Firing rate (a.u.)')

            remove_top_right_spines(target_axes=ax)
            create_vlines(
                target_axes=ax, total_sim_time=total_t,
                trial_sim_time=self.steps_sim, interval_time=self.steps_isi
            )

        fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.85, 1))
        fig.show()

        return fig


    def vary_dev_loc_and_get_peaks(self, isi_time=0.005):
        """
        :param isi_time: inter-stimulus interval (sec)

        Simulate neural responses to an oddball sequence with varying placement of deviant stimulus in the sequence.
        The deviant stimulus cannot be placed first or last in the sequence.

        :return:
        max_amplitudes: array (3, number of deviant stimulus indices).
            peak amplitude of neural activity to deviant stimulus in different places in the sequence.
        amp_fig: figure
            a plot showing peak amplitude of responses to deviant stimulus in different place in the sequence.
        dev_fig: figure
            a plot showing neural responses to deviant stimulus in different places in the sequence.
        """

        # define the number of deviant stimulus indices: n_img - 2 to exclude the first and the last slot
        num_dev_loc = self.num_images - 2

        # define stimulus presentation and ISI durations (in simulation steps)
        self.simParams['isi_time'] = isi_time
        sim_steps = int(self.simParams['sim_time'] / self.simParams['dt'])
        isi_steps = int(self.simParams['isi_time'] / self.simParams['dt'])

        # # create a figure: responses to deviant stimulus vs time
        # dev_response_fig, dr_axs = plt.subplots(nrows=3, ncols=1, sharex='all')

        # array to save max amplitudes
        max_amplitudes = np.zeros((3, num_dev_loc))

        # loop over deviant indices
        for d_idx in range(num_dev_loc):

            # define the deviant stimulus position
            self.pos_deviant = d_idx + 2
            # simulate neural responses
            self.simulate(record='pyr')
            # define start and end indices of deviant stimulus presentation time window
            deviant_time_slice = slice(
                (d_idx + 1) * sim_steps + (d_idx + 2) * isi_steps,
                (d_idx + 2) * (sim_steps + isi_steps)
            )
            # select neuron groups to extract activities from
            activities = [
                self.net.activity_log['layer_1']['rep_r'][deviant_time_slice],
                self.net.activity_log['layer_0']['ppe_pyr'][deviant_time_slice],
                self.net.activity_log['layer_0']['npe_pyr'][deviant_time_slice]
            ]

            # get max (peak) amplitude across varying locations of deviant stimulus within the oddball sequence
            # loop over the selected neuron groups
            for ng_i, activity in enumerate(activities):
                # save max (peak) amplitude
                max_amplitudes[ng_i, d_idx] = np.max(activity)

        # plot
        # set colors for rep (purple), PE+ (red), and PE- (blue)
        colors = [
            plt.colormaps.get_cmap(cmap_i)(np.linspace(start=0, stop=1, num=num_dev_loc * 2)[::-1])
            for cmap_i in ['Purples', 'Reds', 'Blues']
        ]

        # set titles for subplots
        amp_titles = ['rep', 'PE+', 'PE-']

        # create a figure: max (peak) amplitude of response to deviant stimulus vs deviant stimulus position
        plt.close('all')
        amp_fig, amp_axs = plt.subplots(nrows=3, ncols=1, sharex='all')

        for amp_ax_idx, amp_ax in enumerate(amp_axs.flat):
            # plot max (peak) amplitude of responses to deviant stimulus
            amp_ax.plot(max_amplitudes[amp_ax_idx], c=colors[amp_ax_idx][0], lw=2)
            # remove top and right spines of subplot
            remove_top_right_spines(target_axes=amp_axs[amp_ax_idx])
            # set subplot title
            amp_ax.set_title(amp_titles[amp_ax_idx])
            # define x-axis ticks (deviant stimulus location index)
            amp_ax.set_xticks(np.arange(num_dev_loc), np.arange(1, num_dev_loc + 1))
            # add axis labels
            if amp_ax_idx == 1:
                amp_ax.set_ylabel('Peak amplitutde')
            elif amp_ax_idx == 2:
                amp_ax.set_xlabel('Deviant location')
        # adjust figure margin
        amp_fig.tight_layout()
        amp_fig.show()

        return max_amplitudes, amp_fig

    def compute_mmn(self, signal_type=None, sim_time=2000, isi_time=5):

        if signal_type is None:
            signal = self.net.activity_log['layer_0']['ppe_pyr']
        else:
            signal = self.net.activity_log[signal_type[0]][signal_type[1]]

        ss = np.zeros(sim_time)
        for seq_i in range(self.num_images):
            if seq_i != self.pos_deviant - 1:
                ss += np.array(signal[seq_i * sim_time + (seq_i + 1) * isi_time: (seq_i + 1) * (sim_time + isi_time)])
        ss /= self.num_images - 1

        deviant_start = (self.pos_deviant - 1) * sim_time + self.pos_deviant * isi_time
        deviant_end = self.pos_deviant * sim_time + self.pos_deviant * isi_time
        sd = np.array(signal[deviant_start: deviant_end])

        mmn_fig, mmn_axs = plt.subplots(1, 1)
        mmn_axs.plot(ss, label='Standard')
        mmn_axs.plot(sd, label='Deviant')
        mmn_axs.plot(sd - ss, label='Difference')
        remove_top_right_spines(target_axes=mmn_axs)
        # mmn_axs.spines['top'].set_visible(False)
        # mmn_axs.spines['right'].set_visible(False)
        mmn_axs.set_xlabel('Time (ms)')
        mmn_axs.set_ylabel('Firing rate (a.u.)')

        mmn_fig.tight_layout()

        return mmn_fig
