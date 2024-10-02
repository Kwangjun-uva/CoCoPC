import numpy as np
import matplotlib.pyplot as plt
from script.pc_network import test_oddball, generate_input, remove_top_right_spines


class oddball_expt(object):

    def __init__(self, len_seq, loc_dev, simParams, pretrained_weights, dataset):

        self.num_images = len_seq
        self.loc_deviant = loc_dev
        self.simParams = simParams
        self.weights = pretrained_weights
        self.dataset = dataset

        self.didx = None
        self.net = None
        self.sequence = None

    def generate_oddball_inputs(self, img_s, img_d):

        # create a sequence with a length (n_seq) and a deviant stimulation location (dev_loc)
        oddball_seq = np.repeat(img_s[None, ...], self.num_images, axis=0)
        oddball_seq[self.loc_deviant - 1] = img_d

        # show the sequence
        xy_dim = np.sqrt(oddball_seq[0].shape[0]).astype(int)
        oddball_seq_fig, oddball_seq_axs = plt.subplots(
            nrows=1, ncols=self.num_images, sharex='all', sharey='all', figsize=(self.num_images * 3, 3)
        )
        for i, ax in enumerate(oddball_seq_axs.flat):
            ax.imshow(oddball_seq[i].reshape(xy_dim, xy_dim), cmap='gray')
            ax.axis('off')

            if i == self.loc_deviant - 1:
                ax.set_title('deviant')
            else:
                ax.set_title('standard')

        oddball_seq_fig.suptitle('Oddball stimulus sequence')
        oddball_seq_fig.tight_layout()  # rect=[0, 0.03, 1, 0.95])

        return oddball_seq, oddball_seq_fig

    def get_dev_idx(self, idcs_list):

        if self.didx is None:
            return np.random.choice(idcs_list, 1, replace=False)[0]
        else:
            return self.didx

    def simulate(self, sim_type, record='all'):

        ref_img = self.dataset['test_x'][0]
        ref_class = self.dataset['test_y'][0]

        if sim_type == 'training_same':
            idx_same_class = np.argwhere(self.dataset['train_y'] == ref_class).flatten()
            idx_random = self.get_dev_idx(idcs_list=idx_same_class)
            odd_img = self.dataset['train_x'][idx_random]

        elif sim_type == 'training_different':
            idx_diff_class = np.argwhere(self.dataset['train_y'] != ref_class).flatten()
            idx_random = self.get_dev_idx(idcs_list=idx_diff_class)
            odd_img = self.dataset['train_x'][idx_random]

        elif sim_type == 'test_same':
            idx_same_class = np.argwhere(self.dataset['test_y'] == ref_class).flatten()
            idx_random = self.get_dev_idx(idcs_list=idx_same_class)
            odd_img = self.dataset['test_x'][idx_random]

        elif sim_type == 'test_different':
            idx_diff_class = np.argwhere(self.dataset['test_y'] != ref_class).flatten()
            idx_random = self.get_dev_idx(idcs_list=idx_diff_class)
            odd_img = self.dataset['test_x'][idx_random]

        elif sim_type in ['mnist', 'fmnist']:
            # edit sim_params to FMNIST
            self.simParams['dataset'] = 'fmnist'
            # edit sim_params to FMNIST
            self.simParams['n_class'] = 10
            # edit sim_params to FMNIST
            self.simParams['n_sample'] = 2
            # fmnist input
            fmnist_x, fmnist_y, _, _, _ = generate_input(
                simParams=self.simParams, class_choice=None, shuffle=True
            )
            idx_random = 1
            train_img_dim = np.sqrt(ref_img.shape[0]).astype(int)
            odd_img_dim = np.sqrt(fmnist_x[idx_random].shape[0]).astype(int)
            odd_img = np.zeros((train_img_dim, train_img_dim))
            odd_img[2:30, 2:30] = fmnist_x[idx_random].reshape(odd_img_dim, odd_img_dim)
            odd_img = odd_img.flatten()

        # oddball figure
        oddball_x, odd_fig = self.generate_oddball_inputs(img_s=ref_img, img_d=odd_img)

        # oddball simulation
        oddball_net, re_fig = test_oddball(
            sim_params=self.simParams, weights=self.weights, oddball_x=oddball_x,
            record=record
        )

        self.didx = idx_random
        self.net = oddball_net
        self.sequence = oddball_x

        return odd_fig, re_fig

    def vary_devLoc_and_get_amplitudes(self):

        num_dev_loc = self.num_images - 2

        self.simParams['isi_time'] = 0.005
        sim_time = int(self.simParams['sim_time'] / self.simParams['dt'])
        isi_time = int(self.simParams['isi_time'] / self.simParams['dt'])

        dev_response_fig, dr_axs = plt.subplots(nrows=3, ncols=1, sharex='all')
        colors = [
            plt.colormaps.get_cmap(cmap_i)(np.linspace(start=0, stop=1, num=(num_dev_loc) * 2)[::-1])
            for cmap_i in ['Purples', 'Reds', 'Blues']
        ]

        amp_fig, amp_axs = plt.subplots(3, 1, sharex='all')
        max_amplitudes = np.zeros((3, num_dev_loc))

        for d_idx in range(num_dev_loc):

            self.loc_deviant = d_idx + 1
            plt.close('all')
            _, _ = self.simulate(
                sim_type='test_different',
                record='all'
            )

            deviant_start_idx = (d_idx + 1) * sim_time + (d_idx + 2) * isi_time
            deviant_end_idx = (d_idx + 2) * (sim_time + isi_time)

            neuron_groups = [
                self.net.errors['layer_1']['rep_r'],
                self.net.errors['layer_0']['ppe_pyr'],
                self.net.errors['layer_0']['npe_pyr']
            ]

            for ng_i, neuron_group in enumerate(neuron_groups):

                signal = neuron_group[slice(deviant_start_idx, deviant_end_idx)]
                dr_axs[ng_i].plot(signal, c=colors[ng_i][d_idx], lw=2)  # , label=f'{d_idx + 1}')
                remove_top_right_spines(target_axes=dr_axs[ng_i])
                # dr_axs[ng_i].spines['top'].set_visible(False)
                # dr_axs[ng_i].spines['right'].set_visible(False)
                if ng_i == 1:
                    dr_axs[ng_i].set_ylabel('Firing rate ( a.u.)')
                elif ng_i == 2:
                    dr_axs[ng_i].set_xlabel('Time (ms)')

                max_amplitudes[ng_i, d_idx] = np.max(signal)

        dev_response_fig.tight_layout()

        amp_titles = ['rep', 'PE+', 'PE-']
        for ng_i, _ in enumerate(neuron_groups):
            amp_axs[ng_i].plot(max_amplitudes[ng_i], c=colors[ng_i][0], lw=2)
            remove_top_right_spines(target_axes=amp_axs[ng_i])
            # amp_axs[ng_i].spines['top'].set_visible(False)
            # amp_axs[ng_i].spines['right'].set_visible(False)
            amp_axs[ng_i].set_title(amp_titles[ng_i])
            amp_axs[ng_i].set_xticks(np.arange(num_dev_loc), np.arange(1, num_dev_loc + 1))
            if ng_i == 1:
                amp_axs[ng_i].set_ylabel('Firing rate (a.u.)')
            elif ng_i == 2:
                amp_axs[ng_i].set_xlabel('Deviant location')

        amp_fig.tight_layout()

        return max_amplitudes, amp_fig, dev_response_fig

    def compute_mmn(self, signal_type=None, sim_time=2000, isi_time=5):

        if signal_type is None:
            signal = self.net.errors['layer_0']['ppe_pyr']
        else:
            signal = self.net.errors[signal_type[0]][signal_type[1]]

        ss = np.zeros(sim_time)
        for seq_i in range(self.num_images):
            if seq_i != self.loc_deviant - 1:
                ss += np.array(signal[seq_i * sim_time + (seq_i + 1) * isi_time: (seq_i + 1) * (sim_time + isi_time)])
        ss /= self.num_images - 1

        deviant_start = (self.loc_deviant - 1) * sim_time + self.loc_deviant * isi_time
        deviant_end = self.loc_deviant * sim_time + self.loc_deviant * isi_time
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

# def oddball_input_generator(standard_img, deviant_img, n_seq, dev_loc):
#     # create a sequence with a length (n_seq) and a deviant stimulation location (dev_loc)
#     oddball_seq = np.repeat(standard_img[None, ...], n_seq, axis=0)
#     oddball_seq[dev_loc - 1] = deviant_img
#
#     # show the sequence
#     xy_dim = np.sqrt(oddball_seq[0].shape[0]).astype(int)
#     oddball_seq_fig, oddball_seq_axs = plt.subplots(nrows=1, ncols=n_seq, sharex='all', sharey='all', figsize=(n_seq * 3, 3))
#     for i, ax in enumerate(oddball_seq_axs.flat):
#         ax.imshow(oddball_seq[i].reshape(xy_dim, xy_dim), cmap='gray')
#         ax.axis('off')
#
#         if i == dev_loc - 1:
#             ax.set_title('deviant')
#         else:
#             ax.set_title('standard')
#
#     oddball_seq_fig.suptitle('Oddball stimulus sequence')
#     oddball_seq_fig.tight_layout()#rect=[0, 0.03, 1, 0.95])
#
#     return oddball_seq, oddball_seq_fig


# def get_dev_idx(deviant_idx, idcs_list):
#     if deviant_idx is None:
#         return np.random.choice(idcs_list, 1, replace=False)[0]
#     else:
#         return deviant_idx
#
#
# def oddball_simulation(sim_type, sim_params, weights, dataset, dev_idx=None, len_seq=5, loc_deviant=3, record='all'):
#
#     ref_img = dataset['test_x'][0]
#     ref_class = dataset['test_y'][0]
#
#     if sim_type == 'training_same':
#         idx_same_class = np.argwhere(dataset['train_y'] == ref_class).flatten()
#         idx_random = get_dev_idx(deviant_idx=dev_idx, idcs_list=idx_same_class)
#         odd_img = dataset['train_x'][idx_random]
#     elif sim_type == 'training_different':
#         idx_diff_class = np.argwhere(dataset['train_y'] != ref_class).flatten()
#         idx_random = get_dev_idx(deviant_idx=dev_idx, idcs_list=idx_diff_class)
#         odd_img = dataset['train_x'][idx_random]
#     elif sim_type == 'test_same':
#         idx_same_class = np.argwhere(dataset['test_y'] == ref_class).flatten()
#         idx_random = get_dev_idx(deviant_idx=dev_idx, idcs_list=idx_same_class)
#         odd_img = dataset['test_x'][idx_random]
#     elif sim_type == 'test_different':
#         idx_diff_class = np.argwhere(dataset['test_y'] != ref_class).flatten()
#         idx_random = get_dev_idx(deviant_idx=dev_idx, idcs_list=idx_diff_class)
#         odd_img = dataset['test_x'][idx_random]
#     elif sim_type == 'fmnist':
#         # edit sim_params to FMNIST
#         sim_params['dataset'] = 'fmnist'
#         # edit sim_params to FMNIST
#         sim_params['n_class'] = 10
#         # edit sim_params to FMNIST
#         sim_params['n_sample'] = 3
#         # fmnist input
#         fmnist_x, fmnist_y, _, _, _ = generate_input(
#             simParams=sim_params, class_choice=None, shuffle=True
#         )
#         train_img_dim = np.sqrt(ref_img.shape[0]).astype(int)
#         odd_img_dim = np.sqrt(fmnist_x[1].shape[0]).astype(int)
#         odd_img = np.zeros((train_img_dim, train_img_dim))
#         odd_img[2:30, 2:30] = fmnist_x[1].reshape(odd_img_dim, odd_img_dim)
#         odd_img = odd_img.flatten()
#     elif sim_type == 'novel_set':
#         # edit sim_params to FMNIST
#         sim_params['dataset'] = 'fmnist'
#         # edit sim_params to FMNIST
#         sim_params['n_class'] = 10
#         # edit sim_params to FMNIST
#         fmnist_x, fmnist_y, _, _, _ = generate_input(
#             simParams=sim_params, class_choice=None, shuffle=True
#         )
#         odd_img = fmnist_x[1]
#         idx_random = np.random.choice(np.arange(len(fmnist_y)), 1, replace=False)[0]
#         ref_img = fmnist_x[idx_random]
#
#     # oddball figure
#     oddball_x, odd_fig = oddball_input_generator(
#         standard_img=ref_img, deviant_img=odd_img, n_seq=len_seq, dev_loc=loc_deviant
#     )
#
#     # oddball simulation
#     oddball_net, re_fig = test_oddball(
#         sim_params=sim_params, weights=weights, oddball_x=oddball_x,
#         record=record
#     )
#
#     return oddball_x, oddball_net, odd_fig, re_fig
#
#
# def compute_mmn(signal, n_seq, d_loc, sim_time=2000, isi_time=5):
#     ss = np.zeros(sim_time)
#     for seq_i in range(n_seq):
#         if seq_i != d_loc - 1:
#             ss += np.array(signal[seq_i * sim_time + (seq_i + 1) * isi_time: (seq_i + 1) * (sim_time + isi_time)])
#     ss /= n_seq - 1
#
#     sd = np.array(signal[(d_loc - 1) * sim_time + d_loc * isi_time: d_loc * sim_time + d_loc * isi_time])
#
#     mmn_fig, mmn_axs = plt.subplots(1, 1)
#     mmn_axs.plot(ss, label='Standard')
#     mmn_axs.plot(sd, label='Deviant')
#     mmn_axs.plot(sd - ss, label='Difference')
#     remove_top_right_spines(target_axes=mmn_axs)
#     # mmn_axs.spines['top'].set_visible(False)
#     # mmn_axs.spines['right'].set_visible(False)
#     mmn_axs.set_xlabel('Time (ms)')
#     mmn_axs.set_ylabel('Firing rate (a.u.)')
#
#     mmn_fig.tight_layout()
#
#     return mmn_fig
#
#
# def vary_devLoc_and_get_amplitudes(n_seq, dataset, sim_params, weights, dimg_idx=1135):
#
#     sim_params['isi_time'] = 0.005
#     sim_time = int(sim_params['sim_time'] / sim_params['dt'])
#     isi_time = int(sim_params['isi_time'] / sim_params['dt'])
#
#     dev_response_fig, dr_axs = plt.subplots(3, 1, sharex='all')
#     colors = [
#         plt.colormaps.get_cmap('Purples')(np.linspace(0, 1, (n_seq - 2) * 2)[::-1]),
#         plt.colormaps.get_cmap('Reds')(np.linspace(0, 1, (n_seq - 2) * 2)[::-1]),
#         plt.colormaps.get_cmap('Blues')(np.linspace(0, 1, (n_seq - 2) * 2)[::-1])
#     ]
#
#     amp_fig, amp_axs = plt.subplots(3, 1, sharex='all')
#     max_amplitudes = np.zeros((3, n_seq - 2))
#
#     for d_idx in range(n_seq - 2):
#
#         plt.close('all')
#         oddball_inputs, oddball_net, oddball_seq, oddball_response = oddball_simulation(
#             sim_type='test_different',
#             sim_params=sim_params,
#             weights=weights,
#             dataset=dataset,
#             dev_idx=dimg_idx,
#             len_seq=n_seq,
#             loc_deviant=d_idx + 1,
#             record='all'
#         )
#
#         deviant_start_idx = (d_idx + 1) * sim_time + (d_idx + 2) * isi_time
#         deviant_end_idx = (d_idx + 2) * (sim_time + isi_time)
#
#         neuron_groups = [
#             oddball_net.errors['layer_1']['rep_r'],
#             oddball_net.errors['layer_0']['ppe_pyr'],
#             oddball_net.errors['layer_0']['npe_pyr']
#         ]
#
#         for ng_i, neuron_group in enumerate(neuron_groups):
#
#             signal = neuron_group[slice(deviant_start_idx, deviant_end_idx)]
#             dr_axs[ng_i].plot(signal, c=colors[ng_i][d_idx], lw=2)  # , label=f'{d_idx + 1}')
#             remove_top_right_spines(target_axes=dr_axs[ng_i])
#             # dr_axs[ng_i].spines['top'].set_visible(False)
#             # dr_axs[ng_i].spines['right'].set_visible(False)
#             if ng_i == 1:
#                 dr_axs[ng_i].set_ylabel('Firing rate ( a.u.)')
#             elif ng_i == 2:
#                 dr_axs[ng_i].set_xlabel('Time (ms)')
#
#             max_amplitudes[ng_i, d_idx] = np.max(signal)
#
#     dev_response_fig.tight_layout()
#
#     amp_titles = ['rep', 'PE+', 'PE-']
#     for ng_i, _ in enumerate(neuron_groups):
#         amp_axs[ng_i].plot(max_amplitudes[ng_i], c=colors[ng_i][0], lw=2)
#         remove_top_right_spines(target_axes=amp_axs[ng_i])
#         # amp_axs[ng_i].spines['top'].set_visible(False)
#         # amp_axs[ng_i].spines['right'].set_visible(False)
#         amp_axs.set_title(amp_titles[ng_i])
#         amp_axs[ng_i].set_xticks(np.arange(5), np.arange(1, 6))
#         if ng_i == 1:
#             amp_axs[ng_i].set_ylabel('Firing rate (a.u.)')
#         elif ng_i == 2:
#             amp_axs[ng_i].set_xlabel('Deviant location')
#
#     amp_fig.tight_layout()
#
#     return max_amplitudes, amp_fig, dev_response_fig
