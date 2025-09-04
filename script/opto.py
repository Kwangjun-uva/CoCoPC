import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from script.pc_network import network
from script.tools import sample_imgs, generate_reconstruction_pad


class opto_simulator(object):

    def __init__(self, sim_params, pretrained_weights, dataset, n_class=10, n_sample=3):

        # image dataset
        self.dataset = dataset

        # sample images from the dataset
        self.images, self.labels = self._get_random_samples(
            n_sample_class=n_class, n_sample_per_class=n_sample
        )
        # define the number of samples across all samples and the number of pixels of images
        self.n_total_sample, self.n_pxl = self.images.shape

        # define simulation time steps for stimulus presentation and inter-stimulus interval (ISI)
        self.t_sim = int(sim_params['sim_time'] / sim_params['dt'])
        self.t_isi = int(sim_params['isi_time'] / sim_params['dt'])

        # initialize network
        self.opto_network = network(simParams=sim_params, pretrained_weights=pretrained_weights)

    def _get_random_samples(self, n_sample_class, n_sample_per_class):

        # randomly sample image indices from dataset
        sample_idcs = sample_imgs(
            self.dataset['test_y'],
            n_class=n_sample_class,
            n_sample=n_sample_per_class,
            class_choice=None
        )
        # shuffle indices
        np.random.shuffle(sample_idcs)
        # get images and labels from the random selection
        images = self.dataset['test_x'][sample_idcs]
        labels = self.dataset['test_y'][sample_idcs]

        return images, labels

    def simulate_optoSielncing(self, silence_target, record_var='error'):

        # initialize network
        self.opto_network.initialize_network(batch_size=self.n_total_sample)
        self.opto_network.initialize_activity_log()
        self.opto_network.reset_silencer()

        # set target neuron groups for silencing
        if silence_target is None:
            pass
        else:
            for (target_layer, target_neuron) in silence_target:
                print(target_layer, target_neuron)
                self.opto_network.activate_silencer(target_layer=target_layer, target_neuron=target_neuron)

        # simulate isi
        for _ in trange(self.t_isi):
            self.opto_network.compute(inputs=np.zeros(self.images.T.shape), record=record_var)
        # simulate stimulus presentation
        for _ in trange(self.t_sim):
            self.opto_network.compute(inputs=self.images.T, record=record_var)

    def run_experiment(self, silencing_targets=['none', 'pyr', 'pv', 'sst', 'vip'], recon_fig_nrow=5):

        # create figure dictionaries for reconstructions and responses
        recon_figs = {}
        response_figs = {}

        # loop over target neuron types for optogenetic silencing experiment
        for neuron_type in silencing_targets:

            print(f'silencing {neuron_type}... ')
            # silence the target neuron type in both PE+ and PE- microcircuits
            if neuron_type == 'none':
                ppe_target = None
                npe_target = None
            else:
                ppe_target = f'ppe_{neuron_type}'
                npe_target = f'npe_{neuron_type}'

            # simulate
            self.simulate_optoSielncing(
                silence_target=[('layer_0', ppe_target), ('layer_0', npe_target)],
                record_var='error'
            )

            # response
            response_figs[neuron_type] = self._plot_responses()
            # reconstruction
            recon_figs[neuron_type] = self._get_predictions(nrow=recon_fig_nrow)

        return recon_figs, response_figs

    def _plot_responses(self):

        font = {'family': 'Arial',
                'color': 'darkred',
                'weight': 'normal',
                'size': 20,
                'color': 'black'
                }

        plt.close('all')
        response_fig, response_axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
        # plot from stimulus onset (self.t_isi:)
        response_axs.plot(self.opto_network.activity_log['layer_0']['ppe_pyr'][self.t_isi:], lw=2.0, c='r')
        response_axs.plot(self.opto_network.activity_log['layer_0']['npe_pyr'][self.t_isi:], lw=2.0, c='b')
        # place 100 ms bar
        response_axs.plot(np.arange(0, 101), -0.1 * np.ones(101), linewidth=5, c='black')
        response_axs.text(-60, -0.1, '100 ms', fontdict=font)
        # remove spines
        for spine_loc in ['top', 'right', 'bottom', 'left']:
            response_axs.spines[spine_loc].set_visible(False)
        # remove x- and y-ticks
        response_axs.set_yticks([])
        response_axs.set_xticks([])
        # adjust figure margin
        response_fig.tight_layout()
        response_fig.show()

        return response_fig

    def _get_predictions(self, nrow=5):

        plt.close('all')
        # get reconstructions
        recon_none = self.opto_network.weights['01'].T @ self.opto_network.network['layer_1']['rep_r']
        # show in grid
        _, _, pred_pad = generate_reconstruction_pad(img_mat=recon_none, nx=nrow)
        # plot
        recon_fig, recon_axs = plt.subplots(nrows=1, ncols=1)
        pred_im = recon_axs.imshow(pred_pad, cmap='gray', vmin=0, vmax=1)
        # colorbar to show firing rate
        recon_fig.colorbar(pred_im, ax=recon_axs, shrink=0.5)
        # remove axis
        recon_axs.axis('off')
        # adjust figure margin
        recon_fig.tight_layout()
        # fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/fig6_recon_{target_neuron_type}.png', dpi=300, bbox_inches='tight')
        recon_fig.show()

        return recon_fig
