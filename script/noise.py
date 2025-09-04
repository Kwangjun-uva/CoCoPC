import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

from tqdm import trange

from script.pc_network import network
from script.tools import sample_imgs, generate_reconstruction_pad, noise_generator


class noise_simulator(object):

    jit_type = 'constant'
    jit_lvl = 0.0

    def __init__(self, sim_params, weights, dataset, sample_class=10, sample_per_class=10):

        # load training results
        self.weights = weights

        # sample images for simulation
        self.imgs = self._sample_dataset(dataset=dataset, nClass=sample_class, nSample=sample_per_class)
        # number of sampled images
        self.n_imgs = len(self.imgs)
        # x and y dimensions for image
        self.img_xy_dim = tuple(np.repeat(np.sqrt(self.imgs.shape[-1]).astype(int), 2))

        # initialize network
        self.net = network(simParams=sim_params, pretrained_weights=self.weights)

        # sim parameters
        self.steps_isi = int(sim_params['isi_time'] / sim_params['dt'])
        self.steps_sim = int(sim_params['sim_time'] / sim_params['dt'])

        # noise parameters
        # noise source: {'ext', 'int'}
        self.noise_source = None
        # noise distribution: {'normal', 'uniform', 'constant'}
        self.noise_dist = None
        # noise value: [0, 1]
        self.noise_val = 0.0
        # simulated noise levels
        self.noise_levels = None

        # external noise
        self.ext_noise = None
        # noisy images = sampled images + external noise
        self.noisy_imgs = None

        # dictionary to save SSIM and MSE values
        self.ssim_dict = None
        self.mse_dict = None
        # dictionary to save original, noisy, and reconstructed input sample
        self.img_dict = None

    def _sample_dataset(self, dataset, nClass=10, nSample=2):
        """
        :param nClass: int. The number of classes to sample from.
        :param nSample: int. The number of samples per class to sample.
        :return: array (nClass * nSample, img_dim). Sampled images.

        randomly samples images from dataset
        """

        # pick random samples: 2 samples per class (default: nClass = 10)
        sample_idcs = sample_imgs(img_labels=dataset['test_y'], n_class=nClass, n_sample=nSample)
        # shuffle
        np.random.shuffle(sample_idcs)

        # sample images with the shuffled indices
        sampled_imgs = dataset['test_x'][sample_idcs]

        return sampled_imgs

    def add_noise(self):

        # description for noise
        noise_desc = (
                r'$\eta_{%s}$' % self.noise_source
                + r' ~ $\mathcal{N}$' * (self.noise_dist == 'normal')
                + r' ~ $\matcal{U}$' * (self.noise_dist == 'uniform')
                + f' (0, {self.noise_val:.3f})' * (self.noise_dist in ['normal', 'uniform'])
                + f' = {self.noise_val:.3f}' * (self.noise_dist == 'constant')
        )

        # external noise
        if self.noise_source == 'ext':
            # generate external noise
            ext_noise, _ = noise_generator(
                noise_type=self.noise_dist, noise_lvl=self.noise_val, target_shape=self.imgs.shape
            )

        # internal noise (jitter)
        elif self.noise_source == 'int':
            # set external noise to zero
            ext_noise = 0.0
            # define internal noise (jitter) distribution
            int_noise_dist = 'constant' * (self.noise_source == 'ext') + self.noise_dist * (self.noise_source == 'int')
            # define internal noise (jitter) level
            int_noise_val = self.noise_val * (self.noise_source == 'int')
            # add internal noise
            self.net.initialize_weights(pretrained_weights=self.weights, jitter_type=int_noise_dist, jitter_lvl=int_noise_val)

        else:
            raise ValueError('Noise source must be either "ext" or "int"')

        # add external noise to input
        self.noisy_imgs = np.clip(a=ext_noise * (self.noise_source == 'ext') + self.imgs, a_min=0.0, a_max=1.0)

        return noise_desc

    def _initialize_metric_dict(self, lvls, img_idx=0):

        # initialize ssim, mse variables
        self.ssim_dict = {key_i: np.zeros(len(lvls)) for key_i in ['clean_vs_noise', 'clean_vs_recon']}
        self.mse_dict = {key_i: np.zeros(len(lvls)) for key_i in ['clean_vs_noise', 'clean_vs_recon']}

        # initialize sample image dictionary for visualization of noisy input
        self.img_dict = {
            'original_img': self.imgs[img_idx].reshape(*self.img_xy_dim),
            'input_img': np.zeros((len(lvls), *self.img_xy_dim)),
            'recon_img': np.zeros((len(lvls), *self.img_xy_dim))
        }

    def run_experiment(self, noise_source: str, noise_dist: str, noise_levels: list, plot_recons=False):

        # initialize SSIM and MSE metrics
        self._initialize_metric_dict(lvls=noise_levels)

        #
        self.noise_levels = noise_levels
        # loop over noise level
        for noise_i, noise_val in enumerate(noise_levels):

            # set noise parameters
            self.noise_source = noise_source
            self.noise_dist = noise_dist
            self.noise_val = noise_val

            print(f'\nsimulating noise level of {noise_val}...')
            # training reconstruction
            plt.close('all')

            # add ext noise to sampled images or int noise to weights
            noise_desc = self.add_noise()

            # simulate
            self.simulate(input_x=self.noisy_imgs)

            # plot reconstructions
            noise_recon_fig = self._plot_reconstruction()
            # set title for the plot
            noise_recon_fig.suptitle(noise_desc)

            # set image shape
            img_shape = (self.n_imgs, *self.img_xy_dim)
            # original images
            clean_imgs = self.imgs.reshape(img_shape)
            # noisy images
            noisy_imgs = self.noisy_imgs.reshape(img_shape)
            # reconstructed images
            recon_imgs = self._get_predictions().reshape(img_shape)
            # save reconstruction
            self.img_dict['recon_img'][noise_i] = recon_imgs[0]
            self.img_dict['input_img'][noise_i] = noisy_imgs[0]

            # print(f'clean images shape: {clean_imgs.shape}, noisy images shape: {noisy_imgs.shape}, recon images shape: {recon_imgs.shape}')

            # calculate ssim and mse per sample image
            for i in trange(self.n_imgs, desc=f'{noise_i + 1}/{len(noise_levels)}', leave=False):
                # compare clean input vs noise input
                self.mse_dict['clean_vs_noise'][noise_i] += mse(clean_imgs[i], noisy_imgs[i])
                self.ssim_dict['clean_vs_noise'][noise_i] += ssim(clean_imgs[i], noisy_imgs[i],
                                                    data_range=noisy_imgs[i].max() - noisy_imgs[i].min())

                # compare clean input vs reconstruction from noisy input
                self.mse_dict['clean_vs_recon'][noise_i] += mse(clean_imgs[i], recon_imgs[i])
                self.ssim_dict['clean_vs_recon'][noise_i] += ssim(clean_imgs[i], recon_imgs[i],
                                                          data_range=recon_imgs[i].max() - recon_imgs[i].min())

        # take mean
        self.mse_dict = {k: v / self.n_imgs for k, v in self.mse_dict.items()}
        self.ssim_dict = {k: v / self.n_imgs for k, v in self.ssim_dict.items()}

        print('mse_dict: ', self.mse_dict)
        print('mse_dict: ', self.ssim_dict)

    def _get_input(self):
        return (self.net.network['layer_0']['rep_e']).T

    def _get_predictions(self):
        return (self.net.weights['01'].T @ self.net.network['layer_1']['rep_r']).T

    def _plot_reconstruction(self):
        """
        :return: a two-column figure with input on the left and reconstruction on the right.
        """

        # number of columns for reconstruction image pad
        _, _, input_pad = generate_reconstruction_pad(img_mat=self._get_input(), nx=len(self.imgs) // 10)
        _, _, pred_pad = generate_reconstruction_pad(img_mat=self._get_predictions(), nx=len(self.imgs) // 10)

        # plot
        recon_fig, recon_axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
        # noisy input
        input_imgs = recon_axs[0].imshow(input_pad, cmap='gray', vmin=0.0, vmax=1.0)
        recon_fig.colorbar(input_imgs, ax=recon_axs[0], shrink=0.4)
        recon_axs[0].set_title('input')
        # prediction
        pred_imgs = recon_axs[1].imshow(pred_pad, cmap='gray', vmin=0.0, vmax=1.0)
        recon_fig.colorbar(pred_imgs, ax=recon_axs[1], shrink=0.4)
        recon_axs[1].set_title('prediction')
        # remove axis
        for ax in recon_axs.flatten():
            ax.axis('off')
        # set title
        recon_fig.suptitle(f'Reconstructions at layer 0')
        # adjust margins
        recon_fig.tight_layout()

        return recon_fig

    def simulate(self, input_x, record='error'):

        # initialize network
        self.net.initialize_network(batch_size=len(self.imgs))
        # reset error
        self.net.initialize_activity_log()

        # isi
        for _ in trange(self.steps_isi):
            self.net.compute(inputs=np.zeros(input_x.T.shape), record=None)

        # stimulus presentation
        for _ in trange(self.steps_sim):
            self.net.compute(inputs=input_x.T, record=record)

    def plot_mse_ssim(self):

        plt.close('all')

        # pick color codes for MSE and SSIM
        mse_color = '#298c8c'  # plt.colormaps.get_cmap(colormap)(0.1)
        ssim_color = '#f1a226'  # plt.colormaps.get_cmap(colormap)(0.9)

        noise_fig, noise_ax1 = plt.subplots(figsize=(8, 4))

        # MSE
        noise_ax1.plot(self.noise_levels, self.mse_dict['clean_vs_recon'], color=mse_color, lw=2)
        # SSIM
        noise_ax2 = noise_ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        noise_ax2.plot(self.noise_levels, self.ssim_dict['clean_vs_recon'], color=ssim_color, lw=2)

        # for external noise, plot original vs noisy image comparison
        if self.noise_source == 'ext':
            noise_ax1.plot(self.noise_levels, self.mse_dict['clean_vs_noise'], color=mse_color, lw=2, ls='--')
            noise_ax2.plot(self.noise_levels, self.ssim_dict['clean_vs_noise'], color=ssim_color, lw=2, ls='--')
        # for internal noise, there is no 'noisy' image
        else:
            pass

        # axis range
        noise_ax1.set_ylim([0, 0.04])
        noise_ax2.set_ylim([0, 1.0])
        # axis labels
        noise_ax1.set_xlabel(r'Noise level ($\sigma_{%s}$)' % self.noise_source, fontsize=15)
        noise_ax1.set_ylabel('MSE', color=mse_color, fontsize=15)
        noise_ax2.set_ylabel('SSIM', color=ssim_color, fontsize=15)  # we already handled the x-label with ax1
        # axis ticks
        noise_ax1.set_xticks(self.noise_levels)
        # axis tick colors
        noise_ax1.tick_params(axis='y', labelcolor=mse_color)
        noise_ax2.tick_params(axis='y', labelcolor=ssim_color)
        # axis tick size
        noise_ax1.tick_params(axis='both', which='major', labelsize=15)
        noise_ax2.tick_params(axis='both', which='major', labelsize=15)
        # spines
        for axis in ['top', 'bottom', 'left', 'right']:
            noise_ax1.spines[axis].set_linewidth(1.5)

        noise_fig.tight_layout()  # otherwise the right y-label is slightly clipped

        noise_fig.show()

        return noise_fig

    def show_reconstruction_across_noise_level(self):

        num_row = 2
        num_col = len(self.noise_levels)


        fig, axs = plt.subplots(nrows=num_row, ncols=num_col, sharex='all', sharey='all', figsize=(2*num_col, 8))
        # Plot noisy input and reconstruction
        for col_i in range(num_col):
            # plot noisy images in the first row (index 0)
            axs[0, col_i].imshow(self.img_dict['input_img'][col_i], cmap='gray')
            axs[0, col_i].axis('off')
            axs[0, col_i].set_title(r'$\sigma_{%s}$ = %.2f' % (self.noise_source, self.noise_levels[col_i]))

            # plot reconstruction in the second row (index 1)
            axs[1, col_i].imshow(self.img_dict['recon_img'][col_i], cmap='gray')
            axs[1, col_i].axis('off')

        # Define the titles for each row.
        row_titles = ['noisy input', 'reconstruction']
        y_coords = np.linspace(1, 0, num_row + 1)
        for row_i, title in enumerate(row_titles):
            # Place the title between the top and bottom of its row.
            y_pos = (y_coords[row_i] + y_coords[row_i + 1]) / 2
            # Adjusting the position slightly to place it above the plots.
            y_pos += 0.4 / num_row
            fig.text(0.5, y_pos, title, ha='center', va='center', fontsize=12)

        fig.tight_layout()
        fig.show()

        return fig


# def plot_mse_ssim(noise_origin, noise_lvls, mse_dict, ssim_dict, colormap='PuOr'):
#
#     plt.close('all')
#
#     # pick color codes for MSE and SSIM
#     mse_color = '#298c8c' #plt.colormaps.get_cmap(colormap)(0.1)
#     ssim_color = '#f1a226' #plt.colormaps.get_cmap(colormap)(0.9)
#
#     noise_fig, noise_ax1 = plt.subplots(figsize=(8, 4))
#
#     # MSE
#     noise_ax1.plot(noise_lvls, mse_dict['noise_recon'], color=mse_color, lw=2)
#     # SSIM
#     noise_ax2 = noise_ax1.twinx()  # instantiate a second Axes that shares the same x-axis
#     noise_ax2.plot(noise_lvls, ssim_dict['noise_recon'], color=ssim_color, lw=2)
#
#     # for external noise, plot original vs noisy image comparison
#     if noise_origin == 'ext':
#         noise_ax1.plot(noise_lvls, mse_dict['noise'], color=mse_color, lw=2, ls='--')
#         noise_ax2.plot(noise_lvls, ssim_dict['noise'], color=ssim_color, lw=2, ls='--')
#     # for internal noise, there is no 'noisy' image
#     else:
#         pass
#
#     # axis range
#     noise_ax1.set_ylim([0, 0.04])
#     noise_ax2.set_ylim([0, 1.0])
#     # axis labels
#     noise_ax1.set_xlabel(r'Noise level ($\sigma_{%s}$)' % noise_origin, fontsize=15)
#     noise_ax1.set_ylabel('MSE', color=mse_color, fontsize=15)
#     noise_ax2.set_ylabel('SSIM', color=ssim_color, fontsize=15)  # we already handled the x-label with ax1
#     # axis ticks
#     noise_ax1.set_xticks(noise_lvls)
#     # axis tick colors
#     noise_ax1.tick_params(axis='y', labelcolor=mse_color)
#     noise_ax2.tick_params(axis='y', labelcolor=ssim_color)
#     # axis tick size
#     noise_ax1.tick_params(axis='both', which='major', labelsize=15)
#     noise_ax2.tick_params(axis='both', which='major', labelsize=15)
#     # spines
#     for axis in ['top', 'bottom', 'left', 'right']:
#         noise_ax1.spines[axis].set_linewidth(1.5)
#
#     noise_fig.tight_layout()  # otherwise the right y-label is slightly clipped
#
#     noise_fig.show()
#
#     return noise_fig
#
# def show_reconstruction_across_noise_level(sampled_img_dict, noise_lvls, noise_source):
#
#     num_row = 3
#     num_col = len(noise_lvls)
#
#     empty_box = np.zeros(sampled_img_dict['original_img'].shape)
#
#     fig, axs = plt.subplots(nrows=num_row, ncols=num_col, sharex='all', sharey='all')
#     for i, ax in enumerate(axs.flat):
#         col_i = i % num_col
#         if i == 0:
#             ax.imshow(sampled_img_dict['original_img'], cmap='gray')
#         elif (i > 0) and (i < num_col):
#             ax.imshow(empty_box, cmap='gray')
#         elif (i >= num_col) and (i < num_col * 2):
#             ax.imshow(sampled_img_dict['input_img'][col_i], cmap='gray')
#             ax.set_title('noisy input \n' + r' $\sigma_{%s}$ = %.2f' % (noise_source, noise_lvls[col_i]))
#         else:
#             ax.imshow(sampled_img_dict['recon_img'][col_i], cmap='gray')
#             ax.set_title('reconstruction')
#
#     fig.tight_layout()
#     fig.show()
#
#     return fig

    # # sample images in dictionary
    # sample_img_dict = {
    #     'original': self.imgs[0].reshape(*self.img_xy_dim),
    #     'noise': self.ext_noise,
    #     'noisy': self.imgs + noise
    # }
    #
    # # plot: [original, noise, noisy]
    # plot_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(3, 9))
    # for ax_i, (img_key, img) in enumerate(sample_img_dict.items()):
    #     axs[ax_i].imshow(img, **plot_args)
    #     axs[ax_i].axis('off')
    #     axs[ax_i].set_title(img_key)
    # fig.tight_layout()
    #
    # return fig, sample_img_dict
#
#
# def get_sample_imgs(img_dict, sigma_ext):
#     # pick a sample image
#     original = img_dict['original_img']
#     img_shape = original.shape
#
#     # generate noise
#     n, _ = noise_generator(noise_type='normal', noise_lvl=sigma_ext, target_shape=img_shape)
#
#     # sample images in dictionary
#     sample_img_dict = {
#         'original': original,
#         'noise': n,
#         'noisy': original + n
#     }
#
#     # plot: [original, noise, noisy]
#     plot_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
#     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(3, 9))
#     for ax_i, (img_key, img) in enumerate(sample_img_dict.items()):
#         axs[ax_i].imshow(img, **plot_args)
#         axs[ax_i].axis('off')
#         axs[ax_i].set_title(img_key)
#     fig.tight_layout()
#
#     return fig, sample_img_dict
#
#
# def get_recons(noise_lvls, img_dict):
#     plot_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
#     recon_fig, recon_axs = plt.subplots(1, img_dict['sample_imgs'].shape[0])
#     for i, img in enumerate(img_dict['sample_imgs']):
#         recon_axs[i].imshow(img, **plot_args)
#         recon_axs[i].axis('off')
#         recon_axs[i].set_title(f'{noise_lvls[i]:.2f}')
#     recon_fig.suptitle(r'Reconstructions with varying $\sigma_{%s}$' % "ext")
#     recon_fig.tight_layout()
#
#     return recon_fig
#
#
# def get_pop_response(noise_origin, noise_val, sim_params, pretrained_weights, dataset, ran_idcs, noise_type='normal'):
#     if noise_origin == 'ext':
#         noise_input, _ = noise_generator(
#             noise_type=noise_type, noise_lvl=noise_val, target_shape=dataset['test_x'].shape
#         )
#         noise_input = np.clip(noise_input + dataset['test_x'], 0.0, 1.0)
#         noise_jit_type = 'constant'
#         noise_jit_lvl = 0.0
#
#     else:
#         noise_input = dataset['test_x']
#         noise_jit_type = noise_type
#         noise_jit_lvl = noise_val
#
#     _, popResponse_net, popResponse_rep = test_reconstruction(
#         sim_param=sim_params, weights=pretrained_weights,
#         layer=0, input_vector=noise_input, sample_idcs=ran_idcs,
#         jit_type=noise_jit_type, jit_lvl=noise_jit_lvl
#     )
#
#     pop_activities = {
#         'rep': popResponse_rep.mean(axis=1).mean(axis=0),
#         'ppe': popResponse_net.errors['layer_0']['ppe_pyr'],
#         'npe': popResponse_net.errors['layer_0']['npe_pyr']
#     }
#
#     fig_title = r'with $\sigma_{%s}$ = %.2f' % (noise_origin, noise_val)
#     ax_labels = ['Representation', r'PE $+$', r'PE $-$']
#     ax_colors = ['purple', 'r', 'b']
#
#     popResponse_fig, popResponse_axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 6))
#     for ax_i, (pop_key, pop_fr) in enumerate(pop_activities.items()):
#         popResponse_axs[ax_i].plot(pop_fr, c=ax_colors[ax_i])
#         popResponse_axs[ax_i].set_title(ax_labels[ax_i], c=ax_colors[ax_i])
#
#         if ax_i == 1:
#             popResponse_axs[ax_i].set_ylabel('Firing rate (a.u.)')
#         elif ax_i == 2:
#             popResponse_axs[ax_i].set_xlabel('Time (ms)')
#         else:
#             pass
#
#     popResponse_fig.suptitle(f'Population activity ' + fig_title)
#     popResponse_fig.tight_layout()
#
#     return popResponse_fig, pop_activities
