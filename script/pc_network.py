import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import trange
import time
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import os
import datetime
from script.tools import (jorge, sigmoid, ReLu, sample_imgs, pickle_save,
                          generate_reconstruction_pad, create_dir)


def test_noise(
        sim_param: dict, pretrained_weights: np.ndarray,
        noise_levels: np.ndarray, noise_type: tuple,
        test_images: np.ndarray, test_sample_idcs: list,
        layer: object = 0
) -> object:
    """
    :param sim_param: simulation parameters
    :param pretrained_weights:
    :param noise_levels:
    :param noise_type: pick one from each list (['int', 'ext'], ['normal', 'uniform', 'constant'])
    :param test_images:
    :param test_sample_idcs:
    :param layer:
    :return:
    """

    # noise labels
    noise_source, noise_dist = noise_type

    # reconstruction parameters
    n_train_samples, _ = test_images.shape
    n_test_samples = len(test_sample_idcs)
    img_dim = int(np.sqrt(test_images.shape[1]))
    input_imgs = test_images[test_sample_idcs].T.reshape(
        img_dim, img_dim, n_test_samples
    )

    # initialize ssim, mse variables
    ssim_dict = initialize_metric_dict(perturb_lvls=noise_levels)
    mse_dict = initialize_metric_dict(perturb_lvls=noise_levels)
    img_dict = {
        'original_img': input_imgs[:, :, 0],
        'sample_imgs': np.zeros(
            (len(noise_levels), img_dim, img_dim)
        )
    }

    for noise_i, noise_val in enumerate(noise_levels):

        # training reconstruction
        plt.close('all')

        noise_desc = (
                r'$\eta_{%s}$' % noise_source
                + r' ~ $\mathcal{N}$' * (noise_dist == 'normal')
                + r' ~ $\matcal{U}$' * (noise_dist == 'uniform')
                + f' (0, {noise_val:.3f})' * (noise_dist in ['normal', 'uniform'])
                + f' = {noise_val:.3f}' * (noise_dist == 'constant')
        )

        # external noise
        if noise_source == 'ext':
            # generate noisy input
            noise_input, _ = noise_generator(
                noise_type=noise_dist, noise_lvl=noise_val, target_shape=test_images.shape
            )

        # internal noise (jitter)
        elif noise_source == 'int':
            noise_input = 0.0

        sim_inputs = np.clip(noise_input * (noise_source == 'ext') + test_images, 0.0, 1.0)
        sim_jit_dist = 'constant' * (noise_source == 'ext') + noise_dist * (noise_source == 'int')
        sim_jit_lvl = noise_val * (noise_source == 'int')

        # simulate inference
        noise_recon_fig, noise_net, noise_rep = test_reconstruction(
            sim_param=sim_param, weights=pretrained_weights,
            layer=layer, input_vector=sim_inputs, sample_idcs=test_sample_idcs,
            jit_type=sim_jit_dist, jit_lvl=sim_jit_lvl
        )

        # set title
        noise_recon_fig.suptitle(noise_desc)

        recon_imgs = (noise_net.weights['01'].T @ noise_net.network['layer_1']['rep_r']).reshape(
            img_dim, img_dim, n_test_samples)
        noisy_imgs = noise_net.network['layer_0']['rep_e'].reshape(
            img_dim, img_dim, n_test_samples)

        img_dict['sample_imgs'][noise_i] = recon_imgs[:, :, 0]

        # calculate ssim and mse
        for i in trange(n_test_samples, desc=f'{noise_i + 1}/{len(noise_levels)}', leave=False):
            # compare clean input vs noise input
            mse_dict['noise'][noise_i] += mse(input_imgs[:, :, i], noisy_imgs[:, :, i])
            ssim_dict['noise'][noise_i] += ssim(input_imgs[:, :, i], noisy_imgs[:, :, i],
                                                data_range=noisy_imgs[:, :, i].max() - noisy_imgs[:, :, i].min())

            # compare clean input vs reconstruction from noisy input
            mse_dict['noise_recon'][noise_i] += mse(input_imgs[:, :, i], recon_imgs[:, :, i])
            ssim_dict['noise_recon'][noise_i] += ssim(input_imgs[:, :, i], recon_imgs[:, :, i],
                                                      data_range=recon_imgs[:, :, i].max() - recon_imgs[:, :, i].min())

    mse_dict = {k: v / n_test_samples for k, v in mse_dict.items()}
    ssim_dict = {k: v / n_test_samples for k, v in ssim_dict.items()}

    return noise_net, noise_recon_fig, mse_dict, ssim_dict, img_dict


def noise_generator(noise_type, noise_lvl, target_shape):
    noise_func = {
        'uniform': [np.random.uniform, [-noise_lvl, noise_lvl],
                    r'external noise ~ $\mathcal{U}$' + f' (0, {noise_lvl:.3f})'],
        'normal': [np.random.normal, [0, noise_lvl],
                   r'external noise ~ $\mathcal{N}$' + f' (0, {noise_lvl:.3f})'],
        'constant': [return_self, [noise_lvl],
                     f'external noise = (0, {noise_lvl:.2f})']
    }

    return noise_func[noise_type][0](*noise_func[noise_type][1], target_shape), noise_func[noise_type][2]


def return_self(x, size):
    return np.ones(size) * x


def initialize_metric_dict(perturb_lvls, dict_keys=None):
    # initialize ssim, mse variables
    if dict_keys is None:
        dict_keys = ['noise', 'noise_recon']

    return {key_i: np.zeros(len(perturb_lvls)) for key_i in dict_keys}


def get_training_test_set(ds_key, num_class, num_sample, class_choice=None, max_fr=30, shuffle=True):
    training_set = {
        'mnist': tf.keras.datasets.mnist,
        'fmnist': tf.keras.datasets.fashion_mnist,
        'gray_cifar-10': tf.keras.datasets.cifar10
    }

    if ds_key not in list(training_set.keys()):
        raise ValueError('Please chooose from: mnist, fmnist, gray_cifar-10.')

    (x_train, y_train), (x_test, y_test) = training_set[ds_key].load_data()

    if ds_key == 'gray_cifar-10':
        x_train = tf.image.rgb_to_grayscale(x_train).numpy()[:, :, :, 0]
        x_test = tf.image.rgb_to_grayscale(x_test).numpy()[:, :, :, 0]
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]

    train_idx = sample_imgs(y_train, num_class, num_sample, class_choice=class_choice)
    test_idx = sample_imgs(y_test, num_class, num_sample, class_choice=class_choice)

    if shuffle:
        np.random.shuffle(train_idx)

    data_train = x_train[train_idx] / 255.0 * max_fr
    data_test = x_test[test_idx] / 255.0 * max_fr

    n_imgs, dim_x, dim_y = data_train.shape

    data_train = data_train.reshape(n_imgs, dim_x * dim_y)
    label_train = y_train[train_idx]
    data_test = data_test.reshape(n_imgs, dim_x * dim_y)
    label_test = y_test[test_idx]

    return data_train, label_train, data_test, label_test


def generate_input(simParams, class_choice=None, shuffle=False):
    input_type = simParams['dataset']
    num_class = simParams['n_class']
    num_sample = simParams['n_sample']
    max_fr = simParams['max_fr']

    x_input, y_input, x_test, y_test = get_training_test_set(
        ds_key=input_type,
        num_class=num_class, num_sample=num_sample,
        max_fr=max_fr,
        class_choice=class_choice, shuffle=shuffle
    )

    img_dim = np.sqrt(x_input.shape[1]).astype(int)

    if num_class * num_sample > 1:
        if num_class == 1:
            fig = plt.figure()
            plt.imshow(x_input[0].reshape(img_dim, img_dim), cmap='gray')
            plt.title(input_type)
        else:
            # find one sample per class
            sample_idcs = [np.argwhere(y_input == y_unique)[0][0] for y_unique in np.unique(y_input)]

            fig, axs = plt.subplots(1, num_class, figsize=(2 * num_class, 2))
            for ax_i, ax in enumerate(axs):
                img_idx = sample_idcs[ax_i]
                img = ax.imshow(x_input[img_idx].reshape(img_dim, img_dim), cmap='gray')
                ax.axis("off")
                fig.colorbar(img, ax=ax, shrink=0.6)
                ax.set_title(f'{input_type}: class {ax_i + 1}')
    else:
        fig, axs = plt.subplots(1, 1)
        img = axs.imshow(x_input[0].reshape(img_dim, img_dim), cmap='gray')
        axs.axis("off")
        fig.colorbar(img, shrink=0.6)
        axs.set_title(input_type)

    fig.tight_layout()

    dataset = {
        'train_x': x_input,
        'train_y': y_input,
        'test_x': x_test,
        'test_y': y_test
    }

    return dataset, fig


def oddball_input(img1, img2, n_repeat):
    oddball_inputs = np.array([img1] * n_repeat + [img2] + [img1])
    n_imgs, n_pixel = oddball_inputs.shape
    img_dim = int(np.sqrt(n_pixel))

    odd_fig, odd_axs = plt.subplots(nrows=1, ncols=n_imgs, sharex='all', sharey='all')
    for seq_i, ax in enumerate(odd_axs.flatten()):
        ax.imshow(oddball_inputs[seq_i].reshape(img_dim, img_dim), cmap='gray')
        ax.axis('off')
        ax.set_title(f'seq #{seq_i + 1}')
    odd_fig.suptitle('Oddball stimulus sequence')
    odd_fig.tight_layout()

    return oddball_inputs, odd_fig


def test_oddball(
        sim_params, weights, oddball_x,
        record='error',
        bg_exc=0.0, bg_inh=0.0,
        jit_type='constant', jit_lvl=0.0
):
    oddball_net = network(
        simParams=sim_params,
        pretrained_weights=weights,
        bg_exc=bg_exc, bg_inh=bg_inh, jitter_lvl=jit_lvl, jitter_type=jit_type
    )

    if sim_params['symmetric_weight']:
        computor = oddball_net.compute
    else:
        computor = oddball_net.compute_ind

    #
    n_imgs, n_pixel = oddball_x.shape
    # initialize network
    oddball_net.initialize_network(batch_size=1)
    # reset error
    oddball_net.initialize_error()

    steps_isi = int(sim_params['isi_time'] / sim_params['dt'])
    steps_sim = int(sim_params['sim_time'] / sim_params['dt'])

    for i in trange(n_imgs):

        # isi
        for t_step in range(steps_isi):
            computor(inputs=np.zeros(oddball_x[i].reshape(-1, 1).shape), record=record)
        # test_net.add_err()

        # stimulus presentation
        for t_step in range(steps_sim):
            computor(inputs=oddball_x[i].reshape(-1, 1), record=record)

        # output figure
        total_t = n_imgs * (steps_isi + steps_sim)
        re_fig = plt.figure()
        if record == 'error':
            # error figure
            re_axs = re_fig.add_subplot(111)
            re_axs.plot(oddball_net.errors['layer_0']['ppe_pyr'], label='PE+')
            re_axs.plot(oddball_net.errors['layer_0']['npe_pyr'], label='PE-')
            re_axs.set_xlabel('time (ms)')
            re_axs.set_ylabel('Firing rate (Hz)')
            # re_axs.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
            leg = re_axs.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
            for h, t in zip(leg.legendHandles, leg.get_texts()):
                t.set_color(h.get_facecolor()[0])

        elif record == 'all':
            # error + rep figure
            re_ax1 = re_fig.add_subplot(311)
            re_ax1.plot(oddball_net.errors['layer_1']['rep_r'], label='rep', c='purple')
            re_ax2 = re_fig.add_subplot(312)
            re_ax2.plot(oddball_net.errors['layer_0']['ppe_pyr'], label='PE+', c='r')
            re_ax3 = re_fig.add_subplot(313)
            re_ax3.plot(oddball_net.errors['layer_0']['npe_pyr'], label='PE-', c='b')

            for ax_i, ax in enumerate(re_fig.axes):

                if ax_i == 2:
                    ax.set_xlabel('Time (ms)')
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                if ax_i == 1:
                    ax.set_ylabel('Firing rate (a.u.)')
                else:
                    ax.set_ylabel('')

            # leg = re_fig.legend(loc='upper right')
            # for h, t in zip(leg.legendHandles, leg.get_texts()):
            #     t.set_color(h.get_color())

        elif record == 'interneurons':

            in_colors = ['Blues', 'Greens', 'Wistia']
            for ax_i, (neuron_group, response) in enumerate(oddball_net.errors['layer_0'].items()):
                axi = re_fig.add_subplot(2, 3, ax_i + 1)
                axi.plot(response, c=plt.cm.get_cmap(in_colors[ax_i % 3])(0.75), lw=2)
                axi.set_title(neuron_group)
                if ax_i < 3:
                    axi.set_xticklabels([])
                elif ax_i == 4:
                    axi.set_xlabel('Time (ms)')

        for ax in re_fig.axes:
            create_vlines(
                target_axes=ax, total_sim_time=total_t,
                trial_sim_time=int(sim_params['sim_time'] / sim_params['dt']),
                interval_time=int(sim_params['isi_time'] / sim_params['dt']))
            remove_top_right_spines(target_axes=ax)

    return oddball_net, re_fig


def create_vlines(target_axes, total_sim_time, trial_sim_time, interval_time):
    for yi in np.arange(0, total_sim_time, trial_sim_time + interval_time):
        target_axes.axvline(x=yi, ls='--', c='black')
        target_axes.axvline(x=yi + interval_time, ls='--', c='black')


def remove_top_right_spines(target_axes, target_spines=None):
    if target_spines is None:
        target_spines = ['top', 'right']
    for spine in target_spines:
        target_axes.spines[spine].set_visible(False)


# def get_recon_in_time():
#     fig, axs = plt.subplots(11, n_imgs, sharex='all', sharey='all', figsize=(8, 11))
#     for i, ax in enumerate(axs.flatten()):
#         if i < n_imgs:
#             ax.imshow(oddball_x[i].reshape(28, 28), cmap='gray')
#             # pass
#         else:
#             img_idx = i % n_imgs
#             time_var = i // n_imgs
#             time_slices = slice(100 * (time_var - 1), 100 * time_var)
#             ax.imshow((oddball_net.weights['01'].T @ rep_save[img_idx, :, time_slices].mean(axis=1)).reshape(28, 28),
#                       cmap='gray')
#
#     cols = [f'Seq {i + 1}' for i in range(n_imgs)]
#     rows = ['Input'] + [f'{i * 100 + 100} ms' for i in range(10)]
#
#     for ax, col in zip(axs[0], cols):
#         ax.set_title(col)
#
#     for ax, row in zip(axs[:, 0], rows):
#         ax.set_ylabel(row, rotation=90, size='large')
#
#     for ax in axs.flat:
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     fig.suptitle('reconstruction every 100 ms')
#     fig.savefig(project_dir + 'one_layer_model/oddball/recon_in_time.png', dpi=300)
#     fig.show()


def infer_from_existing_model(
        sim_params, weights, datasets,
        bg_exc=0.0, bg_inh=0.0, jitter_lvl=0.0, jitter_type='constant'
):
    train_x, train_y, test_x, test_y = datasets.values()
    show_idx = sim_params['recon_img_idcs']

    test_net = network(
        simParams=sim_params, pretrained_weights=weights,
        bg_exc=bg_exc, bg_inh=bg_inh, jitter_lvl=jitter_lvl, jitter_type=jitter_type
    )

    test_net.initialize_network(batch_size=test_x[show_idx].T.shape[1])
    test_net.initialize_error()

    steps_isi = int(sim_params['isi_time'] / sim_params['dt'])
    steps_sim = int(sim_params['sim_time'] / sim_params['dt'])

    # isi
    for t_step in range(steps_isi):
        test_net.compute(inputs=np.zeros(test_x[show_idx].T.shape), record='error')

    # stimulus presentation
    rep_save = np.zeros((*test_net.network['layer_1']['rep_r'].shape, 1000))
    for t_step in range(steps_sim):
        test_net.compute(inputs=test_x[show_idx].T, record='all')
        # rep_save += model.network['layer_1']['rep_r']
        rep_save[:, :, t_step] = test_net.network['layer_1']['rep_r']

    return test_net, rep_save


# def test_opto(model):
#
#     neuron_types = ['ppe_pyr', 'ppe_pv', 'ppe_sst', 'ppe_vip', 'npe_pyr', 'npe_pv', 'npe_sst', 'npe_vip']
#
#     for ni in len(neuron_types):
#     # for nt in neuron_types:
#
#         opto_net = network(
#             neurons_per_layer=model.net_size,
#             bu_rate=model.bu_rate, td_rate=model.td_rate,
#             tau_exc=model.tau_exc, tau_inh=model.tau_inh,
#             symm_w=True, pretrained_weights=model.weights,
#             bg_exc=0.0, bg_inh=0.0, jitter_lvl=0.0, jitter_type='constant'
#         )
#
#         tlayer = 'layer_0'
#         tneuron = neuron_types[ni]
#         opto_net.activate_silencer(target_layer=tlayer, target_neuron=tneuron)
#         computor = opto_net.compute
#         opto_net.initialize_network(batch_size=test_x[noise_recon_idx].T.shape[1])
#         opto_net.initialize_error()
#         # isi
#         for t_step in range(100):
#             computor(inputs=np.zeros(test_x[noise_recon_idx].T.shape), record=None)
#         rep_save = np.zeros((*opto_net.network['layer_1']['rep_r'].shape, 1000))
#         # stimulus presentation
#         for t_step in range(1000):
#             computor(inputs=test_x[noise_recon_idx].T, record='error')
#             # rep_save += model.network['layer_1']['rep_r']
#             rep_save[:, :, t_step] = opto_net.network['layer_1']['rep_r']
#
#         layer = 0
#         input_fr = opto_net.network[f'layer_{layer}']['rep_e']
#         pred_fr = opto_net.weights[f'{layer}{layer + 1}'].T @ opto_net.network[f'layer_{layer + 1}']['rep_r']
#
#         # plot reconstruction
#
#         _, _, input_pad = generate_reconstruction_pad(img_mat=input_fr)  # , nx=4)
#         _, _, pred_pad = generate_reconstruction_pad(img_mat=pred_fr)  # , nx=4)
#
#         recon_fig, recon_axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all',
#                                             figsize=(10, 4))
#         plt_min = np.min(input_fr)
#         plt_max = np.max(input_fr)
#         input_imgs = recon_axs[0].imshow(input_pad, cmap='gray', vmin=plt_min, vmax=plt_max)
#         recon_fig.colorbar(input_imgs, ax=recon_axs[0], shrink=0.4)
#         recon_axs[0].set_title('input')
#         pred_imgs = recon_axs[1].imshow(pred_pad, cmap='gray', vmin=plt_min, vmax=plt_max)
#         recon_fig.colorbar(pred_imgs, ax=recon_axs[1], shrink=0.4)
#         recon_axs[1].set_title('prediction')
#         for ax in recon_axs.flatten():
#             ax.axis('off')
#
#         recon_fig.suptitle(f'reconstruction at layer {layer}')
#         recon_fig.tight_layout()
#
#         recon_fig.savefig(model_dir + f'opto/{tneuron}_recon_scaled.png', dpi=300)
#         print(nt + ' ends!')
#
#     reordering_idx = []
#     for yi in np.unique(test_y[noise_recon_idx]):
#         reordering_idx.append(np.argwhere(test_y[noise_recon_idx] == yi).flatten())
#
#     # # label
#     # opto_reps = opto_net.network['layer_1']['rep_r'][:, reordering_idx]
#     # opto_inputs = opto_net.network['layer_0']['rep_e'][:, reordering_idx]
#     # opto_labels = test_y[noise_recon_idx][reordering_idx]
#     #
#     # rdm_list = []
#     #
#     # n_neuron, n_sample = opto_reps.shape
#     # obs_desc = {'digits': np.array([str(x) for x in opto_labels])}
#     # # construct dataset for RDM
#     # chn_desc = {'neurons': np.array([f'neuron_{str(x + 1)}' for x in np.arange(opto_reps.shape[0])])}
#     #
#     # rdm_list.append(
#     #     rsatoolbox.data.Dataset(
#     #         measurements=opto_inputs.T,
#     #         descriptors={'Area': 'input'},
#     #         obs_descriptors=obs_desc,
#     #         channel_descriptors={'neurons': np.array([f'neuron_{str(x + 1)}' for x in np.arange(opto_inputs.shape[0])])}
#     #     )
#     # )
#     #
#     # rdm_list.append(
#     #     rsatoolbox.data.Dataset(
#     #         measurements=opto_reps.T,
#     #         descriptors={r'Area': 'Area 1, $r_{VIP-} = 0$' },
#     #         obs_descriptors=obs_desc,
#     #         channel_descriptors={'neurons': np.array([f'neuron_{str(x + 1)}' for x in np.arange(opto_reps.shape[0])])}
#     #     )
#     # )
#     #
#     # rdm_corr = rsatoolbox.rdm.calc_rdm(
#     #         rdm_list,
#     #         method='correlation'
#     #     )
#
#     # RDM plot
#     opto_rdm, rdm_axs, rdm_dict = rsatoolbox.vis.show_rdm(rdm_corr,
#                                                           cmap='RdBu', show_colorbar='figure',
#                                                           rdm_descriptor='Area',
#                                                           n_column=2)
#
#     return opto_rdm

def test_reconstruction(sim_param, weights,
                        layer, input_vector, sample_idcs,
                        record='error',
                        bg_exc=0.0, bg_inh=0.0, jit_type='constant', jit_lvl=0.0):

    # input_vector = (n_sample, n_pixel) -> test_samples = (n_pixel, n_sample)
    test_samples = input_vector[sample_idcs].T

    if test_samples.shape == 1:
        test_samples = test_samples.reshape(test_samples.shape[0], 1)
    else:
        pass

    # simulate
    test_net, rep = run_test(
        sim_param=sim_param, weights=weights, inputs_x=test_samples,
        record=record,
        bg_exc=bg_exc, bg_inh=bg_inh, jit_type=jit_type, jit_lvl=jit_lvl
    )

    if sim_param['symmetric_weight']:

        # input_fr = pred_fr = (n_pixel, n_sample)
        input_fr = test_net.network[f'layer_{layer}']['rep_e']
        pred_fr = test_net.weights[f'{layer}{layer + 1}'].T @ test_net.network[f'layer_{layer + 1}']['rep_r']

        # plot reconstruction

        nx = len(sample_idcs) // 5
        _, _, input_pad = generate_reconstruction_pad(img_mat=input_fr, nx=nx)
        _, _, pred_pad = generate_reconstruction_pad(img_mat=pred_fr, nx=nx)

        recon_fig, recon_axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
        # figsize=(int(len(sample_idcs / nx) * 2), nx))
        input_imgs = recon_axs[0].imshow(input_pad, cmap='gray', vmin=0.0, vmax=1.0)
        recon_fig.colorbar(input_imgs, ax=recon_axs[0], shrink=0.4)
        recon_axs[0].set_title('input')
        pred_imgs = recon_axs[1].imshow(pred_pad, cmap='gray', vmin=0.0, vmax=1.0)
        recon_fig.colorbar(pred_imgs, ax=recon_axs[1], shrink=0.4)
        recon_axs[1].set_title('prediction')
        for ax in recon_axs.flatten():
            ax.axis('off')

        recon_fig.suptitle(f'reconstruction at layer {layer}')
        recon_fig.tight_layout()

    else:
        recon_fig = test_net.batch_plots_ind(1, 1, test_net.network[f'layer_{layer}']['rep_e'].shape[1])

    return recon_fig, test_net, rep


def run_test(
        sim_param, weights, inputs_x, record='error', bg_exc=0.0, bg_inh=0.0, jit_type='constant', jit_lvl=0.0
):
    # FIX to have sim_params as input
    test_net = network(
        simParams=sim_param, pretrained_weights=weights,
        bg_exc=bg_exc, bg_inh=bg_inh, jitter_lvl=jit_lvl, jitter_type=jit_type
    )

    if sim_param['symmetric_weight']:
        computor = test_net.compute
    else:
        computor = test_net.compute_ind

    # initialize network
    test_net.initialize_network(batch_size=inputs_x.shape[1])
    # reset error
    test_net.initialize_error()

    # isi
    steps_isi = int(sim_param['isi_time'] / sim_param['dt'])
    for _ in trange(steps_isi):
        computor(inputs=np.zeros(inputs_x.shape), record=None)

    # stimulus presentation
    steps_sim = int(sim_param['sim_time'] / sim_param['dt'])
    # rep saver
    rep_save = np.zeros((*test_net.network['layer_1']['rep_r'].shape, steps_sim))
    # sim
    for t_step in trange(steps_sim):
        computor(inputs=inputs_x, record=record)
        # rep_save += model.network['layer_1']['rep_r']
        rep_save[:, :, t_step] = test_net.network['layer_1']['rep_r']

    # rep_save /= t_sim

    return test_net, rep_save


class network(object):

    def __init__(
            self,
            simParams,
            pretrained_weights=None,
            bg_exc=0.0, bg_inh=0.0,
            jitter_lvl=0.0, jitter_type='constant',
    ):

        self.bu_rate = simParams['bu_rate']
        self.td_rate = simParams['td_rate']
        self.tau_exc = simParams['tau_exc']
        self.tau_inh = simParams['tau_inh']

        self.bg_exc = bg_exc
        self.bg_inh = bg_inh

        self.resolution = simParams['dt']

        self.net_size = simParams['net_size']

        self.run_error = {}
        self.weights = {}
        self.np_weights = {f'layer_{i}': {} for i in range(len(self.net_size))}
        self.network = {}
        self.silence_target = []

        if simParams['symmetric_weight']:
            self.initialize_weights(weights=pretrained_weights, jitter_type=jitter_type, jitter_lvl=jitter_lvl)
        else:
            self.initialize_ind_weights(weights=pretrained_weights)

        self.noise = False

    def generate_np_weights(self, jitter_lvl, jitter_type, target_shape, layer='curr'):

        # pe_exc * 4, pe_inh * 3, rep * 3, in * 4
        # in = ['e, pyr+', 'e, pyr-', 'e, pv-', 'e, sst-']
        if layer == 'curr':
            np_weight_keys = [
                'pyr, pyr', 'pyr, sst', 'pyr, pv', 'pyr, vip',
                'sst, pyr', 'pv, pyr', 'vip, sst',
                'e, pyr+', 'e, pyr-', 'e, pv-', 'e, sst-'
            ]
        elif layer == 'above':
            np_weight_keys = ['e, r', 'i, e']
        else:
            raise ValueError('Choose either curr or above.')

        np_weights = {
            np_weight: np.maximum(
                1 + noise_generator(noise_type=jitter_type, noise_lvl=jitter_lvl, target_shape=(target_shape, 1))[0],
                0.0
            )
            for np_weight in np_weight_keys
        }

        return np_weights

    def initialize_weights(self, weights, jitter_type, jitter_lvl):

        if weights is None:
            # inter-layer syn str
            for pc_i, layer in enumerate(self.net_size):
                # random intialization
                if pc_i > 0:
                    self.weights[f'{pc_i - 1}{pc_i}'] = np.random.uniform(
                        size=(self.net_size[pc_i], self.net_size[pc_i - 1])
                    ) / np.sqrt(self.net_size[pc_i - 1])
                    # self.weights[f'{pc_i - 1}{pc_i}'] = {
                    #     'ppe': np.random.uniform(
                    #         size=(self.net_size[pc_i], self.net_size[pc_i - 1])
                    #     ) / np.sqrt(self.net_size[pc_i - 1]),
                    #     'npe': np.random.uniform(
                    #         size=(self.net_size[pc_i], self.net_size[pc_i - 1])
                    #     ) / np.sqrt(self.net_size[pc_i - 1])
                    # }
                    self.np_weights[f'layer_{pc_i - 1}'].update(
                        self.generate_np_weights(
                            jitter_lvl=jitter_lvl, jitter_type=jitter_type, target_shape=self.net_size[pc_i - 1]
                        )
                    )
                    self.np_weights[f'layer_{pc_i}'].update(
                        self.generate_np_weights(
                            jitter_lvl=jitter_lvl, jitter_type=jitter_type,
                            target_shape=self.net_size[pc_i], layer='above'
                        )
                    )

                    # self.weights[f'{pc_i - 1}{pc_i}'] = np.ones(
                    #     shape=(self.net_size[pc_i], self.net_size[pc_i - 1])
                    # ) / np.sqrt(self.net_size[pc_i - 1] * self.net_size[pc_i])
                    # self.weights[f'{i - 1}{i}'] = sprandn(
                    #     m=self.net_size[i],
                    #     n=self.net_size[i - 1],
                    #     density=0.2,
                    #     data_rvs=norm(loc=1 / np.sqrt(self.net_size[i - 1]), scale=0.1).rvs
                    # ).A
        # import the pretrained weights
        else:
            self.weights = weights
            for pc_i, layer in enumerate(self.net_size):
                if pc_i > 0:
                    self.np_weights[f'layer_{pc_i - 1}'].update(
                        self.generate_np_weights(
                            jitter_lvl=jitter_lvl, jitter_type=jitter_type, target_shape=self.net_size[pc_i - 1]
                        )
                    )
                    self.np_weights[f'layer_{pc_i}'].update(
                        self.generate_np_weights(
                            jitter_lvl=jitter_lvl, jitter_type=jitter_type,
                            target_shape=self.net_size[pc_i], layer='above'
                        )
                    )

    # np weights not implemented
    def initialize_ind_weights(self, weights):

        self.setup_ind_learning()

        # bu_weights = [
        #     'ppe_pyr, rep_e', 'npe_pyr, rep_i'
        # ]
        # td_weights = [
        #     'rep_r, ppe_pv', 'rep_r, ppe_sst', 'rep_r, ppe_vip',
        #     'rep_r, npe_pyr', 'rep_r, npe_sst', 'rep_r, npe_vip'
        # ]
        if weights is None:
            # inter-layer syn str
            for pc_i, layer in enumerate(self.net_size):

                # random intialization

                if pc_i > 0:
                    self.weights[f'{pc_i - 1}{pc_i}'] = {
                        weight_label: np.random.uniform(
                            size=(self.net_size[pc_i], self.net_size[pc_i - 1])
                        ) / np.sqrt(self.net_size[pc_i - 1])
                        for weight_label in self.bu_weight_keys
                    }
                    self.weights[f'{pc_i - 1}{pc_i}'].update({
                        weight_label: np.random.uniform(
                            size=(self.net_size[pc_i - 1], self.net_size[pc_i])
                        ) / np.sqrt(self.net_size[pc_i - 1])
                        for weight_label in self.td_weight_keys
                    })

        # import the pretrained weights
        else:
            self.weights = weights

    def initialize_network(self, batch_size):

        self.bat_size = batch_size

        rep_neurons = ['e', 'i', 'r']
        neuron_types = ['pyr', 'pv', 'sst', 'vip']
        pe_types = ['ppe', 'npe']

        # make this layerwise: for example, 3-layer has layer 0, 1, and 2

        for i, layer in enumerate(self.net_size):

            self.network[f'layer_{i}'] = {}

            # populate the network with different neuron types
            for rep in rep_neurons:
                self.network[f'layer_{i}'][f'rep_{rep}'] = np.zeros((self.net_size[i], self.bat_size))
            # self.network[f'layer_{i}']['rep_r'] = sprandn(
            #     m=self.net_size[i],
            #     n=self.bat_size,
            #     density=0.2,
            #     data_rvs=norm(loc=15, scale=5).rvs,
            #     random_state=np.random.seed(1234)
            # ).A  # .flatten()
            if i < len(self.net_size) - 1:
                for pe in pe_types:
                    for neuron in neuron_types:
                        self.network[f'layer_{i}'][f'{pe}_{neuron}'] = np.zeros(
                            (self.net_size[i], self.bat_size))  # , T_steps))

                # if i < len(self.net_size) - 1:
                self.run_error[f'layer_{i}'] = {
                    'ppe_pyr': [],
                    'npe_pyr': []
                }

    def initialize_error(self):

        self.errors = {
            layer: {
                # 'ppe': [],
                # 'npe': []
            }
            for layer, neurons in self.network.items() if layer != f'layer_{len(self.network)}'
        }

    def add_err(self):
        # mean across samples and neurons
        for i in range(len(self.net_size) - 1):
            self.run_error[f'layer_{i}']['ppe_pyr'].append(np.mean(self.errors[f'layer_{i}']['ppe_pyr']))
            self.run_error[f'layer_{i}']['npe_pyr'].append(np.mean(self.errors[f'layer_{i}']['npe_pyr']))

    def activate_silencer(self, target_layer, target_neuron):
        self.silence_target.append((target_layer, target_neuron))

    def deactivate_silencer(self, target_layer, target_neuron):
        self.silence_target.remove((target_layer, target_neuron))

    def dr(self, r, inputs, func='relu', ei_type='inh', silence=False):

        if ei_type == 'exc':
            tau = self.tau_exc
            bg = self.bg_exc
        elif ei_type == 'inh':
            tau = self.tau_inh
            bg = self.bg_inh
        else:
            raise ValueError('not supported')

        if func == 'jorge':
            fx = jorge(x=inputs)
        elif func == 'sigmoid':
            fx = sigmoid(x=inputs)
        elif func == 'relu':
            fx = ReLu(x=inputs)
        else:
            raise ValueError('not supported')

        new_r = r + (-r + fx + np.random.normal(loc=0.0, scale=bg, size=fx.shape)) * (self.resolution / tau)

        if silence:
            return np.zeros(new_r.shape)
        else:
            return new_r

    def compute(self, inputs, record='error'):

        self.inputs = inputs
        # ['layer_0', 'layer_1', 'layer_2']
        # ['rep_e', 'rep_i', 'rep_r', 'ppe_pyr', 'ppe_pv', 'ppe_sst', 'npe_pyr', 'npe_pv', 'npe_sst']
        # (n_neuron, size_batch)

        # np_weights
        # np_weight_keys = [
        #     'pyr, pyr', 'pyr, sst', 'pyr, pv', 'pyr, vip',
        #     'sst, pyr', 'pv, pyr', 'vip, sst',
        #     'e, r', 'i, e',
        #     'e, pyr+', 'e, pyr-', 'e, pv-', 'e, sst-'
        # ]

        for layer_i, (layer, neurons) in enumerate(self.network.items()):

            td_rate = self.td_rate  # * 10 ** (-(layer_i - 1))
            bu_rate = self.bu_rate  # * 10 ** (-(layer_i - 1))

            if layer_i < len(self.network) - 1:

                # input layer
                if layer_i == 0:
                    # rep_e
                    self.network[layer]['rep_e'] = self.dr(
                        r=self.network[layer]['rep_e'],
                        inputs=self.inputs,
                        ei_type='exc',
                        silence=(layer, 'rep_e') in self.silence_target
                    )

                # middle layers
                else:
                    w_ppe = (
                            self.weights[f'{layer_i - 1}{layer_i}'] @
                            self.network[f'layer_{layer_i - 1}']['ppe_pyr']
                    )
                    w_npe = (
                            self.weights[f'{layer_i - 1}{layer_i}'] @
                            self.network[f'layer_{layer_i - 1}']['npe_pyr']
                    )
                    # w_ppe = self.weights[f'{layer_i - 1}{layer_i}']['ppe'] @
                    # self.network[f'layer_{layer_i - 1}']['ppe_pyr']
                    # w_npe = self.weights[f'{layer_i - 1}{layer_i}']['npe'] @
                    # self.network[f'layer_{layer_i - 1}']['npe_pyr']

                    # rep_e
                    self.network[layer]['rep_e'] = self.dr(
                        r=self.network[layer]['rep_e'],
                        inputs=(
                                td_rate * self.network[layer]['npe_pyr'] +
                                bu_rate * w_ppe
                        ),  # -
                        # self.np_weights[layer]['i, e'] * self.network[layer]['rep_i'],
                        ei_type='exc',
                        silence=(layer, 'rep_e') in self.silence_target
                    )
                    # rep_i
                    self.network[layer]['rep_i'] = self.dr(
                        r=self.network[layer]['rep_i'],
                        inputs=(
                                td_rate * self.network[layer]['ppe_pyr'] +
                                bu_rate * w_npe
                        ),
                        silence=(layer, 'rep_i') in self.silence_target
                    )
                    # rep_r
                    # n_neurons = self.network[layer]['rep_r'].shape[0]
                    # w_val = 0.0
                    # later_w = np.ones((n_neurons, n_neurons)) * -w_val
                    # np.fill_diagonal(later_w, w_val)
                    self.network[layer]['rep_r'] = self.dr(
                        r=self.network[layer]['rep_r'],
                        inputs=(
                                self.np_weights[layer]['e, r'] * self.network[layer]['rep_e'] -
                                self.np_weights[layer]['i, e'] * self.network[layer]['rep_i'] +
                                self.network[layer]['rep_r']
                            # + self.noise * np.random.normal(loc=0, scale=self.bg_exc, size=self.network[layer]['rep_r'].shape)
                        ),
                        ei_type='exc',
                        silence=(layer, 'rep_r') in self.silence_target
                    )

                prediction = self.weights[f'{layer_i}{layer_i + 1}'].T @ self.network[f'layer_{layer_i + 1}']['rep_r']
                # prediction_ppe = self.weights[f'{layer_i}{layer_i + 1}']['ppe'].T @
                # self.network[f'layer_{layer_i + 1}']['rep_r']
                # prediction_npe = self.weights[f'{layer_i}{layer_i + 1}']['npe'].T @
                # self.network[f'layer_{layer_i + 1}']['rep_r']

                # ppe_pyr
                self.network[layer]['ppe_pyr'] = self.dr(
                    r=self.network[layer]['ppe_pyr'],
                    inputs=(
                            self.np_weights[layer]['e, pyr+'] * self.network[layer]['rep_e'] -
                            self.np_weights[layer]['pv, pyr'] * self.network[layer]['ppe_pv'] -
                            self.np_weights[layer]['sst, pyr'] * self.network[layer]['ppe_sst'] +
                            self.np_weights[layer]['pyr, pyr'] * self.network[layer]['ppe_pyr']
                        # + self.noise * np.random.normal(loc=0, scale=self.bg_exc, size=self.network[layer]['ppe_pyr'].shape)
                    ),
                    ei_type='exc',
                    silence=(layer, 'ppe_pyr') in self.silence_target
                )
                # ppe_pv
                self.network[layer]['ppe_pv'] = self.dr(
                    r=self.network[layer]['ppe_pv'],
                    inputs=(
                            self.np_weights[layer]['pyr, pv'] * self.network[layer]['ppe_pyr'] +
                            prediction
                    ),
                    # prediction_ppe,
                    silence=(layer, 'ppe_pv') in self.silence_target
                )
                # ppe_sst
                self.network[layer]['ppe_sst'] = self.dr(
                    r=self.network[layer]['ppe_sst'],
                    inputs=(
                            self.np_weights[layer]['pyr, sst'] * self.network[layer]['ppe_pyr'] -
                            self.np_weights[layer]['vip, sst'] * self.network[layer]['ppe_vip'] +
                            prediction
                    ),
                    # prediction_ppe,
                    silence=(layer, 'ppe_sst') in self.silence_target
                )
                # ppe_vip
                self.network[layer]['ppe_vip'] = self.dr(
                    r=self.network[layer]['ppe_vip'],
                    inputs=(
                            self.np_weights[layer]['pyr, vip'] * self.network[layer]['ppe_pyr'] +
                            prediction
                    ),
                    # prediction_ppe,
                    silence=(layer, 'ppe_vip') in self.silence_target
                )

                # npe_pyr
                self.network[layer]['npe_pyr'] = self.dr(
                    r=self.network[layer]['npe_pyr'],
                    inputs=(
                            self.np_weights[layer]['e, pyr-'] * self.network[layer]['rep_e'] +
                            prediction +
                            # prediction_npe +
                            self.np_weights[layer]['pyr, pyr'] * self.network[layer]['npe_pyr'] -
                            self.np_weights[layer]['pv, pyr'] * self.network[layer]['npe_pv'] -
                            self.np_weights[layer]['sst, pyr'] * self.network[layer]['npe_sst']
                        # + self.noise * np.random.normal(loc=0, scale=self.bg_exc, size=self.network[layer]['npe_pyr'].shape)
                    ),
                    ei_type='exc',
                    silence=(layer, 'npe_pyr') in self.silence_target
                )
                # npe_pv
                self.network[layer]['npe_pv'] = self.dr(
                    r=self.network[layer]['npe_pv'],
                    inputs=(
                            self.np_weights[layer]['e, pv-'] * self.network[layer]['rep_e'] +
                            self.np_weights[layer]['pyr, pv'] * self.network[layer]['npe_pyr']
                    ),
                    silence=(layer, 'npe_pv') in self.silence_target
                )
                # npe_sst
                self.network[layer]['npe_sst'] = self.dr(
                    r=self.network[layer]['npe_sst'],
                    inputs=(
                            self.np_weights[layer]['e, sst-'] * self.network[layer]['rep_e'] +
                            prediction +
                            # prediction_npe +
                            self.np_weights[layer]['pyr, sst'] * self.network[layer]['npe_pyr'] -
                            self.np_weights[layer]['vip, sst'] * self.network[layer]['npe_vip']
                    ),
                    silence=(layer, 'npe_sst') in self.silence_target
                )
                # npe_vip
                self.network[layer]['npe_vip'] = self.dr(
                    r=self.network[layer]['npe_vip'],
                    inputs=(
                            prediction +
                            # prediction_npe +
                            self.np_weights[layer]['pyr, vip'] * self.network[layer]['npe_pyr']
                    ),
                    silence=(layer, 'npe_vip') in self.silence_target
                )

            # top layer
            else:
                ppe = self.weights[f'{layer_i - 1}{layer_i}'] @ self.network[f'layer_{layer_i - 1}']['ppe_pyr']
                npe = self.weights[f'{layer_i - 1}{layer_i}'] @ self.network[f'layer_{layer_i - 1}']['npe_pyr']
                # ppe = self.weights[f'{layer_i - 1}{layer_i}']['ppe'] @ self.network[f'layer_{layer_i - 1}']['ppe_pyr']
                # npe = self.weights[f'{layer_i - 1}{layer_i}']['npe'] @ self.network[f'layer_{layer_i - 1}']['npe_pyr']

                # rep_e
                self.network[layer]['rep_e'] = self.dr(
                    r=self.network[layer]['rep_e'],
                    inputs=ppe,
                    ei_type='exc',
                    silence=(layer, 'rep_e') in self.silence_target
                )
                # rep_i
                self.network[layer]['rep_i'] = self.dr(
                    r=self.network[layer]['rep_i'],
                    inputs=npe,
                    silence=(layer, 'rep_i') in self.silence_target
                )
                # rep_r
                self.network[layer]['rep_r'] = self.dr(
                    r=self.network[layer]['rep_r'],
                    inputs=(
                            self.np_weights[layer]['e, r'] * self.network[layer]['rep_e'] -
                            self.np_weights[layer]['i, e'] * self.network[layer]['rep_i'] +
                            self.network[layer]['rep_r']
                    ),
                    ei_type='exc',
                    silence=(layer, 'rep_r') in self.silence_target
                )
        if record is not None:
            self.record(record)

    def compute_ind(self, inputs, record='error'):

        self.inputs = inputs
        # ['layer_0', 'layer_1', 'layer_2']
        # ['rep_e', 'rep_i', 'rep_r', 'ppe_pyr', 'ppe_pv', 'ppe_sst', 'npe_pyr', 'npe_pv', 'npe_sst']
        # (n_neuron, size_batch)
        # bu_weights = ['ppe_pyr, rep_e', 'npe_pyr, rep_i']
        # td_weights = ['rep_r, ppe_pv', 'rep_r, ppe_sst', 'rep_r, ppe_vip',
        #               'rep_r, npe_pyr', 'rep_r, npe_sst', 'rep_r, npe_vip']

        for layer_i, (layer, neurons) in enumerate(self.network.items()):

            td_rate = self.td_rate  # * 10 ** (-(layer_i - 1))
            bu_rate = self.bu_rate  # * 10 ** (-(layer_i - 1))

            if layer_i < len(self.network) - 1:

                # input layer
                if layer_i == 0:
                    # rep_e
                    self.network[layer]['rep_e'] = self.dr(
                        r=self.network[layer]['rep_e'],
                        inputs=self.inputs,
                        ei_type='exc'
                    )

                # middle layers
                else:
                    w_ppe = (
                            self.weights[f'{layer_i - 1}{layer_i}']['ppe_pyr, rep_e'] @
                            self.network[f'layer_{layer_i - 1}']['ppe_pyr']
                    )
                    w_npe = (
                            self.weights[f'{layer_i - 1}{layer_i}']['npe_pyr, rep_i'] @
                            self.network[f'layer_{layer_i - 1}']['npe_pyr']
                    )

                    # rep_e
                    self.network[layer]['rep_e'] = self.dr(
                        r=self.network[layer]['rep_e'],
                        inputs=(
                                td_rate * self.network[layer]['npe_pyr'] +
                                bu_rate * w_ppe -
                                self.network[layer]['rep_i']
                        ),
                        ei_type='exc'
                    )
                    # rep_i
                    self.network[layer]['rep_i'] = self.dr(
                        r=self.network[layer]['rep_i'],
                        inputs=(
                                td_rate * self.network[layer]['ppe_pyr'] +
                                bu_rate * w_npe +
                                self.network[layer]['rep_e']
                        )
                    )
                    # rep_r
                    self.network[layer]['rep_r'] = self.dr(
                        r=self.network[layer]['rep_r'],
                        inputs=self.network[layer]['rep_e'],
                        ei_type='exc'
                    )

                # ppe_pyr
                self.network[layer]['ppe_pyr'] = self.dr(
                    r=self.network[layer]['ppe_pyr'],
                    inputs=(
                            self.network[layer]['rep_e'] -
                            self.network[layer]['ppe_pv'] -
                            self.network[layer]['ppe_sst'] +
                            self.network[layer]['ppe_pyr']
                    ),
                    ei_type='exc'
                )
                # ppe_pv
                prediction = (
                        self.weights[f'{layer_i}{layer_i + 1}']['rep_r, ppe_pv'] @
                        self.network[f'layer_{layer_i + 1}']['rep_r']
                )
                self.network[layer]['ppe_pv'] = self.dr(
                    r=self.network[layer]['ppe_pv'],
                    inputs=self.network[layer]['ppe_pyr'] + prediction
                )
                # ppe_sst
                prediction = (
                        self.weights[f'{layer_i}{layer_i + 1}']['rep_r, ppe_sst'] @
                        self.network[f'layer_{layer_i + 1}']['rep_r']
                )
                self.network[layer]['ppe_sst'] = self.dr(
                    r=self.network[layer]['ppe_sst'],
                    inputs=(
                            self.network[layer]['ppe_pyr'] -
                            self.network[layer]['ppe_vip'] +
                            prediction
                    )
                )
                # ppe_vip
                prediction = (
                        self.weights[f'{layer_i}{layer_i + 1}']['rep_r, ppe_vip'] @
                        self.network[f'layer_{layer_i + 1}']['rep_r']
                )
                self.network[layer]['ppe_vip'] = self.dr(
                    r=self.network[layer]['ppe_vip'],
                    inputs=(
                            self.network[layer]['ppe_pyr'] +
                            prediction
                    )
                )

                # npe_pyr
                prediction = (
                        self.weights[f'{layer_i}{layer_i + 1}']['rep_r, npe_pyr'] @
                        self.network[f'layer_{layer_i + 1}']['rep_r']
                )
                self.network[layer]['npe_pyr'] = self.dr(
                    r=self.network[layer]['npe_pyr'],
                    inputs=(
                            self.network[layer]['rep_e'] +
                            prediction +
                            self.network[layer]['npe_pyr'] -
                            self.network[layer]['npe_pv'] -
                            self.network[layer]['npe_sst']
                    ),
                    ei_type='exc'
                )
                # npe_pv
                self.network[layer]['npe_pv'] = self.dr(
                    r=self.network[layer]['npe_pv'],
                    inputs=(
                            self.network[layer]['rep_e'] +
                            self.network[layer]['npe_pyr']
                    )
                )
                # npe_sst
                prediction = (
                        self.weights[f'{layer_i}{layer_i + 1}']['rep_r, npe_sst'] @
                        self.network[f'layer_{layer_i + 1}']['rep_r']
                )
                self.network[layer]['npe_sst'] = self.dr(
                    r=self.network[layer]['npe_sst'],
                    inputs=(
                            self.network[layer]['rep_e'] +
                            prediction +
                            self.network[layer]['npe_pyr'] -
                            self.network[layer]['npe_vip']
                    )
                )
                # npe_vip
                prediction = (
                        self.weights[f'{layer_i}{layer_i + 1}']['rep_r, npe_vip'] @
                        self.network[f'layer_{layer_i + 1}']['rep_r']
                )
                self.network[layer]['npe_vip'] = self.dr(
                    r=self.network[layer]['npe_vip'],
                    inputs=prediction + self.network[layer]['npe_pyr']
                )

            # top layer
            else:
                ppe = (
                        self.weights[f'{layer_i - 1}{layer_i}']['ppe_pyr, rep_e'] @
                        self.network[f'layer_{layer_i - 1}']['ppe_pyr']
                )
                npe = (
                        self.weights[f'{layer_i - 1}{layer_i}']['npe_pyr, rep_i'] @
                        self.network[f'layer_{layer_i - 1}']['npe_pyr']
                )

                # rep_e
                self.network[layer]['rep_e'] = self.dr(
                    r=self.network[layer]['rep_e'],
                    inputs=bu_rate * ppe - self.network[layer]['rep_i'],
                    ei_type='exc'
                )
                # rep_i
                self.network[layer]['rep_i'] = self.dr(
                    r=self.network[layer]['rep_i'],
                    inputs=bu_rate * npe + self.network[layer]['rep_e']
                )
                # rep_r
                self.network[layer]['rep_r'] = self.dr(
                    r=self.network[layer]['rep_r'],
                    inputs=self.network[layer]['rep_e'],
                    ei_type='exc'
                )
        if record is not None:
            self.record(record)

    def add_noise(self):
        self.noise = True

    def get_error(self):

        for layer, _ in self.errors.items():
            self.errors[layer]['ppe'].append(
                # average over neurons
                np.mean(
                    # average over batches
                    np.mean(self.network[layer]['ppe_pyr'], axis=1),
                )
            )
            self.errors[layer]['npe'].append(
                # average over neurons
                np.mean(
                    # average over batches
                    np.mean(self.network[layer]['npe_pyr'], axis=1)
                )
            )

    def record(self, var):

        if var == 'error':
            var_list = ['ppe_pyr', 'npe_pyr']
        elif var == 'rep':
            var_list = ['rep_r']
        elif var == 'all':
            var_list = ['ppe_pyr', 'npe_pyr', 'rep_r']
        elif var == 'interneurons':
            pe_circuits = ['ppe', 'npe']
            neurons = ['pv', 'sst', 'vip']
            var_list = [f'{pe_i}_{n_i}' for pe_i in pe_circuits for n_i in neurons]
        else:
            raise ValueError('Please choose valid neuron group (s).')

        for layer, neuron_dict in self.errors.items():
            for var in var_list:
                if (var not in neuron_dict.keys()) and (var in self.network[layer].keys()):
                    neuron_dict[var] = []
                else:
                    pass

        for layer, neuron_dict in self.errors.items():
            for neuron_i, fr in neuron_dict.items():
                if neuron_i in var_list:
                    # neuron_vals = np.mean(self.network[layer][neuron_i], axis=1)
                    # if neuron_i != 'rep_r':
                    #     neuron_vals = np.mean(neuron_vals)
                    # self.errors[layer][neuron_i].append(neuron_vals)
                    self.errors[layer][neuron_i].append(
                        # average over neurons
                        np.mean(
                            # average over batches
                            np.mean(self.network[layer][neuron_i], axis=1),
                        )
                    )

    # def reset_error(self):
    #
    #         self.errors = {
    #             layer: {
    #                 'ppe': [],
    #                 'npe': []
    #             }
    #             for layer, neurons in self.network.items() if layer != f'layer_{len(self.network) - 1}'
    #         }

    # make this function add to the list, instead of computing mse every instance
    def plot_error(self, curr_ep, n_ep):

        # error plot
        err_len = len(self.network) - 1

        if err_len == 1:
            err_fig, err_axs = plt.subplots(3, 1)
            for i, (layer, pes) in enumerate(self.errors.items()):
                if i < err_len:
                    err_axs[0].plot(pes['ppe_pyr'], c='r')
                    err_axs[0].set_title('pPE')
                    err_axs[1].plot(pes['npe_pyr'], c='b', label='nPE')
                    err_axs[1].set_title('nPE')
                    err_axs[2].plot(np.add(pes['ppe_pyr'], pes['npe_pyr']), c='black')
                    err_axs[2].set_title('PE')

        else:
            err_fig, err_axs = plt.subplots(nrows=len(self.network) - 1, ncols=3, sharex='all')  # , sharey='col')
            # err_axs = err_axs.flatten()
            for i, (layer, pes) in enumerate(self.errors.items()):
                if i < err_len:
                    err_axs[i, 0].plot(pes['ppe_pyr'], c='r', label='pPE')
                    err_axs[i, 1].plot(pes['npe_pyr'], c='b', label='nPE')
                    err_axs[i, 2].plot([sum(x) for x in zip(pes['ppe_pyr'], pes['npe_pyr'])], c='black', label='PE')

                    # label: layer
                    err_axs[i, 1].set_title(f'{layer}')
                    # label: x-axis
                    err_axs[i, 0].set_ylabel('firing rate (Hz)')

                    # legend
                    if i == 0:
                        err_fig.legend(loc='upper right')
                    elif i == len(self.errors) - 1:
                        err_axs[i, 1].set_xlabel('time (ms)')

        err_fig.suptitle(f'MSE at epoch #{curr_ep}/{n_ep}')
        err_fig.tight_layout()

        return err_fig

    def plot_run_error(self, epoch_i, tsim):

        net_len = len(self.net_size) - 1
        re_fig, re_axs = plt.subplots(nrows=3, ncols=net_len, sharex='all')
        if net_len == 1:
            re_axs = re_axs.reshape(-1, 1)

        for i in range(len(self.net_size) - 1):
            ppe = self.run_error[f'layer_{i}']['ppe_pyr']
            npe = self.run_error[f'layer_{i}']['npe_pyr']
            pe = [sum(x) for x in zip(ppe, npe)]

            err_slice = slice(epoch_i * tsim, (epoch_i + 1) * tsim)

            re_axs[0, i].plot(ppe[err_slice], c='r')
            re_axs[0, i].set_title('pPE')
            re_axs[0, i].set_xticks(np.arange(0, tsim + 1, tsim // 5))
            re_axs[0, i].set_xticklabels([])

            re_axs[1, i].plot(npe[err_slice], c='b')
            re_axs[1, i].set_title('nPE')
            re_axs[1, i].set_ylabel('MSE')
            re_axs[1, i].set_xticks(np.arange(0, tsim + 1, tsim // 5))
            re_axs[1, i].set_xticklabels([])

            re_axs[2, i].plot(pe[err_slice], c='black')
            re_axs[2, i].set_title('PE')
            re_axs[2, i].set_xlabel('training epoch #')
            re_axs[2, i].set_xticks(np.arange(0, tsim + 1, tsim // 5))
            re_axs[2, i].set_xticklabels(np.arange(0, tsim + 1, tsim // 5) + epoch_i * tsim)

        re_fig.suptitle('MSE over training epochs')
        re_fig.tight_layout()

        return re_fig

    def learn(self, lr=1e-2, alpha=1e-4):

        for i, (layers, weight) in enumerate(self.weights.items()):
            l1_reg = alpha * np.maximum(0.0, weight)
            # l1_reg_ppe = alpha * np.maximum(0.0, weight['ppe'])
            # l1_reg_npe = alpha * np.maximum(0.0, weight['npe'])

            dwp = lr * np.einsum(
                # 'ij, kj -> ki',
                'ik, jk -> ji',
                self.network[f'layer_{i}']['ppe_pyr'],
                self.network[f'layer_{i + 1}']['rep_r']
            ) / (2 * self.bat_size)
            dwn = lr * np.einsum(
                # 'ij, kj -> ki',
                'ik, jk -> ji',
                self.network[f'layer_{i}']['npe_pyr'],
                self.network[f'layer_{i + 1}']['rep_r']
            ) / (2 * self.bat_size)

            # self.weights[layers]['ppe'] = np.maximum(0.0, weight['ppe'] + dwp - l1_reg_ppe)
            # self.weights[layers]['npe'] = np.maximum(0.0, weight['npe'] - dwn - l1_reg_npe)
            self.weights[layers] = np.maximum(0.0, weight + dwp - dwn - l1_reg)
            # self.weights[layers] = weight + dwp - dwn

    def setup_ind_learning(self):

        self.bu_weight_keys = ['ppe_pyr, rep_e', 'npe_pyr, rep_i']
        self.td_weight_keys = ['rep_r, ppe_pv', 'rep_r, ppe_sst', 'rep_r, ppe_vip',
                               'rep_r, npe_pyr', 'rep_r, npe_sst', 'rep_r, npe_vip']
        weight_sign = [1, 1, -1, 1, -1, 1]
        self.w_signs = {bu_w.split(', ')[1]: 1 for bu_w in self.bu_weight_keys}
        self.w_signs.update({td_w.split(', ')[1]: 1 * ws for td_w, ws in zip(self.td_weight_keys, weight_sign)})

    def learn_ind(self, lr=1e-2, alpha=1e-4):

        # bu_weights = ['ppe_pyr, rep_e', 'npe_pyr, rep_i']
        # td_weights = ['rep_r, ppe_pv', 'rep_r, ppe_sst', 'rep_r, ppe_vip',
        #               'rep_r, npe_pyr', 'rep_r, npe_sst', 'rep_r, npe_vip']
        # weight_sign = [1, 1, -1, 1, -1, 1]
        # w_signs = {bu_w.split(', ')[1]: 1 for bu_w in bu_weights}
        # w_signs.update({td_w.split(', ')[1]: ws for td_w, ws in zip(td_weights, weight_sign)})

        for layer_idx, (layer, weight_dict) in enumerate(self.weights.items()):
            for neuron_pairs, weights in weight_dict.items():

                neuron_i, neuron_j = neuron_pairs.split(', ')

                if neuron_pairs in self.bu_weight_keys:
                    layer_i = f'layer_{layer_idx}'
                    layer_j = f'layer_{layer_idx + 1}'
                else:
                    layer_i = f'layer_{layer_idx + 1}'
                    layer_j = f'layer_{layer_idx}'

                dw = -lr * np.einsum(
                    'ij, kj -> ki',
                    self.network[layer_i][neuron_i],
                    self.w_signs[neuron_j] * self.network[layer_j][neuron_j]
                ) / self.bat_size

                l1_reg = alpha * np.maximum(0.0, weights)
                self.weights[layer][neuron_pairs] = np.maximum(0.0, weights - dw - l1_reg)

    def plots(self, curr_ep, n_epoch, idx=None):

        # input, pred, err
        ncol = 4
        # layers
        nrow = len(self.network)

        # plot
        int_fig = plt.figure(figsize=(ncol * 3, nrow * 3))
        grid_plot = plt.GridSpec(nrows=nrow, ncols=ncol)

        for i, (layer, _) in enumerate(self.network.items()):

            # input to layer
            if idx is None:
                layer_input = self.network[layer]['rep_e']
            else:
                layer_input = self.network[layer]['rep_e'][:, idx]
            img_dim = np.sqrt(layer_input.shape[0]).astype(int)
            input_ax = int_fig.add_subplot(grid_plot[i, 0])
            input_image = input_ax.imshow(layer_input.reshape(img_dim, img_dim), cmap='gray')
            int_fig.colorbar(input_image, ax=input_ax, shrink=0.4)
            input_ax.axis('off')
            input_ax.set_title(f'L{i} input')

            if layer != f'layer_{len(self.network) - 1}':
                # pred to layer
                if idx is None:
                    layer_pred = self.weights[f'{i}{i + 1}'].T @ self.network[f'layer_{i + 1}']['rep_r']
                else:
                    layer_pred = self.weights[f'{i}{i + 1}'].T @ self.network[f'layer_{i + 1}']['rep_r'][:, idx]
                img_dim = np.sqrt(layer_pred.shape[0]).astype(int)
                pred_ax = int_fig.add_subplot(grid_plot[i, 1])
                pred_image = pred_ax.imshow(layer_pred.reshape(img_dim, img_dim), cmap='gray')
                int_fig.colorbar(pred_image, ax=pred_ax, shrink=0.4)
                pred_ax.axis('off')
                pred_ax.set_title(f'L{i} pred')

                # +ve err to layer
                if idx is None:
                    layer_ppe = self.network[layer]['ppe_pyr']
                else:
                    layer_ppe = self.network[layer]['ppe_pyr'][:, idx]
                img_dim = np.sqrt(layer_ppe.shape[0]).astype(int)
                ppe_ax = int_fig.add_subplot(grid_plot[i, 2])
                ppe_image = ppe_ax.imshow(layer_ppe.reshape(img_dim, img_dim), cmap='Reds')
                int_fig.colorbar(ppe_image, ax=ppe_ax, shrink=0.4)
                ppe_ax.axis('off')
                ppe_ax.set_title(f'L{i} pPE')

                # -ve err to layer
                if idx is None:
                    layer_npe = self.network[layer]['npe_pyr']
                else:
                    layer_npe = self.network[layer]['npe_pyr'][:, idx]
                img_dim = np.sqrt(layer_npe.shape[0]).astype(int)
                npe_ax = int_fig.add_subplot(grid_plot[i, 3])
                npe_image = npe_ax.imshow(layer_npe.reshape(img_dim, img_dim), cmap='Blues')
                int_fig.colorbar(npe_image, ax=npe_ax, shrink=0.4)
                npe_ax.axis('off')
                npe_ax.set_title(f'L{i} nPE')

        int_fig.tight_layout()
        int_fig.suptitle(f'Epoch #{curr_ep}/{n_epoch}')

        return int_fig

    def batch_plots(self, img_idcs, curr_ep, n_epoch):

        # input, pred, err+, err-
        ncol = 4
        # layers
        nrow = len(self.net_size) - 1

        # plot
        batch_fig, batch_ax = plt.subplots(
            nrows=nrow, ncols=ncol, sharex='row', sharey='row',
            figsize=(ncol * 5, nrow * 5)
        )
        batch_ax = batch_ax.flatten()
        for i, (layer, _) in enumerate(self.network.items()):

            if i < nrow:
                # input to layer
                # rand_idx = np.random.choice(np.arange(self.network[layer]['rep_e'].shape[1]), 16, replace=False)
                layer_input = self.network[layer]['rep_e'][:, img_idcs]
                layer_pred = (self.weights[f'{i}{i + 1}'].T @ self.network[f'layer_{i + 1}']['rep_r'])[:, img_idcs]
                layer_ppe = self.network[layer]['ppe_pyr'][:, img_idcs]
                layer_npe = self.network[layer]['npe_pyr'][:, img_idcs]

                subplt_inp = [layer_input, layer_pred, layer_ppe, layer_npe]
                subplt_idx = [i * ncol + j for j in range(ncol)]
                subplt_lbl = ['input', 'prediction', 'PE+', 'PE-']
                subplt_clr = ['gray', 'gray', 'Reds', 'Blues']

                for k in range(len(subplt_idx)):
                    self.add_subplot(batch_ax[subplt_idx[k]], batch_fig, subplt_inp[k], subplt_clr[k])
                    batch_ax[subplt_idx[k]].set_title(f'L{i} {subplt_lbl[k]}')

        batch_fig.suptitle(f'Epoch #{curr_ep}/{n_epoch}')

        return batch_fig

    @staticmethod
    def add_subplot(ax, fig, inputs, colors='gray'):

        x_ticks, y_ticks, image = generate_reconstruction_pad(
            img_mat=inputs, nx=4
        )
        input_subplot = ax.imshow(image, cmap=colors)
        ax.set_xticks(np.arange(*x_ticks))
        ax.set_yticks(np.arange(*y_ticks))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, ls='--')
        fig.colorbar(input_subplot, ax=ax, shrink=0.2)
        ax.axis('off')

        # return ax

    def batch_plots_ind(self, curr_ep, n_epoch, n_img_show=16):

        # input, pred, err+, err-
        ncol = 9
        # layers
        nrow = len(self.network) - 1

        # plot
        batch_fig, batch_ax = plt.subplots(
            nrows=nrow, ncols=ncol, sharex='row', sharey='row',
            figsize=(ncol * 3, nrow * 3)
        )
        batch_ax = batch_ax.flatten()
        for i, (layer, _) in enumerate(self.network.items()):

            if i < nrow:
                # input to layer
                rand_idx = np.random.choice(np.arange(self.network[layer]['rep_e'].shape[1]), n_img_show, replace=False)
                layer_input = self.network[layer]['rep_e'][:, rand_idx]

                # get predictions to each neuron type
                rep_r = self.network[f'layer_{i + 1}']['rep_r']
                layer_pred = {}
                for weight_pair_key, weight_pair in self.weights[f'{i}{i + 1}'].items():
                    pre_neuron, post_neuron = weight_pair_key.split(', ')
                    if pre_neuron == 'rep_r':
                        layer_pred[post_neuron] = (weight_pair @ rep_r)[:, rand_idx]

                layer_ppe = self.network[layer]['ppe_pyr'][:, rand_idx]
                layer_npe = self.network[layer]['npe_pyr'][:, rand_idx]

                subplt_inp = [layer_input] + [pred_r for _, pred_r in layer_pred.items()] + [layer_ppe, layer_npe]
                subplt_idx = [i * ncol + j for j in range(ncol)]
                subplt_lbl = ['input'] + [pred_key for pred_key, _ in layer_pred.items()] + ['PE+', 'PE-']
                subplt_clr = ['gray'] * 7 + ['Reds', 'Blues']

                for k in range(len(subplt_idx)):
                    self.add_subplot(batch_ax[subplt_idx[k]], batch_fig, subplt_inp[k], subplt_clr[k])
                    batch_ax[subplt_idx[k]].set_title(f'L{i} {subplt_lbl[k]}')

        batch_fig.suptitle(f'Epoch #{curr_ep}/{n_epoch}')

        return batch_fig


def error_plot(model, n_batch, n_epoch, time_steps):
    # extraploate mse per batch and epoch to plot against mse per tstep
    batch_x = np.linspace(0, n_batch * n_epoch * time_steps + 1, n_batch * n_epoch)
    epoch_x = np.linspace(0, n_batch * n_epoch * time_steps + 1, n_epoch)

    # row = layer, col = PE type
    net_len = len(model.net_size) - 1
    fig, axs = plt.subplots(nrows=len(model.net_size) - 1, ncols=2, sharex='all', figsize=(10, 5))
    if net_len == 1:
        axs = axs.reshape(1, -1)

    err_dict = {}
    for i, (layer_i, pe_dict) in enumerate(model.errors.items()):
        err_dict[layer_i] = {}
        for j, (signed_err, errors) in enumerate(pe_dict.items()):
            # save to dict
            err_dict[layer_i][signed_err] = {
                'per_tstep': errors,
                'per_batch': [np.mean(errors[batch_i * time_steps: (1 + batch_i) * time_steps])
                              for batch_i in range(n_batch * n_epoch)],
                'per_epoch': [np.mean(errors[epoch_i * (n_batch * time_steps): (1 + epoch_i) * (n_batch * time_steps)])
                              for epoch_i in range(n_epoch)]
            }
            # plot
            axs[i, j].plot(
                err_dict[layer_i][signed_err]['per_tstep'],
                c='b', alpha=0.2, label='layer avg'
            )
            axs[i, j].plot(
                batch_x, err_dict[layer_i][signed_err]['per_batch'],
                c='g', alpha=0.4, label='batch avg'
            )
            axs[i, j].plot(
                epoch_x, err_dict[layer_i][signed_err]['per_epoch'],
                c='r', label='epoch avg', alpha=1.0)

            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].set_title(f'{layer_i}, {signed_err}')
            axs[i, j].set_xticks([])

            if i == len(pe_dict.keys()) - 2:
                axs[i, j].set_xticks(
                    np.arange(
                        len(err_dict[layer_i][signed_err]['per_tstep']) + 1
                    )[::time_steps * n_batch]
                )
                axs[i, j].set_xticklabels(np.arange(n_epoch + 1))

    # axis labels
    fig.text(0.5, 0.005, 'Training epoch # ', ha='center')
    fig.text(0.05, 0.5, 'Firing rate (Hz)', va='center', rotation='vertical')

    # legend
    handles, labels = fig.axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    return err_dict, fig


def save_model(save_path, dataset, params, weights):
    # create model directory
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

    # save dataset
    pickle_save(save_path, 'dataset.pkl', dataset)
    # save params
    pickle_save(save_path, 'sim_params.pkl', params)
    # save weights
    pickle_save(save_path, 'weights.pkl', weights)


def run_simulation(simParams, dataset, save=True):

    create_dir('../results/')
    # define input size
    input_size = dataset['train_x'].shape[1]
    # define layer size
    net_size = [input_size, 28 ** 2]  # , 12 ** 2]
    # update sim parameters
    simParams.update({'net_size': net_size})

    # simulation params
    time_steps = int(simParams['sim_time'] / simParams['dt'])
    isi_steps = int(simParams['isi_time'] / simParams['dt'])
    n_batch = simParams['n_class'] * simParams['n_sample'] // simParams['batch_size']
    n_epoch = simParams['n_epoch']

    # plot interval (ms)
    plot_interval = simParams['plot_interval']
    # weight update interval (ms)
    learn_interval = time_steps // 1
    simParams.update({'learn_interval': learn_interval})

    recon_img_idcs = np.random.choice(simParams['batch_size'], simParams['recon_sample_n'], replace=False)
    simParams.update({'recon_img_idcs': recon_img_idcs})

    # create network
    net = network(simParams=simParams, pretrained_weights=None)
    # initialize network
    net.initialize_network(batch_size=simParams['batch_size'])
    # initialize mean error log
    net.initialize_error()

    start_time = time.time()
    # for epoch_i in trange(n_epoch, desc='epoch'):
    for epoch_i in range(n_epoch):

        # for img_i in range(n_img):
        for batch_i in range(n_batch):

            # reset network
            net.initialize_network(batch_size=simParams['batch_size'])
            # define current batch to feed
            curr_batch = dataset['train_x'][
                         batch_i * simParams['batch_size']: (batch_i + 1) * simParams['batch_size']].T

            # inter-stimulus interval
            for _ in range(isi_steps):
                # simulate batch
                net.compute(inputs=np.zeros(curr_batch.shape))

            # stimulus presentation
            for _ in trange(
                    time_steps,
                    desc=f'Epoch #{epoch_i + 1}/{n_epoch}, batch #{batch_i + 1}/{n_batch}',
                    leave=False
            ):
                # simulate batch
                net.compute(inputs=curr_batch, record='error')

            # weight update
            net.learn(lr=simParams['lr'], alpha=simParams['w_decay'])

            # plot intermediate results
            if ((epoch_i == 0) or ((epoch_i + 1) % plot_interval == 0)) and ((batch_i + 1) == n_batch):
                plt.close('all')

                int_fig = net.batch_plots(recon_img_idcs, epoch_i + 1, simParams['n_epoch'])
                int_fig.show()

                err_dict, err_plt = error_plot(net, n_batch, epoch_i + 1, time_steps + isi_steps)
                err_plt.suptitle(f'Epoch #{epoch_i + 1}/{n_epoch}')
                err_plt.show()

    end_time = time.time()
    total_sim_time = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print(f'total simulation time: {total_sim_time}')

    # save
    if save:
        # save trained weights
        pickle_save(simParams['model_dir'], 'weights.pkl', net.weights)
        # save training and test datasets
        pickle_save(simParams['model_dir'], 'dataset.pkl', dataset)
        # save simulation parameters
        pickle_save(simParams['model_dir'], 'sim_params.pkl', simParams)
        # save prediction error firing rates across training epochs
        pickle_save(simParams['model_dir'], 'errors.pkl', net.errors)
    else:
        pass

    return net

def plot_training_errors(sim_params, errs):

    # fig 2B - errors
    tsim = int(sim_params['sim_time'] / sim_params['dt'])
    tisi = int(sim_params['isi_time'] / sim_params['dt'])
    plt.close('all')
    aa = np.zeros(sim_params['n_epoch'])
    bb = np.zeros(sim_params['n_epoch'])
    len_epoch = int(tsim + tisi) * int(sim_params['n_class'] * sim_params['n_sample'] / sim_params['batch_size'])
    for i in range(sim_params['n_epoch']):
        curr_epoch_ppe = errs['layer_0']['ppe_pyr'][i * len_epoch: (i + 1) * len_epoch]
        curr_epoch_npe = errs['layer_0']['npe_pyr'][i * len_epoch: (i + 1) * len_epoch]
        aa[i] = curr_epoch_ppe[-1]
        bb[i] = curr_epoch_npe[-1]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(aa, c='#CA181D', lw=3.0, label='PE+')
    ax.plot(bb, c='#2070B4', lw=3.0, label='PE-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_xlabel('training iteration', fontsize=20)
    ax.set_ylabel('firing rate (a.u.)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.legend(labelcolor='linecolor', fontsize=20, edgecolor='white')  # , fancybox=True, shadow=True)
    fig.tight_layout()

    return fig