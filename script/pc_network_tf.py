import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import trange
from scipy.sparse import random as sprandn
from scipy.stats import norm
from tools import jorge, sigmoid, ReLu, sample_imgs, generate_reconstruction_pad

def input_modifier(inp, noise, occlude):
    img_dim = tf.sqrt(inp.shape[1]).astype(int)

    # augment inputs
    if noise != 0:
        inp_pattern = tf.nn.relu(
            inp + tf.random.normal(shape=inp.shape, mean=0.0, stddev=noise, dtype=tf.float32))

    else:
        if occlude:
            n_samples = inp.shape[0]
            inp_mask = np.ones((n_samples, img_dim, img_dim))
            for mask_i in range(n_samples):
                x_start = np.random.choice(np.arange(5, 18), 1)[0]
                mask_range = slice(x_start, x_start + 9)
                inp_mask[mask_i, mask_range, mask_range] = tf.reduce_min(inp)
            inp_mask = tf.reshape(inp_mask, shape=inp.shape)
            inp_pattern = inp * tf.cast(inp_mask, tf.float32)
        else:
            inp_pattern = inp

    # (n_neurons, n_samples)
    return tf.transpose(inp_pattern)

def generate_input(input_type, num_class, num_sample, max_fr, shuffle=False):

    if input_type == 'mnist':
        # mnist: lr = 5.0
        (x_train, y_train), (_, y_test) = tf.keras.datasets.mnist.load_data()
        x_idx = sample_imgs(y_train, num_class, num_sample, class_choice=None)
        x_train = tf.constant(x_train)
        plt_title = 'mnist digit'

    elif input_type == 'fmnist':
        # fashion mnist: lr = 5.0, nepoch=30
        (x_train, y_train), (_, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_idx = sample_imgs(y_train, num_class, num_sample, class_choice=None)
        x_train = tf.constant(x_train)
        plt_title = 'fashion mnist'

    elif input_type == 'gray_cifar-10':
        # grayscale cifar-10
        (x_train, y_train), (_, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = tf.image.rgb_to_grayscale(x_train)
        x_idx = sample_imgs(y_train, num_class, num_sample, class_choice=None)
        # x_idx = np.random.choice(np.arange(len(y_train)), n_imgs)
        plt_title = 'cifar-10'
    else:
        raise ValueError('not a supported type')

    x_input = tf.gather(x_train, x_idx) / 255.0 * max_fr

    if num_class * num_sample > 1:
        if num_class == 1:
            fig = plt.figure()
            plt.imshow(x_input[0], cmap='gray')
            plt.title(plt_title)
        else:
            fig, axs = plt.subplots(1, num_class, figsize=(2 * num_class, 2))
            for ax_i, ax in enumerate(axs):
                img = ax.imshow(x_input[::num_sample][ax_i], cmap='gray')
                ax.axis("off")
                fig.colorbar(img, ax=ax, shrink=0.6)
                ax.set_title(f'{plt_title}: class {ax_i + 1}')
    else:
        fig, axs = plt.subplots(1, 1)
        img = axs.imshow(x_input[0], cmap='gray')
        axs.axis("off")
        fig.colorbar(img, shrink=0.6)
        axs.set_title(plt_title)

    fig.tight_layout()

    x_input = tf.resahpe(x_input, (x_input.shape[0], x_input.shape[1] * x_input.shape[2]))
    y_input = y_train[x_idx]

    if shuffle:
        # shuffle idx
        idx = np.arange(len(x_input))
        np.random.shuffle(idx)
        x_input = tf.gather(x_input, idx)
        y_input = y_input[idx]

    else:
        pass

    return x_input, y_input, fig


def oddball_input(x, y, n_repeat):

    img_dim = len(x)
    norm_img_repeat = tf.reshape(tf.tile(x, [n_repeat]), (n_repeat, img_dim))
    oddball_inputs = tf.concat([norm_img_repeat, tf.expand_dims(y, axis=0)], 0)

    n_col = len(oddball_inputs)
    odd_fig, odd_axs = plt.subplots(nrows=1, ncols=n_col, sharex='all', sharey='all')
    for input_idx, ax in enumerate(odd_axs.flatten()):
        ax.imshow(oddball_inputs[input_idx].reshape(32, 32), cmap='gray')
        ax.axis('off')
    odd_fig.tight_layout()

    return oddball_inputs, odd_fig


def test_reconstruction(model, layer, input_vector, n_samples):

    if n_samples % 16 != 0:
        raise ValueError('Pick a number divisible by 16.')
    else:
        pass

    sample_idcs = np.random.choice(np.arange(len(input_vector)), n_samples, replace=False)

    # input_vector = (n_sample, n_pixel) -> test_samples = (n_pixel, n_sample)
    test_samples = tf.transpose(
        tf.gather(
            input_vector, sample_idcs
        )
    )

    if test_samples.shape == 1:
        test_samples = tf.reshape(
            test_samples, (test_samples.shape[0], 1)
        )
    else:
        pass

    # simulate
    run_test(model=model, spb=test_samples.shape[1], inputs_x=test_samples)

    # input_fr = pred_fr = (n_pixel, n_sample)
    input_fr = model.network[f'layer_{layer}']['rep_e']
    pred_fr = tf.transpose(model.weights[f'{layer}{layer + 1}']) @ model.network[f'layer_{layer + 1}']['rep_r']

    # plot reconstruction
    dim_x, dim_y, input_pad = generate_reconstruction_pad(img_mat=input_fr)
    dim_x, dim_y, pred_pad = generate_reconstruction_pad(img_mat=pred_fr)

    # figsize = (horizontal, vertical)
    fig_y = dim_x // tf.sqrt(input_fr.shape[0]).astype(int)
    fig_x = dim_y // tf.sqrt(input_fr.shape[0]).astype(int)
    recon_fig, recon_axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all', figsize=(fig_x, fig_y))
    recon_axs[0].imshow(input_pad, cmap='gray')
    recon_axs[0].set_title('input')
    recon_axs[1].imshow(pred_pad, cmap='gray')
    recon_axs[1].set_title('prediction')
    for ax in recon_axs.flatten():
        ax.axis('off')

    recon_fig.suptitle(f'reconstruction at layer {layer}')
    recon_fig.tight_layout()

    return recon_fig


def run_test(model, spb, inputs_x):
    # initialize
    model.initialize_network(batch_size=spb)

    # isi
    for t_step in range(isi_steps):
        model.compute(inputs=np.zeros(inputs_x.shape))
    # stimulus presentation
    for t_step in range(time_steps):
        model.compute(inputs=inputs_x)


class network(object):

    tau_exc = 20e-3
    tau_inh = 20e-3

    bg_exc = 0  # 4e-3
    bg_inh = 0  # 3e-3

    resolution = 1e-3

    def __init__(self, neurons_per_layer, pretrained_weights=None):

        self.net_size = neurons_per_layer

        self.run_error = {}
        self.weights = {}
        self.network = {}

        self.initialize_weights(weights=pretrained_weights)

    def initialize_weights(self, weights):

        # inter-layer syn str
        for pc_i, layer in enumerate(self.net_size):

            # random intialization
            if weights == None:
                if pc_i > 0:
                    self.weights[f'{pc_i - 1}{pc_i}'] = np.random.uniform(
                        size=(self.net_size[pc_i], self.net_size[pc_i - 1])
                    ) / np.sqrt(self.net_size[pc_i - 1])
                    # self.weights[f'{i - 1}{i}'] = sprandn(
                    #     m=self.net_size[i],
                    #     n=self.net_size[i - 1],
                    #     density=0.2,
                    #     data_rvs=norm(loc=1 / np.sqrt(self.net_size[i - 1]), scale=0.1).rvs
                    # ).A
                # import the pretrained weights
            else:
                self.weights[f'{pc_i - 1}{pc_i}'] = weights[f'{pc_i - 1}{pc_i}']

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
                self.network[f'layer_{i}'][f'rep_{rep}'] = np.zeros((self.net_size[i], self.bat_size))  ## T_steps))
            self.network[f'layer_{i}']['rep_r'] = sprandn(
                m=self.net_size[i],
                n=self.bat_size,
                density=0.2,
                data_rvs=norm(loc=15, scale=5).rvs,
                random_state=np.random.seed(1234)
            ).A  # .flatten()
            for pe in pe_types:
                for neuron in neuron_types:
                    self.network[f'layer_{i}'][f'{pe}_{neuron}'] = np.zeros(
                        (self.net_size[i], self.bat_size))  # , T_steps))

            if i < len(self.net_size) - 1:
                self.run_error[f'layer_{i}'] = {
                    'ppe': [],
                    'npe': []
                }

    def initialize_error(self):

        self.errors = {
            layer: {
                'ppe': [],
                'npe': []
            }
            for layer, neurons in self.network.items() if layer != f'layer_{len(self.network) - 1}'
        }

    def add_err(self):
        # mean across samples and neurons
        for i in range(len(self.net_size) - 1):
            self.run_error[f'layer_{i}']['ppe'].append(np.mean(self.errors[f'layer_{i}']['ppe']))
            self.run_error[f'layer_{i}']['npe'].append(np.mean(self.errors[f'layer_{i}']['npe']))

    def dr(self, r, inputs, func='relu', ei_type='inh'):

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

        return r + (-r + fx + bg) * (self.resolution / tau)

    def compute(self, inputs, bu_rate=1.0, td_rate=1.0):

        self.inputs = inputs
        # ['layer_0', 'layer_1', 'layer_2']
        # ['rep_e', 'rep_i', 'rep_r', 'ppe_pyr', 'ppe_pv', 'ppe_sst', 'npe_pyr', 'npe_pv', 'npe_sst']
        # (n_neuron, size_batch)

        for layer_i, (layer, neurons) in enumerate(self.network.items()):

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
                    w_ppe = self.weights[f'{layer_i - 1}{layer_i}'] @ self.network[f'layer_{layer_i - 1}']['ppe_pyr']
                    w_npe = self.weights[f'{layer_i - 1}{layer_i}'] @ self.network[f'layer_{layer_i - 1}']['npe_pyr']

                    # rep_e
                    self.network[layer]['rep_e'] = self.dr(
                        r=self.network[layer]['rep_e'],
                        inputs=td_rate * self.network[layer]['npe_pyr'] +
                               bu_rate * w_ppe -
                               self.network[layer]['rep_i'],
                        ei_type='exc'
                    )
                    # rep_i
                    self.network[layer]['rep_i'] = self.dr(
                        r=self.network[layer]['rep_i'],
                        inputs=td_rate * self.network[layer]['ppe_pyr'] +
                               bu_rate * w_npe
                    )
                    # rep_r
                    # n_neurons = self.network[layer]['rep_r'].shape[0]
                    # w_val = 0.0
                    # later_w = np.ones((n_neurons, n_neurons)) * -w_val
                    # np.fill_diagonal(later_w, w_val)
                    self.network[layer]['rep_r'] = self.dr(
                        r=self.network[layer]['rep_r'],
                        inputs=self.network[layer]['rep_e'],  # + later_w @ self.network[layer]['rep_r'],
                        ei_type='exc'
                    )

                prediction = self.weights[f'{layer_i}{layer_i + 1}'].T @ self.network[f'layer_{layer_i + 1}']['rep_r']

                # ppe_pyr
                self.network[layer]['ppe_pyr'] = self.dr(
                    r=self.network[layer]['ppe_pyr'],
                    inputs=self.network[layer]['rep_e'] -
                           self.network[layer]['ppe_pv'] -
                           self.network[layer]['ppe_sst'] +
                           self.network[layer]['ppe_pyr'],
                    ei_type='exc'
                )
                # ppe_pv
                self.network[layer]['ppe_pv'] = self.dr(
                    r=self.network[layer]['ppe_pv'],
                    inputs=self.network[layer]['ppe_pyr'] +
                           prediction
                )
                # ppe_sst
                self.network[layer]['ppe_sst'] = self.dr(
                    r=self.network[layer]['ppe_sst'],
                    inputs=self.network[layer]['ppe_pyr'] -
                           self.network[layer]['ppe_vip'] +
                           prediction
                )
                # ppe_vip
                self.network[layer]['ppe_vip'] = self.dr(
                    r=self.network[layer]['ppe_vip'],
                    inputs=self.network[layer]['ppe_pyr'] +
                           prediction
                )

                # npe_pyr
                self.network[layer]['npe_pyr'] = self.dr(
                    r=self.network[layer]['npe_pyr'],
                    inputs=self.network[layer]['rep_e'] +
                           prediction +
                           self.network[layer]['npe_pyr'] -
                           self.network[layer]['npe_pv'] -
                           self.network[layer]['npe_sst'],
                    ei_type='exc'
                )
                # npe_pv
                self.network[layer]['npe_pv'] = self.dr(
                    r=self.network[layer]['npe_pv'],
                    inputs=self.network[layer]['rep_e'] +
                           self.network[layer]['npe_pyr']
                )
                # npe_sst
                self.network[layer]['npe_sst'] = self.dr(
                    r=self.network[layer]['npe_sst'],
                    inputs=self.network[layer]['rep_e'] +
                           prediction +
                           self.network[layer]['npe_pyr'] -
                           self.network[layer]['npe_vip']
                )
                # npe_vip
                self.network[layer]['npe_vip'] = self.dr(
                    r=self.network[layer]['npe_vip'],
                    inputs=prediction +
                           self.network[layer]['npe_pyr']
                )

            # top layer
            else:
                ppe = self.weights[f'{layer_i - 1}{layer_i}'] @ self.network[f'layer_{layer_i - 1}']['ppe_pyr']
                npe = self.weights[f'{layer_i - 1}{layer_i}'] @ self.network[f'layer_{layer_i - 1}']['npe_pyr']

                # rep_e
                self.network[layer]['rep_e'] = self.dr(
                    r=self.network[layer]['rep_e'],
                    inputs=ppe,
                    ei_type='exc'
                )
                # rep_i
                self.network[layer]['rep_i'] = self.dr(
                    r=self.network[layer]['rep_i'],
                    inputs=npe
                )
                # rep_r
                self.network[layer]['rep_r'] = self.dr(
                    r=self.network[layer]['rep_r'],
                    inputs=self.network[layer]['rep_e'],
                    ei_type='exc'
                )

        self.get_error()

    def get_error(self):

        for layer, _ in self.errors.items():
            self.errors[layer]['ppe'].append(
                # average over neurons
                np.mean(
                    # average over batches
                    np.mean(self.network[layer]['ppe_pyr'] ** 2, axis=1),
                )
            )
            self.errors[layer]['npe'].append(
                # average over neurons
                np.mean(
                    # average over batches
                    np.mean(self.network[layer]['npe_pyr'] ** 2, axis=1)
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
        if len(self.network) - 1 == 1:
            err_fig, err_axs = plt.subplots(3, 1)
            for layer, pes in self.errors.items():
                err_fig[0].plot(pes['ppe'], c='r')
                err_fig[0].set_title('pPE')
                err_fig[1].plot(pes['npe'], c='b', label='nPE')
                err_fig[1].set_title('nPE')
                err_fig[2].plot(np.add(pes['ppe'], pes['npe']), c='black')
                err_fig[2].set_title('PE')

        else:
            err_fig, err_axs = plt.subplots(nrows=len(self.network) - 1, ncols=3, sharex='all')  # , sharey='col')
            # err_axs = err_axs.flatten()
            for i, (layer, pes) in enumerate(self.errors.items()):

                err_axs[i, 0].plot(pes['ppe'], c='r', label='pPE')
                err_axs[i, 1].plot(pes['npe'], c='b', label='nPE')
                err_axs[i, 2].plot([sum(x) for x in zip(pes['ppe'], pes['npe'])], c='black', label='PE')

                # label: layer
                err_axs[i, 1].set_title(f'{layer}')
                # label: x-axis
                err_axs[i, 0].set_ylabel('MSE')

                # legend
                if i == 0:
                    err_fig.legend(loc='upper right')
                elif i == len(self.errors) - 1:
                    err_axs[i, 1].set_xlabel('time (ms)')

        err_fig.suptitle(f'MSE at epoch #{curr_ep}/{n_ep}')
        err_fig.tight_layout()

        return err_fig

    def plot_run_error(self):

        re_fig, re_axs = plt.subplots(nrows=3, ncols=len(self.net_size) - 1, sharex='all')
        for i in range(len(self.net_size) - 1):
            re_axs[0, i].plot(self.run_error[f'layer_{i}']['ppe'], c='r')
            re_axs[0, i].set_title('pPE')
            re_axs[1, i].plot(self.run_error[f'layer_{i}']['npe'], c='b')
            re_axs[1, i].set_title('nPE')
            re_axs[1, i].set_ylabel('MSE')
            re_axs[2, i].plot(
                [sum(x) for x in zip(self.run_error[f'layer_{i}']['ppe'], self.run_error[f'layer_{i}']['npe'])],
                c='black')
            re_axs[2, i].set_title('PE')
            re_axs[2, i].set_xlabel('training epoch #')
        re_fig.suptitle('MSE over training epochs')
        re_fig.tight_layout()

        return re_fig

    def learn(self, lr=1e-2, alpha=1e-4):

        for i, (layers, weight) in enumerate(self.weights.items()):

            l1_reg = alpha * np.maximum(0.0, weight)

            dwp = -lr * np.einsum(
                # 'ij, kj -> ki',
                'ik, jk -> ji',
                self.network[f'layer_{i}']['ppe_pyr'],
                self.network[f'layer_{i + 1}']['rep_r']
            ) / self.bat_size
            dwn = -lr * np.einsum(
                # 'ij, kj -> ki',
                'ik, jk -> ji',
                self.network[f'layer_{i}']['npe_pyr'],
                self.network[f'layer_{i + 1}']['rep_r']
            ) / self.bat_size

            self.weights[layers] = np.maximum(0.0, weight - dwp + dwn - l1_reg)
            # self.weights[layers] = weight - dwp + dwn

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
            if idx == None:
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
                if idx == None:
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
                if idx == None:
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
                if idx == None:
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

    def batch_plots(self, curr_ep, n_epoch):

        # input, pred, err+, err-
        ncol = 4
        # layers
        nrow = len(self.network) - 1

        # plot
        # int_fig = plt.figure(figsize=(ncol * 3, nrow * 3))
        # grid_plot = plt.GridSpec(nrows=nrow, ncols=ncol)
        batch_fig, batch_ax = plt.subplots(
            nrows=nrow, ncols=ncol, sharex='all', sharey='all',
            figsize=(ncol * 5, nrow * 5)
        )
        batch_ax = batch_ax.flatten()
        for i, (layer, _) in enumerate(self.network.items()):

            if i < nrow:
                # input to layer
                rand_idx = np.random.choice(np.arange(self.network[layer]['rep_e'].shape[1]), 16, replace=False)
                layer_input = self.network[layer]['rep_e'][:, rand_idx]
                layer_pred = (self.weights[f'{i}{i + 1}'].T @ self.network[f'layer_{i + 1}']['rep_r'])[:, rand_idx]
                layer_ppe = self.network[layer]['ppe_pyr'][:, rand_idx]
                layer_npe = self.network[layer]['npe_pyr'][:, rand_idx]

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

        _, _, image = generate_reconstruction_pad(
            img_mat=inputs, nx=4
        )
        input_subplot = ax.imshow(image, cmap=colors)
        fig.colorbar(input_subplot, ax=ax, shrink=0.2)
        ax.axis('off')

        # return ax


if __name__ == "__main__":

    # simulation params
    sim_time = 2.0
    isi_time = 0.1
    dt = 1e-3
    time_steps = int(sim_time / dt)
    isi_steps = int(isi_time / dt)

    # predictive coding
    n_class = 3
    n_sample = 16
    n_img = n_class * n_sample
    # input_x = (n_class * n_sample, n_pixel)
    input_x, input_y, input_fig = generate_input(
        input_type='fmnist', num_class=n_class, num_sample=n_sample, max_fr=30, shuffle=True)
    input_fig.show()

    # # oddball paradigm
    # input_x, odd_fig = oddball_input(input_x[0], input_x[1], 3)
    # odd_fig.show()
    # n_img = len(input_x)

    #
    input_size = np.shape(input_x)[1]
    bat_size = 16
    n_batch = n_class * n_sample // bat_size
    net_size = [input_size, 400, 100]
    net = network(neurons_per_layer=net_size)
    net.initialize_network(batch_size=bat_size)

    w_fig, w_axs = plt.subplots(nrows=2, ncols=len(net_size) - 1)
    for i, (w_idx, w_mat) in enumerate(net.weights.items()):
        w_axs[0, i].imshow(w_mat, cmap='hot')
        w_axs[0, i].set_title(f'{w_idx} before')
        w_axs[0, i].axis('off')

    n_epoch = 5
    # plot interval (ms)
    plot_interval = 1
    # weight update interval (ms)
    learn_interval = 250

    for epoch_i in trange(n_epoch, desc='epoch'):

        # for img_i in range(n_img):
        for batch_i in range(n_batch):

            # reset error
            net.initialize_error()

            # stim presentation
            for t in range(time_steps):  # , desc=f'stimulus {img_i + 1}/{n_img}'):

                # simulate batch
                curr_batch = input_x[batch_i * bat_size: (batch_i + 1) * bat_size].T
                net.compute(inputs=curr_batch, bu_rate=0.1, td_rate=0.01)
                # # simulate serial / oddball
                # curr_img = input_x.T  # input_x[img_i].T
                # net.compute(inputs=curr_img, bu_rate=0.1, td_rate=0.01)

                # weight update
                if (t + 1) % 100:
                    net.learn(lr=1e-4, alpha=1e-5)
                else:
                    pass

            # add run error
            net.add_err()


            # plot intermediate results
            # if ((epoch_i + 1) % plot_interval == 0):
            if ((epoch_i + 1) % plot_interval == 0) and ((batch_i + 1) == n_batch):

                err_fig = net.plot_error(epoch_i + 1, n_epoch)
                err_fig.show()

                run_err_fig = net.plot_run_error()
                run_err_fig.show()
                # int_fig = net.plots(epoch_i + 1, n_epoch)
                int_fig = net.batch_plots(epoch_i + 1, n_epoch)
                int_fig.show()

                # # test reconstruction
                # recon_fig = test_reconstruction(model=net, layer=0, input_vector=input_x, n_samples=16)
                # recon_fig.show()

            # isi
            for t in range(isi_steps):

                # simulate batch
                net.compute(inputs=np.zeros(curr_batch.shape))
                # # simulate serial / oddball
                # net.compute(inputs=np.zeros(curr_img.shape))
    #
    # for i, (w_idx, w_mat) in enumerate(net.weights.items()):
    #     w_axs[1, i].imshow(w_mat, cmap='hot')
    #     w_axs[1, i].set_title(f'{w_idx} after')
    #     w_axs[1, i].axis('off')
    # w_fig.show()
    #
    # # test reconstruction
    # recon_fig = test_reconstruction(model=net, layer=0, input_vector=input_x, n_samples=64)
    # recon_fig.show()
    #
    # # positive error increases. Why?
    # # why does mse of pe fluctuate?
    # # fr decreases as you move across the hierarchy, why?
    #
    # # occlusion test
    # rand_n = np.random.choice(np.arange(30), 1, replace=False)
    # aa = input_x[rand_n].reshape(28, 28)
    # occlusion_padding = np.ones(aa.shape)
    # occlusion_padding[14:20, 14:20] = 0
    #
    # noise_padding = np.random.normal(0, 3, (28, 28))
    #
    # # occluded input
    # plt.imshow(aa * occlusion_padding, cmap='gray')
    # plt.title('occluded sample')
    # plt.show()
    #
    # # noisy input
    # plt.imshow(aa + noise_padding, cmap='gray')
    # plt.title('noisy sample')
    # plt.show()
    #
    # occluded_input = (aa * occlusion_padding).reshape(784, 1)
    # noisy_input = (aa + noise_padding).reshape(784, 1)
    #
    # # initialize
    # net.initialize_network(batch_size=occluded_input.shape[1])
    #
    # for epoch_i in range(1):
    #
    #     for batch_i in range(1):
    #
    #         # isi
    #         for t in range(isi_steps):
    #             net.compute(inputs=np.zeros(occluded_input.shape), get_error=False)
    #         # stimulus presentation
    #         for t in range(time_steps):
    #             net.compute(inputs=occluded_input, get_error=True)
    #
    #         int_fig = net.plots(1, 1)
    #         int_fig.show()
    #         err_fig = net.plot_error(1, 1)
    #         err_fig.show()
    #
    #         # isi
    #         for t in range(isi_steps):
    #             net.compute(inputs=np.zeros(noisy_input.shape), get_error=False)
    #         # stimulus presentation
    #         for t in range(time_steps):
    #             net.compute(noisy_input, get_error=True)
    #
    #         int_fig = net.plots(1, 1)
    #         int_fig.show()
    #         err_fig = net.plot_error(1, 1)
    #         err_fig.show()

    # # single neuron example: the time course rep circuit
    # neuron_exc = np.zeros(time_steps * 2)
    # neuron_inh = np.zeros(time_steps * 2)
    # neuron_rel = np.zeros(time_steps * 2)
    # tau_exc = 10e-3
    # tau_inh = 2e-3
    # input_exc = 5
    # input_inh = 2
    #
    # # r + (-r + fx + bg) * (self.resolution / tau)
    # for t in range(time_steps * 2 - 1):
    #     if t < time_steps - 1:
    #         neuron_exc[t + 1] = neuron_exc[t] + (1e-3 / tau_exc) * (-neuron_exc[t] + ReLu(input_exc - neuron_inh[t]))
    #         neuron_inh[t + 1] = neuron_inh[t] + (1e-3 / tau_inh) * (-neuron_inh[t] + ReLu(input_inh))
    #         neuron_rel[t + 1] = neuron_rel[t] + (1e-3 / tau_exc) * (-neuron_rel[t] + ReLu(neuron_exc[t]))
    #     else:
    #         neuron_exc[t + 1] = neuron_exc[t] + (1e-3 / tau_exc) * (-neuron_exc[t] + ReLu(0 - neuron_inh[t]))
    #         neuron_inh[t + 1] = neuron_inh[t] + (1e-3 / tau_inh) * (-neuron_inh[t] + ReLu(0))
    #         neuron_rel[t + 1] = neuron_rel[t] + (1e-3 / tau_exc) * (-neuron_rel[t] + ReLu(neuron_exc[t]))
    # sn_fig, sn_axs = plt.subplots(nrows=3, ncols=1, sharex='all', sharey='all')
    # sn_axs[0].plot(neuron_exc)
    # sn_axs[0].set_title('exc')
    # sn_axs[1].plot(neuron_inh)
    # sn_axs[1].set_title('inh')
    # sn_axs[1].set_ylabel('firing rate (Hz)')
    # sn_axs[2].plot(neuron_rel)
    # sn_axs[2].set_title('rel')
    # sn_axs[2].set_xlabel('time (ms)')
    # sn_fig.tight_layout()
    # sn_fig.show()

    # # normalized mse
    # layer_n = 0
    #
    # ss_fig, ss_axs = plt.subplots(nrows=2, ncols=1, sharex='all')
    # ss_axs[0].plot(net.errors[f'layer_{layer_n}}']['ppe'], c='r')
    # ss_axs[0].set_ylabel('MSE')
    # ss_axs[0].set_title('pPE')
    # ss_axs[1].plot(net.errors[f'layer_{layer_n}']['npe'], c='b')
    # ss_axs[1].set_xlabel('time (ms)')
    # ss_axs[1].set_ylabel('MSE')
    # ss_axs[1].set_title('nPE')
    # ss_fig.tight_layout()
    # ss_fig.show()
    #
    # npe = net.errors[f'layer_{layer_n}']['npe'] / np.max(net.errors[f'layer_{layer_n}']['npe'])
    # ppe = net.errors[f'layer_{layer_n}']['ppe'] / np.max(net.errors[f'layer_{layer_n}']['ppe'])
    # plt.plot(ppe, c='r', label='ppe')
    # plt.plot(npe, c='b', label='npe')
    # plt.ylabel('MSE')
    # plt.xlabel('time (ms)')
    # plt.title('normalized MSE at layer 0')
    # plt.legend()
    # plt.show()
