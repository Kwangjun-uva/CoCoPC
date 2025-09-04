import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import datetime

from tqdm import trange

from noise import noise_generator
from tools import f, pickle_save

import pickle


class neuron_group(object):

    def __init__(self, num_neurons, batch_size, dt, tau, bg, func='ReLu'):

        # number of neurons
        self.num_neurons = num_neurons
        # batch size
        self.batch_size = batch_size

        # simulation resolution
        self.dt = dt
        # membrane time constant
        self.tau = tau
        # background noise stddev
        self.bg = bg

        # set activation function
        self.activation_fn = func

        # initialize firing rates to zero
        self.r = tf.zeros(shape=(num_neurons, batch_size), dtype=tf.float32)

    def update_r(self, syn_input, silence=False):

        # shut down activity if silence target
        if silence:
            self.r *= 0.0

        else:
            # activation function
            fx = f(x=syn_input, func_type=self.activation_fn)
            # noise
            ### FIX ### No need to add noise, unless noise is simulated
            noise = tf.random.normal(shape=fx.shape, mean=0.0, stddev=self.bg, dtype=tf.float32)
            # dr
            dr = (-self.r + fx + noise) * (self.dt / self.tau)
            # update r
            self.r += dr


class network(object):

    def __init__(
            self,
            simParams,
            pretrained_weights=None,
            profopol_str=0.0,
            bg_exc=0.0, bg_inh=0.0,
            jitter_lvl=0.0, jitter_type='constant',
    ):

        # number of neurons per layer
        self.net_size = simParams['net_size']
        # number of layers
        self.n_layers = len(self.net_size)
        # batch size: not defined before feeding training data
        self.batch_size = None

        # plastic synaptic weights (between layers)
        self.weights = {}
        # initialize weights
        self.initialize_weights(weights=pretrained_weights)

        # profopol-induced loss of consciousness
        self.profopol_inh = 1.0 - profopol_str

        # time resolution
        self.dt = simParams['dt']
        # bottom-up inference rate
        self.bu_rate = simParams['bu_rate']
        # top-down inference rate
        self.td_rate = simParams['td_rate']
        # membrane time constant for excitatory cells
        self.tau_exc = simParams['tau_exc']
        # membrane time constant for inhibitory cells
        self.tau_inh = simParams['tau_inh']

        # background noise for excitatory cells
        self.bg_exc = bg_exc
        # background noise for inhibitory cells
        self.bg_inh = bg_inh
        # jitter (internal noise)
        self.jitter_lvl = jitter_lvl
        self.jitter_type = jitter_type

        # input
        self.inputs = None

        # mean prediction error per iteration
        self.err_per_iter = {}
        # non-plastic synaptic weights (within layers; among neuron types)
        self.np_weights = {f'layer_{i}': {} for i in range(self.n_layers)}

        # neuron groups (layer > neuron type)
        self.neuron_groups = {}

        # target neuron groups for optogentic silencing experiment
        self.silence_target = []

    def generate_nonplastic_weights(self, target_shape, layer='curr'):

        ### FIX ### move this to noise.py

        # layer 0
        if layer == 'curr':
            # 'pyr, 'pyr': pyramidal to pyramidal cell connections(within L2/3 PE+/- microcircuit)
            # 'pyr, sst': pyramidal to SST cell connections(within L2/3 PE+/- microcircuit)
            # 'pyr, pv': pyramidal to PV cell connections(within L2/3 PE+/- microcircuit)
            # 'pyr, vip': pyramidal to VIP cell connections(within L2/3 PE+/- microcircuit)
            # 'sst, pyr': SST to pyramidal cell connections(within L2/3 PE+/- microcircuit)
            # 'pv, pyr': PV to pyramidal cell connections(within L2/3 PE+/- microcircuit)
            # 'vip, sst': VIP to pyramidal cell connections(within L2/3 PE+/- microcircuit)
            # 'e, pyr+': L4 pyramidal (input) to L2/3 pyramidal cell connections (PE+ microcircuit)
            # 'e, pyr-': L4 pyramidal (input) to L2/3 pyramidal cell connections (PE- microcircuit)
            # 'e, pv-': L4 pyramidal (input) to L2/3 PV cell connections (PE- microcircuit)
            # 'e, sst-': L4 pyramidal (input) to L2/3 SST cell connections (PE- microcircuit)
            np_weight_keys = [
                'pyr, pyr', 'pyr, sst', 'pyr, pv', 'pyr, vip',
                'sst, pyr', 'pv, pyr', 'vip, sst',
                'e, pyr+', 'e, pyr-', 'e, pv-', 'e, sst-'
            ]
        # layer 1
        elif layer == 'above':
            # 'e, r': L4 pyramidal to L5/6 pyramidal cell connections (Rep microcircuit)
            # 'i, e': L4 PV to L4 pyramidal cell connections (Rep microcircuit)
            # np_weight_keys = ['e, r', 'i, e']
            np_weight_keys = ['e, r', 'e, i', 'i, e', 'r, r']
        else:
            raise ValueError('Choose either curr or above.')

        nonplastic_weights = {
            # set lower bound to 0 to ensure that weights are non-negative
            np_weight: tf.maximum(
                x= 1 + noise_generator(
                    noise_type=self.jitter_type,
                    noise_lvl=self.jitter_lvl,
                    target_shape=(target_shape, 1))[0],
                y=0.0
            )
            for np_weight in np_weight_keys
        }

        return nonplastic_weights

    def initialize_weights(self, weights):

        for layer_i, num_neurons in enumerate(self.net_size[:-1]):

            # random initialization
            if weights is None:

                # synaptic strengths between layers (i.e. 0 and 1)
                self.weights[f'{layer_i}{layer_i + 1}'] = tf.random.uniform(
                    shape=(self.net_size[layer_i + 1], num_neurons), minval=0.0, maxval=1.0,
                ) / np.sqrt(self.net_size[layer_i + 1])
            # use pretrained weights
            else:
                self.weights = weights

    def initialize_network(self, batch_size):

        ## FIX ##
        # set batch size
        self.batch_size = batch_size

        # neuron group keys
        rep_neurons = ['e', 'i', 'r']
        neuron_types = ['pyr', 'pv', 'sst', 'vip']
        # PE microcircuit type keys
        pe_types = ['ppe', 'npe']

        # make this layerwise: for example, 3-layer has layer 0, 1, and 2
        for layer_i, num_neurons in enumerate(self.net_size):

            # add layer to the network
            self.neuron_groups[f'layer_{layer_i}'] = {}

            # every layer has Rep microcircuits
            # populate the network with rep neurons
            for rep in rep_neurons:

                rep_tau = self.tau_inh if rep == 'i' else self.tau_exc
                rep_bg = self.bg_inh if rep == 'i' else self.bg_exc
                self.neuron_groups[f'layer_{layer_i}'][f'rep_{rep}'] = neuron_group(
                    num_neurons=num_neurons, batch_size=self.batch_size,
                    dt=self.resolution, tau=rep_tau, bg=rep_bg, func='relu'
                )

            # every layer except the top layer has PE+/- microcircuits
            if layer_i < self.n_layers - 1:
                # populate the network with PE neurons
                for pe in pe_types:
                    for neuron in neuron_types:
                        pe_tau = self.tau_exc if neuron == 'pyr' else self.tau_inh
                        pe_bg = self.bg_exc if neuron == 'pyr' else self.bg_inh
                        self.neuron_groups[f'layer_{layer_i}'][f'{pe}_{neuron}'] = neuron_group(
                    num_neurons=num_neurons, batch_size=self.batch_size,
                    dt=self.resolution, tau=pe_tau, bg=pe_bg, func='relu'
                )
                # add prediction error saver per iteration
                self.err_per_iter[f'layer_{layer_i}'] = {
                    'ppe_pyr': [],
                    'npe_pyr': []
                }
        del self.neuron_groups['layer_0']['rep_i']
        del self.neuron_groups['layer_0']['rep_r']

        ### FIX ### add PE circuit connectivity matrices

    def initialize_error(self):

        ## FIX: what is the difference between self.errors and self.run_error? ###
        self.errors = {f'layer_{layer_i}': {} for layer_i in self.n_layers if layer_i < self.n_layers}

    def record_mean_PE(self):

        # Note: top layer does not have PE microcircuits
        for i in range(self.n_layers - 1):
            # mean PE+ across samples and neurons
            self.err_per_iter[f'layer_{i}']['ppe_pyr'].append(np.mean(self.errors[f'layer_{i}']['ppe_pyr']))
            # mean PE- across samples and neurons
            self.err_per_iter[f'layer_{i}']['npe_pyr'].append(np.mean(self.errors[f'layer_{i}']['npe_pyr']))

    def silencer_function(self, job, target_layer, target_neuron):

        """
        :param job: str. select from ['activate', 'deactivate', 'reset']
        :param target_layer: str. 'layer_N'. e.g., 'layer_0'
        :param target_neuron: str. neuron type. e.g., 'pv'

        activate: set target neuron group for optogenetic silencing experiment
        deactivate: remove target neuron group for optogenetic silencing experiment
        reset: remove all target neuron groups for optogenetic silencing experiment
        """

        if job == 'activate':
            self.silence_target.append((target_layer, target_neuron))
        elif job == 'deactivate':
            self.silence_target.remove((target_layer, target_neuron))
        elif job == 'reset':
            self.silence_target = []
        else:
            raise ValueError('Job needs to be activate, deactivate, or reset.')

    def update_activity(self, inputs, record='error'):

        ### FIX ### is this necessary?
        self.inputs = inputs

        for layer_i, (layer, layer_data) in enumerate(self.neuron_groups.items()):

            # update rep circuit
            rep_inputs = self._get_rep_circuit_input(layer_num=layer_i)
            rep_keys = [ng for ng in layer_data if 'rep' in ng]
            for rep_i, rep_neuron_group in enumerate(rep_keys):
                self.neuron_groups[layer][rep_neuron_group].r.update_r(
                    syn_input=rep_inputs[rep_i], silence=(layer, rep_neuron_group) in self.silence_target)

            # update PE circuits
            pe_inputs = self._get_pe_input(
                layer_num=layer_i, conn_idcs=(0, 9, 54), inp_pat=inp_patterns, syn_pat=syn_patterns
            )
            pe_keys = [ng for ng in layer_data if 'pe' in ng]
            for pe_i, pe_neuron_group in enumerate(pe_keys):
                self.neuron_groups[layer][pe_neuron_group].r.update_r(
                    syn_input=pe_inputs[pe_i], silence=(layer, pe_neuron_group) in self.silence_target)

        if record is not None:
            self.record(record)

    def _get_rep_circuit_input(self, layer_num):

        ### FIX ### need to add top-down errors to add more layers

        # feed external inputs to the bottom layer
        if layer_num == 0:
            return self.inputs

        else:

            # connection matrix (3, 4)
            # rows correspond to post-synaptic neurons: L4 pyr (rep_e), L4 PV (rep_i), and L5/6 pyr (rep_r)
            # columns correspond to weighted activities from pre-synaptic neurons: BU PE+, BU PE-, rep_e, and rep_i
            #
            # L4 pyr (rep_e) receives excitatory bottom-up PE+ from L2/3 pyr and lateral inhibition from L4 PV (rep_i)
            # L4 PV (rep_i) receives excitatory bottom-up PE- from L2/3 pyr and lateral excitation from L4 pyr (rep_e)
            # L5/6 pyr (rep_r) receives excitatory signal from L4 pyr (rep_e)

            conn_mat = tf.constant(
                value=[
                    [1, 0, 0, -1],
                    [0, 1, 1, 0],
                    [0, 0, 1, 0]],
                dtype=tf.float32
            )

            # bottom-up PE+ from L2/3 pyr in l-1 to L4 pyr in l
            bu_ppe = self._get_bu_errors(source_layer=layer_num - 1, target_layer=layer_num, sign='pos')
            # bottom-up PE- from L2/3 pyr in l-1 to L4 PV in l
            bu_npe = self._get_bu_errors(source_layer=layer_num - 1, target_layer=layer_num, sign='neg')

            synaptic_inputs = tf.stack([
                    bu_ppe,
                    bu_npe,
                    self.neuron_groups[layer_num]['rep_e'].r,
                    self.neuron_groups[layer_num]['rep_i'].r
                ]
            )

            return tf.einsum('ij, jkl -> ikl', conn_mat, synaptic_inputs)


    def _get_pe_input(self, layer_num, conn_idcs, inp_pat, syn_pat):

        # firing rates of 4 neuron types in PE+ and PE- circuit (total 8; pe_frs = (8, n_neuron, batch_size))
        pe_frs = tf.stack(
            values=[data.r.shape for group, data in self.neuron_groups['layer_0'].items() if 'pe' in group],
            axis=0
        )

        # BU and TD inputs
        # bottom layer
        if layer_num == 0:
            bu_input = self.inputs
            td_input = self._get_predictions(source_layer=layer_num + 1, target_layer=layer_num)
        # top layer
        elif layer_num == self.n_layers - 1:
            bu_input = self.neuron_groups[f'layer_{layer_num}']['rep_e'].r
            td_input = tf.zeros(shape=(self.net_size[layer_num], self.batch_size), dtype=tf.float32)
        # middle layers
        else:  # This covers all intermediate layers (0 < layer_num < self.n_layers - 1)
            bu_input = self.neuron_groups[f'layer_{layer_num - 1}']['rep_e'].r
            td_input = self._get_predictions(source_layer=layer_num + 1, target_layer=layer_num)

        # (2, n_neuron, batch_size)
        bu_td_input = tf.stack(values=[bu_input, td_input], axis=0)

        # homogenous connectivity across circuits
        if isinstance(conn_idcs, tuple):

            # A connection matrix 4 neuron types (pyr, PV, SST, VIP) in PE+ circuit + 4 in PE- circuit (8, 8)
            # Rows correspond to post-synaptic neurons
            # Columns correspond to pre-synpatic neurons
            # Note that there are no lateral connections between PE+ and PE- circuits (0 in upper right and lower left)
            ### FIX ### this can be constructed prior to this function

            pe_conn_mat = construct_conn_mat(conn_mat=syn_pat[conn_idcs[0]], conn_type='within')
            within_circuit_input = tf.einsum('ij, jkl -> ikl', pe_conn_mat, pe_frs)

            bu_td_conn_mat = construct_conn_mat(conn_mat=inp_pat[list(conn_idcs[1:])], conn_type='bu_td')
            bu_td_inputs = tf.einsum('ij, jkl -> ikl', bu_td_conn_mat, bu_td_input)

            return bu_td_inputs + within_circuit_input

        # heterogeneous connectivity across circuits
        elif isinstance(conn_idcs, np.ndarray):

            # conn_idcs = (3, n_neurons)
            # Row 1: conn mat within PE circuits
            # Row 2: BU/TD input to PE+
            # Row 3: BU/TD input to PE-

            sum_syn_input = tf.zeros(
                shape=(8, self.net_size[layer_num], self.batch_size),
                dtype=tf.float32
            )

            # pe_frs = (8, n_neuron, batch_size)
            # bu_td_input = (2, n_neuron, batch_size)
            for i in self.net_size[layer_num]:
                within_circuit_inputs = (
                        construct_conn_mat(conn_mat=syn_pat[conn_idcs[0, i]], conn_type='syn')
                        @ pe_frs[:, i, :]
                )
                bu_td_inputs = (
                        construct_conn_mat(conn_mat=[conn_idcs[1:, i]], conn_type='bu_td')
                        @ bu_td_input[:, i, :]
                )
                sum_syn_input[:, i, :] = within_circuit_inputs + bu_td_inputs

            return sum_syn_input

        else:
            raise NotImplementedError

    def _get_predictions(self, source_layer, target_layer):
        return tf.transpose(self.weights[f'{target_layer}{source_layer}']) @ self.neuron_groups[source_layer]['rep_r'].r

    def _get_bu_errors(self, source_layer, target_layer, sign):

        if sign == 'pos':
            circuit_type = 'ppe'
        elif sign == 'neg':
            circuit_type = 'npe'
        else:
            raise ValueError('sign must be "pos" or "neg"')

        return self.weights[f'{source_layer}{target_layer}'] @ self.neuron_groups[source_layer][f'{sign}_pyr'].r

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

        # create a list to save errors (if not already present)
        for layer, neuron_dict in self.errors.items():
            for var in var_list:
                if (var not in neuron_dict.keys()) and (var in self.neuron_groups[layer].keys()):
                    neuron_dict[var] = []

        ### FIX ### what's the difference between this and self.get_error?
        ### FIX ### self.errors save not only errors but others if var_list contains other neuron types
        # record errors to the list
        for layer, neuron_dict in self.errors.items():
            neuron_keys = [neuron_key for neuron_key in neuron_dict if neuron_key in var_list]
            for neuron_i in neuron_keys:
                self.errors[layer][neuron_i].append(
                    # average over neurons
                    np.mean(
                        # average over batches
                        np.mean(
                            self.neuron_groups[layer][neuron_i], axis=1),
                    )
                )

    def learn(self, lr=1e-2, alpha=1e-4):

        for layers, weight in self.weights.items():

            l1_reg = alpha * tf.maximum(x=0.0, y=weight)

            dwp = lr * tf.einsum(
                'ik, jk -> ji',
                self.neuron_groups[f'layer_{layers[0]}']['ppe_pyr'].r,
                self.neuron_groups[f'layer_{layers[1]}']['rep_r'].r
            ) / (2 * self.batch_size)
            dwn = lr * tf.einsum(
                'ik, jk -> ji',
                self.neuron_groups[f'layer_{layers[0]}']['npe_pyr'].r,
                self.neuron_groups[f'layer_{layers[1]}']['rep_r'].r
            ) / (2 * self.batch_size)

            self.weights[layers] = tf.maximum(x=0.0, y=weight + dwp - dwn - l1_reg)

    def run(self, simParams, dataset, save=True):

        # define input size
        input_size = dataset['train_x'].shape[1]
        # define layer size
        net_size = [input_size, 28 ** 2]  # , 12 ** 2]
        # update sim parameters
        simParams.update_r({'net_size': net_size})

        # simulation params
        time_steps = int(simParams['sim_time'] / simParams['dt'])
        isi_steps = int(simParams['isi_time'] / simParams['dt'])
        n_batch = simParams['n_class'] * simParams['n_sample'] // simParams['batch_size']
        n_epoch = simParams['n_epoch']

        # plot interval (ms)
        plot_interval = simParams['plot_interval']
        # weight update interval (ms)
        learn_interval = time_steps // 1
        simParams.update_r({'learn_interval': learn_interval})

        recon_img_idcs = np.random.choice(simParams['batch_size'], simParams['recon_sample_n'], replace=False)
        simParams.update_r({'recon_img_idcs': recon_img_idcs})

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

    # make this function add to the list, instead of computing mse every instance
    def plot_error(self, curr_ep, n_ep):

        # error plot
        err_len = self.n_layers - 1

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
            err_fig, err_axs = plt.subplots(nrows=self.n_layers - 1, ncols=3, sharex='all')
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

        ### FIX ### what's run error?
        net_len = len(self.net_size) - 1
        re_fig, re_axs = plt.subplots(nrows=3, ncols=net_len, sharex='all')
        if net_len == 1:
            re_axs = re_axs.reshape(-1, 1)

        for i in range(len(self.net_size) - 1):
            ppe = self.err_per_iter[f'layer_{i}']['ppe_pyr']
            npe = self.err_per_iter[f'layer_{i}']['npe_pyr']
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

#     def plots(self, curr_ep, n_epoch, idx=None):
#
#         # input, pred, err
#         ncol = 4
#         # layers
#         nrow = len(self.neuron_group)
#
#         # plot
#         int_fig = plt.figure(figsize=(ncol * 3, nrow * 3))
#         grid_plot = plt.GridSpec(nrows=nrow, ncols=ncol)
#
#         for i, (layer, _) in enumerate(self.neuron_group.items()):
#
#             # input to layer
#             if idx is None:
#                 layer_input = self.neuron_group[layer]['rep_e']
#             else:
#                 layer_input = self.neuron_group[layer]['rep_e'][:, idx]
#             img_dim = np.sqrt(layer_input.shape[0]).astype(int)
#             input_ax = int_fig.add_subplot(grid_plot[i, 0])
#             input_image = input_ax.imshow(layer_input.reshape(img_dim, img_dim), cmap='gray')
#             int_fig.colorbar(input_image, ax=input_ax, shrink=0.4)
#             input_ax.axis('off')
#             input_ax.set_title(f'L{i} input')
#
#             if layer != f'layer_{len(self.neuron_group) - 1}':
#                 # pred to layer
#                 if idx is None:
#                     layer_pred = self.weights[f'{i}{i + 1}'].T @ self.neuron_group[f'layer_{i + 1}']['rep_r']
#                 else:
#                     layer_pred = self.weights[f'{i}{i + 1}'].T @ self.neuron_group[f'layer_{i + 1}']['rep_r'][:, idx]
#                 img_dim = np.sqrt(layer_pred.shape[0]).astype(int)
#                 pred_ax = int_fig.add_subplot(grid_plot[i, 1])
#                 pred_image = pred_ax.imshow(layer_pred.reshape(img_dim, img_dim), cmap='gray')
#                 int_fig.colorbar(pred_image, ax=pred_ax, shrink=0.4)
#                 pred_ax.axis('off')
#                 pred_ax.set_title(f'L{i} pred')
#
#                 # +ve err to layer
#                 if idx is None:
#                     layer_ppe = self.neuron_group[layer]['ppe_pyr']
#                 else:
#                     layer_ppe = self.neuron_group[layer]['ppe_pyr'][:, idx]
#                 img_dim = np.sqrt(layer_ppe.shape[0]).astype(int)
#                 ppe_ax = int_fig.add_subplot(grid_plot[i, 2])
#                 ppe_image = ppe_ax.imshow(layer_ppe.reshape(img_dim, img_dim), cmap='Reds')
#                 int_fig.colorbar(ppe_image, ax=ppe_ax, shrink=0.4)
#                 ppe_ax.axis('off')
#                 ppe_ax.set_title(f'L{i} pPE')
#
#                 # -ve err to layer
#                 if idx is None:
#                     layer_npe = self.neuron_group[layer]['npe_pyr']
#                 else:
#                     layer_npe = self.neuron_group[layer]['npe_pyr'][:, idx]
#                 img_dim = np.sqrt(layer_npe.shape[0]).astype(int)
#                 npe_ax = int_fig.add_subplot(grid_plot[i, 3])
#                 npe_image = npe_ax.imshow(layer_npe.reshape(img_dim, img_dim), cmap='Blues')
#                 int_fig.colorbar(npe_image, ax=npe_ax, shrink=0.4)
#                 npe_ax.axis('off')
#                 npe_ax.set_title(f'L{i} nPE')
#
#         int_fig.tight_layout()
#         int_fig.suptitle(f'Epoch #{curr_ep}/{n_epoch}')
#
#         return int_fig
#
#     def batch_plots(self, img_idcs, curr_ep, n_epoch):
#
#         # input, pred, err+, err-
#         ncol = 4
#         # layers
#         nrow = self.n_layers - 1
#
#         # plot
#         batch_fig, batch_ax = plt.subplots(
#             nrows=nrow, ncols=ncol, sharex='row', sharey='row',
#             figsize=(ncol * 5, nrow * 5)
#         )
#         batch_ax = batch_ax.flatten()
#         for i, (layer, _) in enumerate(self.neuron_group.items()):
#
#             if i < nrow:
#                 # input to layer
#                 # rand_idx = np.random.choice(np.arange(self.neuron_group[layer]['rep_e'].shape[1]), 16, replace=False)
#                 layer_input = self.neuron_group[layer]['rep_e'][:, img_idcs]
#                 layer_pred = (self.weights[f'{i}{i + 1}'].T @ self.neuron_group[f'layer_{i + 1}']['rep_r'])[:, img_idcs]
#                 layer_ppe = self.neuron_group[layer]['ppe_pyr'][:, img_idcs]
#                 layer_npe = self.neuron_group[layer]['npe_pyr'][:, img_idcs]
#
#                 subplt_inp = [layer_input, layer_pred, layer_ppe, layer_npe]
#                 subplt_idx = [i * ncol + j for j in range(ncol)]
#                 subplt_lbl = ['input', 'prediction', 'PE+', 'PE-']
#                 subplt_clr = ['gray', 'gray', 'Reds', 'Blues']
#
#                 for k in range(len(subplt_idx)):
#                     self.add_subplot(batch_ax[subplt_idx[k]], batch_fig, subplt_inp[k], subplt_clr[k])
#                     batch_ax[subplt_idx[k]].set_title(f'L{i} {subplt_lbl[k]}')
#
#         batch_fig.suptitle(f'Epoch #{curr_ep}/{n_epoch}')
#
#         return batch_fig
#
#     @staticmethod
#     def add_subplot(ax, fig, inputs, colors='gray'):
#
#         x_ticks, y_ticks, image = generate_reconstruction_pad(
#             img_mat=inputs, nx=4
#         )
#         input_subplot = ax.imshow(image, cmap=colors)
#         ax.set_xticks(np.arange(*x_ticks))
#         ax.set_yticks(np.arange(*y_ticks))
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.grid(True, ls='--')
#         fig.colorbar(input_subplot, ax=ax, shrink=0.2)
#         ax.axis('off')
#
#         # return ax
#
#     def batch_plots_ind(self, curr_ep, n_epoch, n_img_show=16):
#
#         # input, pred, err+, err-
#         ncol = 9
#         # layers
#         nrow = len(self.neuron_group) - 1
#
#         # plot
#         batch_fig, batch_ax = plt.subplots(
#             nrows=nrow, ncols=ncol, sharex='row', sharey='row',
#             figsize=(ncol * 3, nrow * 3)
#         )
#         batch_ax = batch_ax.flatten()
#         for i, (layer, _) in enumerate(self.neuron_group.items()):
#
#             if i < nrow:
#                 # input to layer
#                 rand_idx = np.random.choice(np.arange(self.neuron_group[layer]['rep_e'].shape[1]), n_img_show, replace=False)
#                 layer_input = self.neuron_group[layer]['rep_e'][:, rand_idx]
#
#                 # get predictions to each neuron type
#                 rep_r = self.neuron_group[f'layer_{i + 1}']['rep_r']
#                 layer_pred = {}
#                 for weight_pair_key, weight_pair in self.weights[f'{i}{i + 1}'].items():
#                     pre_neuron, post_neuron = weight_pair_key.split(', ')
#                     if pre_neuron == 'rep_r':
#                         layer_pred[post_neuron] = (weight_pair @ rep_r)[:, rand_idx]
#
#                 layer_ppe = self.neuron_group[layer]['ppe_pyr'][:, rand_idx]
#                 layer_npe = self.neuron_group[layer]['npe_pyr'][:, rand_idx]
#
#                 subplt_inp = [layer_input] + [pred_r for _, pred_r in layer_pred.items()] + [layer_ppe, layer_npe]
#                 subplt_idx = [i * ncol + j for j in range(ncol)]
#                 subplt_lbl = ['input'] + [pred_key for pred_key, _ in layer_pred.items()] + ['PE+', 'PE-']
#                 subplt_clr = ['gray'] * 7 + ['Reds', 'Blues']
#
#                 for k in range(len(subplt_idx)):
#                     self.add_subplot(batch_ax[subplt_idx[k]], batch_fig, subplt_inp[k], subplt_clr[k])
#                     batch_ax[subplt_idx[k]].set_title(f'L{i} {subplt_lbl[k]}')
#
#         batch_fig.suptitle(f'Epoch #{curr_ep}/{n_epoch}')
#
#         return batch_fig
#
#
# def error_plot(model, n_batch, n_epoch, time_steps):
#     # extraploate mse per batch and epoch to plot against mse per tstep
#     batch_x = np.linspace(0, n_batch * n_epoch * time_steps + 1, n_batch * n_epoch)
#     epoch_x = np.linspace(0, n_batch * n_epoch * time_steps + 1, n_epoch)
#
#     # row = layer, col = PE type
#     net_len = len(model.net_size) - 1
#     fig, axs = plt.subplots(nrows=len(model.net_size) - 1, ncols=2, sharex='all', figsize=(10, 5))
#     if net_len == 1:
#         axs = axs.reshape(1, -1)
#
#     err_dict = {}
#     for i, (layer_i, pe_dict) in enumerate(model.errors.items()):
#         err_dict[layer_i] = {}
#         for j, (signed_err, errors) in enumerate(pe_dict.items()):
#             # save to dict
#             err_dict[layer_i][signed_err] = {
#                 'per_tstep': errors,
#                 'per_batch': [np.mean(errors[batch_i * time_steps: (1 + batch_i) * time_steps])
#                               for batch_i in range(n_batch * n_epoch)],
#                 'per_epoch': [np.mean(errors[epoch_i * (n_batch * time_steps): (1 + epoch_i) * (n_batch * time_steps)])
#                               for epoch_i in range(n_epoch)]
#             }
#             # plot
#             axs[i, j].plot(
#                 err_dict[layer_i][signed_err]['per_tstep'],
#                 c='b', alpha=0.2, label='layer avg'
#             )
#             axs[i, j].plot(
#                 batch_x, err_dict[layer_i][signed_err]['per_batch'],
#                 c='g', alpha=0.4, label='batch avg'
#             )
#             axs[i, j].plot(
#                 epoch_x, err_dict[layer_i][signed_err]['per_epoch'],
#                 c='r', label='epoch avg', alpha=1.0)
#
#             axs[i, j].spines['top'].set_visible(False)
#             axs[i, j].spines['right'].set_visible(False)
#             axs[i, j].set_title(f'{layer_i}, {signed_err}')
#             axs[i, j].set_xticks([])
#
#             if i == len(pe_dict.keys()) - 2:
#                 axs[i, j].set_xticks(
#                     np.arange(
#                         len(err_dict[layer_i][signed_err]['per_tstep']) + 1
#                     )[::time_steps * n_batch]
#                 )
#                 axs[i, j].set_xticklabels(np.arange(n_epoch + 1))
#
#     # axis labels
#     fig.text(0.5, 0.005, 'Training epoch # ', ha='center')
#     fig.text(0.05, 0.5, 'Firing rate (Hz)', va='center', rotation='vertical')
#
#     # legend
#     handles, labels = fig.axes[1].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper right')
#
#     return err_dict, fig
#
#
# def get_np_weight_keys(conn_mat):
#
#     neuron_types = ['pyr', 'pv', 'sst', 'vip']
#     np_keys = []
#
#     for i, source in enumerate(neuron_types):
#         for j, target in enumerate(neuron_types):
#             if conn_mat[j, i] != 0:
#                 np_keys.append(f'{source}_{target}')
#
#     rep_connectivity = ['e, i', 'i, e', 'e, r']
#     for conn in rep_connectivity:
#
#
#     return np_keys
# def save_model(save_path, dataset, params, weights):
#     # create model directory
#     if os.path.exists(save_path):
#         pass
#     else:
#         os.mkdir(save_path)
#
#     # save dataset
#     pickle_save(save_path, 'dataset.pkl', dataset)
#     # save params
#     pickle_save(save_path, 'sim_params.pkl', params)
#     # save weights
#     pickle_save(save_path, 'weights.pkl', weights)
#
# def plot_training_errors(sim_params, errs):
#
#     # fig 2B - errors
#     tsim = int(sim_params['sim_time'] / sim_params['dt'])
#     tisi = int(sim_params['isi_time'] / sim_params['dt'])
#     plt.close('all')
#     aa = np.zeros(sim_params['n_epoch'])
#     bb = np.zeros(sim_params['n_epoch'])
#     len_epoch = int(tsim + tisi) * int(sim_params['n_class'] * sim_params['n_sample'] / sim_params['batch_size'])
#     for i in range(sim_params['n_epoch']):
#         curr_epoch_ppe = errs['layer_0']['ppe_pyr'][i * len_epoch: (i + 1) * len_epoch]
#         curr_epoch_npe = errs['layer_0']['npe_pyr'][i * len_epoch: (i + 1) * len_epoch]
#         aa[i] = curr_epoch_ppe[-1]
#         bb[i] = curr_epoch_npe[-1]
#     fig, ax = plt.subplots(1, 1, figsize=(7, 5))
#     ax.plot(aa, c='#CA181D', lw=3.0, label='PE+')
#     ax.plot(bb, c='#2070B4', lw=3.0, label='PE-')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_linewidth(2)
#     ax.spines['bottom'].set_linewidth(2)
#     ax.set_xlabel('training iteration', fontsize=20)
#     ax.set_ylabel('firing rate (a.u.)', fontsize=20)
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     fig.legend(labelcolor='linecolor', fontsize=20, edgecolor='white')  # , fancybox=True, shadow=True)
#     fig.tight_layout()
#
#     return fig


def assign_hetero_connectivity(combi_dict, n_neurons):

    # connectivity
    connectivity = np.zeros(shape=(3, n_neurons))
    # within circuit connectivity indices
    within_circuit_conn_idcs = np.unique([tup[0] for tup in list(combi_dict.keys())])
    # connectivity
    connectivity[0] = np.random.choice(a=within_circuit_conn_idcs, size=n_neurons)
    # loop
    for i, circuit_conn in enumerate(connectivity[0]):
        possible_input_combi = np.array([[tup[1], tup[2]] for tup in list(combi_dict.keys()) if tup[0] == circuit_conn])
        random_choice = np.random.choice(a=len(possible_input_combi), size=1)[0]
        connectivity[1:, i] = possible_input_combi[random_choice]

    return connectivity.astype(int)

def construct_conn_mat(conn_mat, conn_type):

    if conn_type == 'within':

        pe_conn_mat = np.zeros((8, 8))
        pe_conn_mat[:4, :4] = conn_mat
        pe_conn_mat[4:, 4:] = conn_mat

        return tf.constant(value=pe_conn_mat, dtype=tf.float32)

    elif conn_type == 'bu_td':

        bu_td_conn_mat = np.zeros((8, 2))
        bu_td_conn_mat[:4] = conn_mat[0]
        bu_td_conn_mat[4:] = conn_mat[1]

        return tf.constant(value=bu_td_conn_mat, dtype=tf.float32)

    else:
        raise ValueError('conn_type must be "within" or "bu_td"')

if __name__ == '__main__':

    with open('../test/combi_dict.pkl', 'rb') as f:
        combi_dict = pickle.load(f)
    with open('../results/sim_params.pkl', 'rb') as f:
        trained_params = pickle.load(f)
    inp_patterns = np.load('../test/input_patterns.npy')
    syn_patterns = np.load('../test/syn_patterns.npy')

    conn_idc_pairs = assign_hetero_connectivity(combi_dict=combi_dict, n_neurons=1024)

    pc_net = network(simParams=trained_params, pretrained_weights=None)