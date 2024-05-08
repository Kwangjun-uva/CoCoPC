import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from parameters import tau_exc, tau_inh, dt, bg_exc, bg_inh
from tools import jorge, sigmoid, ReLu

def dr(r, inputs, func='relu', ei_type='inh'):
    if ei_type == 'exc':
        tau = tau_exc
        bg = bg_exc
    elif ei_type == 'inh':
        tau = tau_inh
        bg = bg_inh

    if func == 'jorge':
        fx = jorge(x=inputs)
    elif func == 'sigmoid':
        fx = sigmoid(x=inputs)
    elif func == 'relu':
        fx = ReLu(x=inputs)

    return r + (-r + fx + bg) * (dt / tau)

def init_network():

    # initialize activity
    network = {
        # R0 circuit
        'r0_e': np.zeros(T_steps),
        'r0_i': np.zeros(T_steps),

        # R1 circuit
        'r1_e': np.zeros(T_steps),
        'r1_i': np.zeros(T_steps),
        'r1_r': np.zeros(T_steps),
        # pPE circuit
        'r_ppe_e': np.zeros(T_steps),
        'r_ppe_pv': np.zeros(T_steps),
        'r_ppe_sst': np.zeros(T_steps),
        'r_ppe_vip': np.zeros(T_steps),
        # nPE circuit
        'r_npe_e': np.zeros(T_steps),
        'r_npe_pv': np.zeros(T_steps),
        'r_npe_sst': np.zeros(T_steps),
        'r_npe_vip': np.zeros(T_steps)
    }

    return network

if __name__ == "__main__":

    # 2-layer predictive coding
    T = 4.0  # sec
    T_steps = int(T / dt)

    # initialize activity
    pc_net = init_network()

    # external input
    bu_input = 0.5

    # E-I synaptic strengths
    w_r0_ei = 1.0
    w_r0_ie = 1.0

    # inter-layer syn str
    w_plastic = {
        '01p': 1.0,
        '01n': 1.0,
    }

    # intra-layer syn str
    w_pyr_inh = 1.0

    # inter-layer syn str
    w_10 = 1.0

    # self-loop weight
    w_self = 1.0

    # learning params
    lr = {'p': 1e1, 'n': 1e1}#{'p': 1e-4, 'n': 1e-4}
    n_epoch = 5
    hebb_window = T_steps -1#int(T_steps / 5)  # int(T_steps * 0.5)

    # activation function
    func_name = 'relu'

    last_r1 = 0.0

    # iterate over multiple learning epochs
    for epoch_i in trange(n_epoch):

        # simulate for T seconds.
        for i in range(T_steps - 1):

            # basal fr
            if i < int(T_steps / 5):
                add = 0
            # bu = 10
            elif (i > int(T_steps / 5)) and (i < 2 * int(T_steps / 5)):
                add = 5.0
            # bu = 5
            elif (i > 2 * int(T_steps / 5)) and (i < 3 * int(T_steps / 5)):
                add = 0.0
            # bu = 30
            elif (i > 3 * int(T_steps / 5)) and (i < 4 * int(T_steps / 5)):
                add = -5.0
            # bu = 20
            elif i > 4 * int(T_steps / 5):
                add = -0.0
            else:
                add = 0.0

            # input to layer 0 rep circuit: e, i -> ie dampens input
            input_r0_e = bu_input + add #- w_r0_ie * pc_net['r0_i'][i] + w_self * pc_net['r0_e'][i]
            #input_r0_i = w_r0_ei * pc_net['r0_e'][i]
            # input to layer 1 rep circuit: e, i, r
            input_r1_e = w_plastic['01p'] * pc_net['r_ppe_e'][i] - pc_net['r1_i'][i] #+ w_self * pc_net['r1_e'][i]  # r1i -> r1e should be one-to-one
            input_r1_i = w_plastic['01n'] * pc_net['r_npe_e'][i] + pc_net['r1_e'][i]
            input_r1_r = pc_net['r1_e'][i] #+ w_self * pc_net['r1_r'][i]
            # input to layer 0 pPE circuit: e, pv, sst, vip
            input_ppe_e = pc_net['r0_e'][i] - pc_net['r_ppe_pv'][i] - pc_net['r_ppe_sst'][i] #+ w_self * pc_net['r_ppe_e'][i]
            input_ppe_pv = w_plastic['01p'] * pc_net['r1_r'][i] #+ w_pyr_inh * pc_net['r_ppe_e'][i]
            input_ppe_sst = w_plastic['01p'] * pc_net['r1_r'][i] - pc_net['r_ppe_vip'][i] #+ w_pyr_inh * pc_net['r_ppe_e'][i]
            input_ppe_vip = w_plastic['01p'] * pc_net['r1_r'][i] #+ w_pyr_inh * pc_net['r_ppe_e'][i]
            # input to layer 0 nPE circuit: e, pv, sst, vip
            input_npe_e = pc_net['r0_e'][i] + w_plastic['01n'] * pc_net['r1_r'][i] - pc_net['r_npe_pv'][i] - pc_net['r_npe_sst'][i] #+ w_self * pc_net['r_npe_e'][i]
            input_npe_pv = pc_net['r0_e'][i] #+ w_pyr_inh * pc_net['r_npe_e'][i]
            input_npe_sst = pc_net['r0_e'][i] + w_plastic['01n'] * pc_net['r1_r'][i] - pc_net['r_npe_vip'][i] #+ w_pyr_inh * pc_net['r_npe_e'][i]
            input_npe_vip = w_plastic['01n'] * pc_net['r1_r'][i] #+ w_pyr_inh * pc_net['r_npe_e'][i]

            # update firing rates
            pc_net['r0_e'][i + 1] = dr(r=pc_net['r0_e'][i], inputs=input_r0_e, func=func_name, ei_type='exc')
            #pc_net['r0_i'][i + 1] = dr(r=pc_net['r0_i'][i], inputs=input_r0_i, func=func_name)

            pc_net['r1_e'][i + 1] = dr(r=pc_net['r1_e'][i], inputs=input_r1_e, func=func_name, ei_type='exc')
            pc_net['r1_i'][i + 1] = dr(r=pc_net['r1_i'][i], inputs=input_r1_i, func=func_name)
            pc_net['r1_r'][i + 1] = dr(r=pc_net['r1_r'][i], inputs=input_r1_r, func=func_name, ei_type='exc')

            pc_net['r_ppe_e'][i + 1] = dr(r=pc_net['r_ppe_e'][i], inputs=input_ppe_e, func=func_name, ei_type='exc')
            pc_net['r_ppe_pv'][i + 1] = dr(r=pc_net['r_ppe_pv'][i], inputs=input_ppe_pv, func=func_name)
            pc_net['r_ppe_sst'][i + 1] = dr(r=pc_net['r_ppe_sst'][i], inputs=input_ppe_sst, func=func_name)
            pc_net['r_ppe_vip'][i + 1] = dr(r=pc_net['r_ppe_vip'][i], inputs=input_ppe_vip, func=func_name)

            pc_net['r_npe_e'][i + 1] = dr(r=pc_net['r_npe_e'][i], inputs=input_npe_e, func=func_name, ei_type='exc')
            pc_net['r_npe_pv'][i + 1] = dr(r=pc_net['r_npe_pv'][i], inputs=input_npe_pv, func=func_name)
            pc_net['r_npe_sst'][i + 1] = dr(r=pc_net['r_npe_sst'][i], inputs=input_npe_sst, func=func_name)
            pc_net['r_npe_vip'][i + 1] = dr(r=pc_net['r_npe_vip'][i], inputs=input_npe_vip, func=func_name)

            if (i + 1) % hebb_window == 0:
                # var
                r1e = pc_net['r1_e'][i + 1]
                r1i = pc_net['r1_i'][i + 1]
                r1r = pc_net['r1_r'][i + 1]

                r0_ppe = pc_net['r_ppe_e'][i + 1]
                r0_npe = pc_net['r_npe_e'][i + 1]

                r0p_pv = pc_net['r_ppe_pv'][i + 1]
                r0p_sst = pc_net['r_ppe_sst'][i + 1]
                r0p_vip = pc_net['r_ppe_vip'][i + 1]

                r0n_sst = pc_net['r_npe_sst'][i + 1]
                r0n_vip = pc_net['r_npe_vip'][i + 1]

                # update weights
                dw_01_p = lr['p'] * (r1r * r0_ppe)
                dw_01_n = lr['n'] * (r1r * r0_npe)

                w_plastic['01p'] = max(w_plastic['01p'] + dw_01_p - dw_01_n, 0.0)
                w_plastic['01n'] = w_plastic['01p']

            else:
                pass

        if (epoch_i + 1) % 1 == 0:

            progress_fig, progress_axs = plt.subplots(nrows=5, ncols=2, sharex='all', sharey='all')

            # input
            progress_axs[0, 0].plot(pc_net['r0_e'], c='black', label='input')
            progress_axs[0, 0].set_title('input: r0')
            # r1
            progress_axs[0, 1].plot(pc_net['r1_r'])
            progress_axs[0, 1].set_title('rep: r1r')
            # pPE, pyr
            progress_axs[1, 0].plot(pc_net['r_ppe_e'], c='red', label='pred to pPE')
            progress_axs[1, 0].set_title('pPE pyr')
            # pPE, pv
            progress_axs[2, 0].plot(pc_net['r_ppe_pv'], c='red', label='pred to pPE')
            progress_axs[2, 0].set_title('pPE PV')
            # pPE, sst
            progress_axs[3, 0].plot(pc_net['r_ppe_sst'], c='red', label='pred to pPE')
            progress_axs[3, 0].set_title('pPE SST')
            # pPE, vip
            progress_axs[4, 0].plot(pc_net['r_ppe_vip'], c='red', label='pred to pPE')
            progress_axs[4, 0].set_title('pPE VIP')

            # nPE, pyr
            progress_axs[1, 1].plot(pc_net['r_npe_e'], c='blue', label='pred to pPE')
            progress_axs[1, 1].set_title('nPE pyr')
            # nPE, pv
            progress_axs[2, 1].plot(pc_net['r_npe_pv'], c='blue', label='pred to pPE')
            progress_axs[2, 1].set_title('nPE PV')
            # nPE, sst
            progress_axs[3, 1].plot(pc_net['r_npe_sst'], c='blue', label='pred to pPE')
            progress_axs[3, 1].set_title('nPE SST')
            # nPE, vip
            progress_axs[4, 1].plot(pc_net['r_npe_vip'], c='blue', label='pred to pPE')
            progress_axs[4, 1].set_title('nPE VIP')

            for ax in progress_axs.flatten():
                ax.plot(pc_net['r0_e'], c='purple', ls='--')

            progress_fig.suptitle(f'learning {epoch_i + 1}/{n_epoch}')
            progress_fig.tight_layout()
            progress_fig.show()

            plt.plot(pc_net['r0_e'], c='black')
            plt.plot(w_plastic['01p'] * pc_net['r1_r'], c='r', label='pPE')
            plt.plot(w_plastic['01n'] * pc_net['r1_r'], c='b', label='nPE')
            plt.ylim([0, 10])
            plt.title(f'input and pred {epoch_i + 1}/{n_epoch}')
            plt.legend()
            plt.show()

    # fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
    # axs = axs.flatten()
    # npe_circuit = [pc_net['r_npe_e'], pc_net['r_npe_pv'], pc_net['r_npe_sst'], pc_net['r_npe_vip']]
    # npe_circuit_label = ['pyr', 'pv', 'sst', 'vip']
    # for i, ax in enumerate(axs):
    #     ax.plot(npe_circuit[i])
    #     ax.set_title(npe_circuit_label[i])
    #
    # fig.tight_layout()
    # fig.show()

    for key, grp in w_plastic.items():
        print (key, grp)