import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from parameters import dt
from practice_scalar import dr
import tensorflow as tf
from scipy import sparse

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_error_over_time(pe_mat, idx_itv, imcolor, plt_name,
                         plot_xy=32, vmin=0, cbar_max=None):
    pes = pe_mat[:, ::idx_itv]
    total_len = pes.shape[1]

    nrow = int(total_len / 5)
    ncol = int(total_len / (total_len / 5))
    prg_pe_fig, prg_pe_axs = plt.subplots(
        nrows=nrow, ncols=ncol,
        sharex=True, sharey=True,
        figsize=(ncol * 3, nrow * 3)
    )

    for prg_i, prg_ax in enumerate(prg_pe_axs.flatten()):
        pe_i = prg_ax.imshow(pes[:, prg_i].reshape(plot_xy, plot_xy), cmap=imcolor, vmin=vmin, vmax=cbar_max)
        plt.colorbar(pe_i, ax=prg_ax, shrink=0.6)
        prg_ax.set_title(f'iter #{idx_itv * prg_i}')
        prg_ax.axis('off')

    prg_pe_fig.suptitle(plt_name)
    prg_pe_fig.tight_layout()

    return prg_pe_fig


def create_weights(target, source, sparsity=False, conn_prob=0.1):
    if sparsity:
        return sparse.random(target, source, density=conn_prob).toarray()
    else:
        return np.random.random((target, source)) / np.sqrt(source)


T = 5.0
T_steps = int(T / dt)

# input
max_fr = 30.0

# # X (3x3)
# input_x = np.zeros(9)
# input_x[::2] = max_fr
# plt_title = 'example X shape'

# # mnist: lr = 5.0
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)))
# input_x = x_train[x_idx].flatten() / 255.0 * max_fr
# plt_title = 'mnist digit'
#
# # fashion mnist: lr = 5.0, nepoch=30
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)))
# input_x = x_train[x_idx].flatten() / 255.0 * max_fr
# plt_title = 'fashion mnist sample'

# grayscale cifar-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_idx = np.random.choice(np.arange(len(x_train)), 1)
input_x = x_train[x_idx].flatten() / 255.0 * max_fr
plt_title = 'cifar-10 sample'

n_ch, img_x, img_y = np.squeeze(x_train[x_idx], axis=0).shape
plt.imshow(input_x.reshape(n_ch, img_x, img_y) / max_fr)
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.title(plt_title)
plt.show()

# network size
n0 = len(input_x)
n0i = int(n0 / 4)
n1 = 1024 # (int(np.sqrt(n0 // 3)) + 4) ** 2 * 3
n1i = n1
n_inh = n0

# R0 circuit: n0 = nPixel, n0i = 3
r0_e = np.zeros((n0, T_steps))
r0_i = np.zeros((n0i, T_steps))

# R1 circuit
r1_e = np.zeros((n1, T_steps))
r1_i = np.zeros((n1i, T_steps))
r1_r = np.zeros((n1, T_steps))
# r1_e[:, 0] = sparse.random(n1, 1, density=0.2).toarray().flatten()
# r1_i[:, 0] = sparse.random(n1i, 1, density=0.2).toarray().flatten()
# error circuits
r_pe_r = np.zeros((n0, T_steps))

# initialize pPE circuit
r_ppe_e = np.zeros((n0, T_steps))
r_ppe_pv = np.zeros((n_inh, T_steps))
r_ppe_sst = np.zeros((n_inh, T_steps))
r_ppe_vip = np.zeros((n_inh, T_steps))

# initialize nPE circuit
r_npe_e = np.zeros((n0, T_steps))
r_npe_pv = np.zeros((n_inh, T_steps))
r_npe_sst = np.zeros((n_inh, T_steps))
r_npe_vip = np.zeros((n_inh, T_steps))

# initialize weights

# bottom layer: rep E-I
w_r0_ei = create_weights(n0i, n0)
w_r0_ie = 0
# top layer: rep E-I
w_r1_ei = create_weights(n1i, n1)
w_r1_ie = 1  # create_weights(n1, n1i)
# bottom-up: pPE pyr -> rep
w_0pe_1e = create_weights(n1, n0)
w_0ne_1i = create_weights(n1, n0)
# # top-down: r1r -> r0s
# w_td = create_weights(n0, n1)
# top-down: rep -> pPE
w_01p_pv = create_weights(n_inh, n1)
w_01p_sst = create_weights(n_inh, n1)
w_01p_vip = create_weights(n_inh, n1)
# top-down: rep -> nPE
w_01n_e = create_weights(n0, n1)
w_01n_sst = create_weights(n_inh, n1)
w_01n_vip = create_weights(n_inh, n1)

# self loops
w_self = 1.0

lr = 1e-4
n_epoch = 1
hebb_window = 100

pmse = []
nmse = []
reconstruction_ppe = {'pv': [], 'sst': [], 'vip': []}
reconstruction_npe = {'e': [], 'sst': [], 'vip': []}

func_type = 'jorge'

for epoch_i in range(n_epoch):

    for time_i in trange(T_steps - 1):

        # if i < int(T_steps * 0.1):
        #     # r0_exc = X - r0_inh
        #     input_r0_e = np.zeros(input_x.shape) - w_r0_ie @ r0_i[:, time_i]
        # else:
        #     # r0_exc = X - r0_inh
        #     input_r0_e = input_x - w_r0_ie @ r0_i[:, time_i]

        input_r0_e = input_x  # - w_r0_ie @ r0_i[:, time_i]
        # r0_inh = w_EI @ r0_exc
        input_r0_i = w_r0_ei @ r0_e[:, time_i]

        # r1_exc = pPE - r1_inh (nPE) + self loop
        input_r1_e = w_0pe_1e @ r_ppe_e[:, time_i] - r1_i[:, time_i]  # + w_r1_ee * r1_e[:, i]
        # r1_inh = nPE + r1_exc
        input_r1_i = w_0ne_1i @ r_npe_e[:, time_i]  # + w_r1_ei @ r1_e[:, time_i]
        # r1 rep = r1_exc + self loop
        input_r1_r = r1_e[:, time_i]  # + w_r1_rr * r1_r[:, i]

        # r0_pPe_exc = dampened input - PV - SST + self loop
        input_ppe_e = r0_e[:, time_i] - r_ppe_pv[:, time_i] - r_ppe_sst[:, time_i]  # + w_r0_pyr * r_ppe_e[:, i]
        # r0_pPE_PV = pred + r0_pPE_exc - self loop
        input_ppe_pv = w_01p_pv @ r1_r[:, time_i] + r_ppe_e[:, time_i]  # - w_r0_pv * r_ppe_pv[:, i]
        # r0_pPE_SST = pred + r0_pPE_exc - VIP
        input_ppe_sst = w_01p_sst @ r1_r[:, time_i] + r_ppe_e[:, time_i] - r_ppe_vip[:, time_i]
        # r0_pPE_VIP = pred + r0_pPE_exc
        input_ppe_vip = w_01p_vip @ r1_r[:, time_i] + r_ppe_e[:, time_i]

        # r0_nPe_exc = dampened input + pred - PV - SST + self loop
        input_npe_e = r0_e[:, time_i] + w_01n_e @ r1_r[:, time_i] - r_npe_pv[:, time_i] - r_npe_sst[:,
                                                                                             time_i]  # + w_r0_pyr * r_npe_e[:, i]
        # r0_nPe_PV = dampened input + r_nPE_exc - self loop
        input_npe_pv = r0_e[:, time_i] + r_npe_e[:, time_i]  # - w_r0_pv * r_npe_pv[:, i]
        # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
        input_npe_sst = r0_e[:, time_i] + w_01n_sst @ r1_r[:, time_i] + r_npe_e[:, time_i] - r_npe_vip[:, time_i]
        # r0_nPe_VIP = pred + r0_nPE_exc
        input_npe_vip = w_01n_vip @ r1_r[:, time_i] + r_npe_e[:, time_i]

        r0_e[:, time_i + 1] = dr(r=r0_e[:, time_i], inputs=input_r0_e, func=func_type, ei_type='exc')
        r0_i[:, time_i + 1] = dr(r=r0_i[:, time_i], inputs=input_r0_i, func=func_type)

        r1_e[:, time_i + 1] = dr(r=r1_e[:, time_i], inputs=input_r1_e, func=func_type, ei_type='exc')
        r1_i[:, time_i + 1] = dr(r=r1_i[:, time_i], inputs=input_r1_i, func=func_type)
        r1_r[:, time_i + 1] = dr(r=r1_r[:, time_i], inputs=input_r1_r, func=func_type, ei_type='exc')

        r_ppe_e[:, time_i + 1] = dr(r=r_ppe_e[:, time_i], inputs=input_ppe_e, func=func_type, ei_type='exc')
        r_ppe_pv[:, time_i + 1] = dr(r=r_ppe_pv[:, time_i], inputs=input_ppe_pv, func=func_type)
        r_ppe_sst[:, time_i + 1] = dr(r=r_ppe_sst[:, time_i], inputs=input_ppe_sst, func=func_type)
        r_ppe_vip[:, time_i + 1] = dr(r=r_ppe_vip[:, time_i], inputs=input_ppe_vip, func=func_type)

        r_npe_e[:, time_i + 1] = dr(r=r_npe_e[:, time_i], inputs=input_npe_e, func=func_type, ei_type='exc')
        r_npe_pv[:, time_i + 1] = dr(r=r_npe_pv[:, time_i], inputs=input_npe_pv, func=func_type)
        r_npe_sst[:, time_i + 1] = dr(r=r_npe_sst[:, time_i], inputs=input_npe_sst, func=func_type)
        r_npe_vip[:, time_i + 1] = dr(r=r_npe_vip[:, time_i], inputs=input_npe_vip, func=func_type)

        # record error
        pmse.append(np.mean(r_ppe_e[:, time_i + 1 - hebb_window:time_i].mean(axis=1) ** 2))
        nmse.append(np.mean(r_npe_e[:, time_i + 1 - hebb_window:time_i].mean(axis=1) ** 2))

        if (time_i + 1) % hebb_window == 0:

            exc1 = r1_e[:, time_i+1-hebb_window:time_i].mean(axis=1)
            inh1 = r1_i[:, time_i+1-hebb_window:time_i].mean(axis=1)
            rep1 = r1_r[:, time_i+1-hebb_window:time_i].mean(axis=1)

            dw_0pe_1e = -lr * (np.einsum('i,j->ij', exc1, r_ppe_e[:, time_i+1-hebb_window:time_i].mean(axis=1).T))
            dw_0ne_1i = -lr * (np.einsum('i,j->ij', inh1, r_npe_e[:, time_i+1-hebb_window:time_i].mean(axis=1).T))

            dw_01p_pv = -lr * (np.einsum('i,j->ij', r_ppe_pv[:, time_i+1-hebb_window:time_i].mean(axis=1).T, rep1))
            dw_01p_sst = -lr * (np.einsum('i,j->ij', r_ppe_sst[:, time_i+1-hebb_window:time_i].mean(axis=1).T, rep1))
            dw_01p_vip = -lr * (np.einsum('i,j->ij', r_ppe_vip[:, time_i+1-hebb_window:time_i].mean(axis=1).T, rep1))

            dw_01n_e = -lr * (np.einsum('i,j->ij', r_npe_e[:, time_i+1-hebb_window:time_i].mean(axis=1).T, rep1))
            dw_01n_sst = -lr * (np.einsum('i,j->ij', r_npe_sst[:, time_i+1-hebb_window:time_i].mean(axis=1).T, rep1))
            dw_01n_vip = -lr * (np.einsum('i,j->ij', r_npe_vip[:, time_i+1-hebb_window:time_i].mean(axis=1).T, rep1))

            # w_0pe_1e = np.maximum(w_0pe_1e - dw_0pe_1e + dw_0ne_1i, 0.0)
            # w_0ne_1i = w_0pe_1e
            w_0pe_1e = np.maximum(w_0pe_1e - dw_0pe_1e, 0.0)
            w_0ne_1i = np.maximum(w_0ne_1i - dw_0ne_1i, 0.0)

            w_01p_pv = np.maximum(w_01p_pv - dw_01p_pv, 0.0)
            w_01p_sst = np.maximum(w_01p_sst - dw_01p_sst, 0.0)
            w_01p_vip = np.maximum(w_01p_vip - dw_01p_vip, 0.0)

            w_01n_e = np.maximum(w_01n_e - dw_01n_e, 0.0)
            w_01n_sst = np.maximum(w_01n_sst - dw_01n_sst, 0.0)
            w_01n_vip = np.maximum(w_01n_vip - dw_01n_vip, 0.0)

            reconstruction_ppe['pv'].append((w_01p_pv @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            reconstruction_ppe['sst'].append((w_01p_sst @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            reconstruction_ppe['vip'].append((w_01p_vip @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            reconstruction_npe['e'].append((w_01n_e @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            reconstruction_npe['sst'].append((w_01n_sst @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            reconstruction_npe['vip'].append((w_01n_vip @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))

# summary plot
sum_fig, sum_axs = plt.subplots(nrows=3, ncols=3, sharex='all', sharey='all', figsize=(15, 15))
# input
inp_plt = sum_axs[0, 0].imshow(r0_e[:, -hebb_window:].mean(axis=1).reshape(n_ch, img_x, img_y) / max_fr)
sum_axs[0, 0].set_title('R0')
# pPE
ppe_plt = sum_axs[0, 1].imshow(r_ppe_e[:, -hebb_window:].mean(axis=1).reshape(n_ch, img_x, img_y) / max_fr)
sum_axs[0, 1].set_title('pPE')
# nPE
npe_plt = sum_axs[0, 2].imshow(r_npe_e[:, -hebb_window:].mean(axis=1).reshape(n_ch, img_x, img_y) / max_fr)
sum_axs[0, 2].set_title('nPE')
# prediction to pPE circuit
ppe_pred = sum_axs[1, 0].imshow(reconstruction_ppe['pv'][-1].reshape(n_ch, img_x, img_y) / max_fr)
sum_axs[1, 0].set_title('Prediction to pv+')
# prediction to pPE circuit
npe_pred = sum_axs[1, 1].imshow(reconstruction_ppe['sst'][-1].reshape(n_ch, img_x, img_y) / max_fr)
sum_axs[1, 1].set_title('Prediction to sst+')
# prediction to pPE circuit
npe_pred = sum_axs[1, 2].imshow(reconstruction_ppe['vip'][-1].reshape(n_ch, img_x, img_y) / max_fr)
sum_axs[1, 2].set_title('Prediction to vip+')
# prediction to nPE circuit
ppe_pred = sum_axs[2, 0].imshow(reconstruction_npe['e'][-1].reshape(n_ch, img_x, img_y) / max_fr)
sum_axs[2, 0].set_title('Prediction to e-')
# prediction to nPE circuit
npe_pred = sum_axs[2, 1].imshow(reconstruction_npe['sst'][-1].reshape(n_ch, img_x, img_y) / max_fr)
sum_axs[2, 1].set_title('Prediction to sst-')
# prediction to nPE circuit
npe_pred = sum_axs[2, 2].imshow(reconstruction_npe['vip'][-1].reshape(n_ch, img_x, img_y) / max_fr)
sum_axs[2, 2].set_title('Prediction to vip-')

for ax_i in sum_axs.flatten():
    ax_i.axis('off')

sum_fig.tight_layout()
sum_fig.show()

# error plot
pe_fig, pe_axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
pe_axs[0].plot(pmse, c='r')
pe_axs[0].set_title('pPE')
pe_axs[1].plot(nmse, c='b')
pe_axs[1].set_title('nPE')

pe_fig.tight_layout()
pe_fig.show()

# # # # reconsturction plot over training epoch
# # # progress_by = int(n_epoch / 10)
# # # progress_fig, progress_axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 5))
# # # for i, ax in enumerate(progress_axs.flatten()):
# # #     img = ax.imshow(reconstruction_ppe[::progress_by][i].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
# # #     progress_fig.colorbar(img, ax=ax, shrink=0.3)
# # #     ax.set_title(f'iter #{(i + 1) * progress_by}')
# # #     ax.set_axis_off()
# # # progress_fig.tight_layout()
# # # progress_fig.show()
# #
# # # # create prediction, pPE and nPE plots in time series (e.g., every 100 ms)
# # # time_int = 1000
# # # ppred_prg = plot_error_over_time(
# # #     w_0pe_1e.T @ r1_r, time_int, 'Reds', plt_name='prediction to pPE', plot_xy=img_xy
# # # )
# # # ppred_prg.show()
# #
# # # npred_prg = plot_error_over_time(
# # #     w_01n_e @ r1_r, time_int, 'Reds', plt_name='prediction to nPE', plot_xy=img_xy
# # # )
# # # npred_prg.show()
# # # ppe_prg = plot_error_over_time(r_ppe_e, time_int, 'Reds', plt_name='pPE', plot_xy=img_xy)
# # # ppe_prg.show()
# # # npe_prg = plot_error_over_time(r_npe_e, time_int, 'Blues', plt_name='nPE', plot_xy=img_xy)
# # # npe_prg.show()
# #
# # # why does pPE increase across time? nPE decreases
# # # check on the pc of scalar value
# # # study the gradient descent
# #
prg_by = 5
n_plots = T_steps // (hebb_window * prg_by) - 1
n_pred_target = len(reconstruction_ppe) + len(reconstruction_npe)

prg_fig, prg_axs = plt.subplots(nrows=n_pred_target, ncols=n_plots, figsize=(2 * n_plots, 6))
prg_axs = prg_axs.flatten()
for j in range(n_plots):
    for i, (key, grp) in enumerate(reconstruction_ppe.items()):
        img = prg_axs[(i * n_plots) + j].imshow(grp[j].reshape(n_ch, img_x, img_y) / max_fr)
        prg_axs[(i * n_plots) + j].set_title(f'{(j + 1) * hebb_window * prg_by} ms')
    for k, (key, grp) in enumerate(reconstruction_npe.items()):
        img2 = prg_axs[(k + 3) * n_plots + j].imshow(grp[j].reshape(n_ch, img_x, img_y) / max_fr)
        prg_axs[(k + 3) * n_plots + j].set_title(f'{(j + 1) * hebb_window * prg_by} ms')

for ax in prg_axs:
    ax.set_axis_off()

prg_fig.tight_layout()
prg_fig.suptitle('Prediction over time', fontsize=16, y=1.05)
prg_fig.show()