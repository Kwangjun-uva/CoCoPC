import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from parameters import dt
from practice_scalar import dr
import tensorflow as tf

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_error_over_time(pe_mat, idx_itv, imcolor, plt_name,
                         plot_xy=32, vmin=0, cbar_max=None):
    pes = pe_mat[:, ::idx_itv]
    total_len = pes.shape[1]

    prg_pe_fig, prg_pe_axs = plt.subplots(
        nrows=int(total_len / 5), ncols=int(total_len / (total_len / 5)),
        sharex=True, sharey=True
    )

    for prg_i, prg_ax in enumerate(prg_pe_axs.flatten()):
        pe_i = prg_ax.imshow(pes[:, prg_i].reshape(plot_xy, plot_xy), cmap=imcolor, vmin=vmin, vmax=cbar_max)
        plt.colorbar(pe_i, ax=prg_ax)
        prg_ax.set_title(f'iter #{25 * prg_i}')
        prg_ax.axis('off')

    prg_pe_fig.suptitle(plt_name)
    prg_pe_fig.tight_layout()

    return prg_pe_fig


T = 0.5
T_steps = int(T / dt)
hebb_window = -1
# hebb_window = int(T_steps * 0.1)

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
# fashion mnist: lr = 5.0, nepoch=30
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_idx = np.random.choice(np.arange(len(x_train)))
input_x = x_train[x_idx].flatten() / 255.0 * max_fr
plt_title = 'fashion mnist sample'

# # grayscale cifar-10
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)))
# input_x = tf.image.rgb_to_grayscale(x_train[x_idx]).numpy().flatten() / 255.0 * max_fr
# plt_title = 'cifar-10 sample'

img_xy = int(np.sqrt(len(input_x)))
plt.imshow(input_x.reshape(img_xy, img_xy), cmap='Reds')
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.title(plt_title)
plt.show()

# network size
n0 = len(input_x)
n0i = int(n0 / 4)
n1 = (int(np.sqrt(n0)) + 4) ** 2
n1i = n1
n_inh = n0

# R0 circuit: n0 = nPixel, n0i = 3
r0_e = np.zeros((n0, T_steps))
r0_i = np.zeros((n0i, T_steps))

# R1 circuit
r1_e = np.zeros((n1, T_steps))
r1_i = np.zeros((n1i, T_steps))
r1_r = np.zeros((n1, T_steps))

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
w_r0_ei = np.random.random((n0i, n0))
w_r0_ie = np.random.random((n0, n0i)) * 1 / (n0 * n0i)
# top layer: rep E-I
w_r1_ei = np.random.random((n1i, n1))
w_r1_ie = np.random.random((n1, n1i)) * 1 / (n1 * n1i)
# bottom-up: pPE pyr -> rep
w_0pe_1e = np.random.random((n1, n0))
w_0ne_1i = np.random.random((n1, n0))
# top-down: rep -> pPE
w_01p_pv = np.random.random((n_inh, n1))
w_01p_sst = np.random.random((n_inh, n1))
w_01p_vip = np.random.random((n_inh, n1))
# top-down: rep -> nPE
w_01n_e = np.random.random((n0, n1))
w_01n_sst = np.random.random((n_inh, n1))
w_01n_vip = np.random.random((n_inh, n1))

lr = 0.0001
n_epoch = 10
psse = []
nsse = []
reconstruction_ppe = []
reconstruction_npe = []

for epoch_i in trange(n_epoch):

    for i in range(T_steps - 1):

        if i < int(T_steps * 0.1):
            # r0_exc = X - r0_inh
            input_r0_e = np.zeros(input_x.shape) - w_r0_ie @ r0_i[:, i]
        else:
            # r0_exc = X - r0_inh
            input_r0_e = input_x - w_r0_ie @ r0_i[:, i]

        # r0_inh = w_EI @ r0_exc
        input_r0_i = w_r0_ei @ r0_e[:, i]

        # r1_exc = pPE - r1_inh (nPE) + self loop
        input_r1_e = w_0pe_1e @ r_ppe_e[:, i] - w_r1_ie @ r1_i[:, i] + r1_e[:, i]
        # r1_inh = nPE + r1_exc
        input_r1_i = w_0ne_1i @ r_npe_e[:, i] + w_r1_ei @ r1_e[:, i]
        # r1 rep = r1_exc + self loop
        input_r1_r = r1_e[:, i] + r1_r[:, i]

        # r0_pPe_exc = dampened input - PV - SST + self loop
        input_ppe_e = r0_e[:, i] - r_ppe_pv[:, i] - r_ppe_sst[:, i] + r_ppe_e[:, i]
        # r0_pPE_PV = pred + r0_pPE_exc - self loop
        input_ppe_pv = w_01p_pv @ r1_r[:, i] + r_ppe_e[:, i] - r_ppe_pv[:, i]
        # r0_pPE_SST = pred + r0_pPE_exc - VIP
        input_ppe_sst = w_01p_sst @ r1_r[:, i] + r_ppe_e[:, i] - r_ppe_vip[:, i]
        # r0_pPE_VIP = pred + r0_pPE_exc
        input_ppe_vip = w_01p_vip @ r1_r[:, i] + r_ppe_e[:, i]

        # r0_nPe_exc = dampened input + pred - PV - SST + self loop
        input_npe_e = r0_e[:, i] + w_01n_e @ r1_r[:, i] - r_npe_pv[:, i] - r_npe_sst[:, i] + r_npe_e[:, i]
        # r0_nPe_PV = dampened input + r_nPE_exc - self loop
        input_npe_pv = r0_e[:, i] + r_npe_e[:, i] - r_npe_pv[:, i]
        # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
        input_npe_sst = r0_e[:, i] + w_01n_sst @ r1_r[:, i] + r_npe_e[:, i] - r_npe_vip[:, i]
        # r0_nPe_VIP = pred + r0_nPE_exc
        input_npe_vip = w_01n_vip @ r1_r[:, i] + r_npe_e[:, i]

        r0_e[:, i + 1] = dr(r=r0_e[:, i], inputs=input_r0_e, ei_type='exc')
        r0_i[:, i + 1] = dr(r=r0_i[:, i], inputs=input_r0_i)

        r1_e[:, i + 1] = dr(r=r1_e[:, i], inputs=input_r1_e, ei_type='exc')
        r1_i[:, i + 1] = dr(r=r1_i[:, i], inputs=input_r1_i)
        r1_r[:, i + 1] = dr(r=r1_r[:, i], inputs=input_r1_r, ei_type='exc')

        r_ppe_e[:, i + 1] = dr(r=r_ppe_e[:, i], inputs=input_ppe_e, ei_type='exc')
        r_ppe_pv[:, i + 1] = dr(r=r_ppe_pv[:, i], inputs=input_ppe_pv)
        r_ppe_sst[:, i + 1] = dr(r=r_ppe_sst[:, i], inputs=input_ppe_sst)
        r_ppe_vip[:, i + 1] = dr(r=r_ppe_vip[:, i], inputs=input_ppe_vip)

        r_npe_e[:, i + 1] = dr(r=r_npe_e[:, i], inputs=input_npe_e, ei_type='exc')
        r_npe_pv[:, i + 1] = dr(r=r_npe_pv[:, i], inputs=input_npe_pv)
        r_npe_sst[:, i + 1] = dr(r=r_npe_sst[:, i], inputs=input_npe_sst)
        r_npe_vip[:, i + 1] = dr(r=r_npe_vip[:, i], inputs=input_npe_vip)

    exc1 = r1_e[:, -hebb_window:].mean(axis=1)
    inh1 = r1_i[:, -hebb_window:].mean(axis=1)
    rep1 = r1_r[:, -hebb_window:].mean(axis=1)

    dw_0pe_1e = -lr * (np.einsum('i,j->ij', exc1, r_ppe_e[:, -hebb_window:].mean(axis=1).T_total))
    dw_0ne_1i = -lr * (np.einsum('i,j->ij', inh1, r_npe_e[:, -hebb_window:].mean(axis=1).T_total))

    dw_01p_pv = -lr * (np.einsum('i,j->ij', r_ppe_pv[:, -hebb_window:].mean(axis=1).T_total, rep1))
    dw_01p_sst = -lr * (np.einsum('i,j->ij', r_ppe_sst[:, -hebb_window:].mean(axis=1).T_total, rep1))
    dw_01p_vip = -lr * (np.einsum('i,j->ij', r_ppe_vip[:, -hebb_window:].mean(axis=1).T_total, rep1))

    dw_01n_e = -lr * (np.einsum('i,j->ij', r_npe_e[:, -hebb_window:].mean(axis=1).T_total, rep1))
    dw_01n_sst = -lr * (np.einsum('i,j->ij', r_npe_sst[:, -hebb_window:].mean(axis=1).T_total, rep1))
    dw_01n_vip = -lr * (np.einsum('i,j->ij', r_npe_vip[:, -hebb_window:].mean(axis=1).T_total, rep1))

    w_0pe_1e = np.maximum(w_0pe_1e - dw_0pe_1e, 0.0)
    w_0ne_1i = np.maximum(w_0ne_1i - dw_0ne_1i, 0.0)

    w_01p_pv = np.maximum(w_01p_pv - dw_01p_pv, 0.0)
    w_01p_sst = np.maximum(w_01p_sst - dw_01p_sst, 0.0)
    w_01p_vip = np.maximum(w_01p_vip - dw_01p_vip, 0.0)

    w_01n_e = np.maximum(w_01n_e - dw_01n_e, 0.0)
    w_01n_sst = np.maximum(w_01n_sst - dw_01n_sst, 0.0)
    w_01n_vip = np.maximum(w_01n_vip - dw_01n_vip, 0.0)

    psse.append(np.sum(r_ppe_e[:, -hebb_window:].mean(axis=1) ** 2))
    nsse.append(np.sum(r_npe_e[:, -hebb_window:].mean(axis=1) ** 2))

    reconstruction_ppe.append((r_ppe_pv + r_ppe_sst)[:, -hebb_window:].mean(axis=1))
    reconstruction_npe.append((w_01n_e @ r1_r + r_npe_pv + r_npe_sst)[:, -hebb_window:].mean(axis=1))

# summary plot
vmax = r0_e.mean(axis=1).max()
plt.figure(figsize=(4, 15))
# input
plt.subplot(511)
plt.imshow(r0_e[:, -hebb_window:].mean(axis=1).reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
plt.colorbar(shrink=0.6)
plt.axis('off')
plt.title('R0')
# pPE
plt.subplot(512)
plt.imshow(r_ppe_e[:, -hebb_window:].mean(axis=1).reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
plt.colorbar(shrink=0.6)
plt.axis('off')
plt.title('pPE')
# nPE
plt.subplot(513)
plt.imshow(r_npe_e[:, -hebb_window:].mean(axis=1).reshape(img_xy, img_xy), cmap='Blues', vmin=0, vmax=vmax)
plt.colorbar(shrink=0.6)
plt.axis('off')
plt.title('nPE')
# prediction to pPE circuit
plt.subplot(514)
plt.imshow(reconstruction_ppe[-1].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
plt.colorbar(shrink=0.6)
plt.axis('off')
plt.title('Prediction to pPE')
# prediction to nPE circuit
plt.subplot(515)
plt.imshow(reconstruction_npe[-1].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
plt.colorbar(shrink=0.6)
plt.axis('off')
plt.title('Prediction to nPE')
plt.tight_layout()
plt.show()

# error plot
plt.figure()
plt.plot(psse, c='r', label='pPE')
plt.plot(nsse, c='b', label='nPE')
plt.title('error')
plt.legend()
plt.show()

# reconsturction plot over training epoch
progress_by = int(n_epoch / 10)
progress_fig, progress_axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 5))
for i, ax in enumerate(progress_axs.flatten()):
    img = ax.imshow(reconstruction_ppe[::progress_by][i].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
    progress_fig.colorbar(img, ax=ax, shrink=0.3)
    ax.set_title(f'iter #{(i + 1) * progress_by}')
    ax.set_axis_off()
progress_fig.tight_layout()
progress_fig.show()

# create prediction, pPE and nPE plots in time series (e.g., every 100 ms)
ppred_prg = plot_error_over_time(
    r_ppe_pv + r_ppe_sst, 25, 'Reds', plt_name='prediction to pPE', plot_xy=img_xy
)
ppred_prg.show()
npred_prg = plot_error_over_time(
    w_01n_e @ r1_r + r_npe_pv + r_npe_sst, 25, 'Reds', plt_name='prediction to nPE', plot_xy=img_xy
)
npred_prg.show()
ppe_prg = plot_error_over_time(r_ppe_e, 25, 'Reds', plt_name='pPE', plot_xy=img_xy)
ppe_prg.show()
npe_prg = plot_error_over_time(r_npe_e, 25, 'Blues', plt_name='nPE', plot_xy=img_xy)
npe_prg.show()

# why does pPE increase across time