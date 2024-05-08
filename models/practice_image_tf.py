import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from parameters import tau_exc, tau_inh, bg_exc, bg_inh, dt
import tensorflow as tf
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
        img = tf.reshape(pes[:, prg_i], shape=(plot_xy, plot_xy))
        pe_i = prg_ax.imshow(img, cmap=imcolor, vmin=vmin, vmax=cbar_max)
        plt.colorbar(pe_i, ax=prg_ax, shrink=0.6)
        prg_ax.set_title(f'iter #{idx_itv * prg_i}')
        prg_ax.axis('off')

    prg_pe_fig.suptitle(plt_name)
    prg_pe_fig.tight_layout()

    return prg_pe_fig


def create_weights(target, source, sparsity=False, conn_prob=0.1):
    if sparsity:
        return tf.sparse.random(target, source, density=conn_prob)

    else:
        return tf.random.uniform((target, source))


def jorge(x, d=10.0, theta=dt * 0.1):
    return (x - theta) / (1 - tf.exp(-d * (x - theta)))


def dr(r, inputs, func, ei_type='inh'):

    if ei_type == 'exc':
        tau = tau_exc
        bg = bg_exc
    elif ei_type == 'inh':
        tau = tau_inh
        bg = bg_inh

    if func == 'jorge':
        fx = jorge(x=inputs)
    else:
        raise('func not implemented yet!')

    return r + (-r + fx + bg) * (dt / tau)


T = 0.5
T_steps = int(T / dt)

# input
max_fr = 1.0

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
x_idx = np.random.choice(np.arange(len(x_train)), 1)[0]
input_x = tf.constant(x_train[x_idx] / 255.0 * max_fr, dtype=tf.float32)
plt_title = 'fashion mnist sample'

# # grayscale cifar-10
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)), 1)
# input_x = tf.squeeze(tf.image.rgb_to_grayscale(x_train[x_idx]), [0, -1]) / tf.reduce_max(x_train) * max_fr
# plt_title = 'cifar-10 sample'


plt.figure()
plt.imshow(input_x, cmap='gray')
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.title(plt_title)
plt.show()

# network size
n0 = np.product(input_x.shape)
n0i = int(n0 / 4)
n1 = (int(np.sqrt(n0)) + 4) ** 2
n1i = n1
n_inh = n0

# R0 circuit: n0 = nPixel, n0i = 3
r0_e = tf.Variable(tf.zeros(shape=(n0, T_steps), dtype=tf.float32))
r0_i = tf.Variable(tf.zeros(shape=(n0i, T_steps), dtype=tf.float32))

# R1 circuit
r1_e = tf.Variable(tf.zeros(shape=(n1, T_steps), dtype=tf.float32))
r1_i = tf.Variable(tf.zeros(shape=(n1i, T_steps), dtype=tf.float32))
r1_r = tf.Variable(tf.zeros(shape=(n1, T_steps), dtype=tf.float32))

# error circuits
r_pe_r = tf.Variable(tf.zeros(shape=(n0, T_steps), dtype=tf.float32))

# initialize pPE circuit
r_ppe_e = tf.Variable(tf.zeros(shape=(n0, T_steps), dtype=tf.float32))
r_ppe_pv = tf.Variable(tf.zeros(shape=(n_inh, T_steps), dtype=tf.float32))
r_ppe_sst = tf.Variable(tf.zeros(shape=(n_inh, T_steps), dtype=tf.float32))
r_ppe_vip = tf.Variable(tf.zeros(shape=(n_inh, T_steps), dtype=tf.float32))

# initialize nPE circuit
r_npe_e = tf.Variable(tf.zeros(shape=(n0, T_steps), dtype=tf.float32))
r_npe_pv = tf.Variable(tf.zeros(shape=(n_inh, T_steps), dtype=tf.float32))
r_npe_sst = tf.Variable(tf.zeros(shape=(n_inh, T_steps), dtype=tf.float32))
r_npe_vip = tf.Variable(tf.zeros(shape=(n_inh, T_steps), dtype=tf.float32))

# initialize weights

# bottom layer: rep E-I
w_r0_ei = create_weights(n0i, n0)
w_r0_ie = 0
# top layer: rep E-I
w_r1_ei = create_weights(n1i, n1)
w_r1_ie = 1  # create_weights(n1, n1i)
# bottom-up: pPE pyr -> rep
w_0pe_1e = create_weights(n1, n0)
w_0ne_1i = w_0pe_1e

# self loops
w_self = 1.0

lr = 1
n_epoch = 10
hebb_window = 100

psse = []
nsse = []
reconstruction_ppe = []
reconstruction_npe = []

func_type = 'jorge'

for epoch_i in range(n_epoch):

    for time_i in trange(T_steps - 1):

        # if i < int(T_steps * 0.1):
        #     # r0_exc = X - r0_inh
        #     input_r0_e = np.zeros(input_x.shape) - w_r0_ie @ r0_i[:, time_i]
        # else:
        #     # r0_exc = X - r0_inh
        #     input_r0_e = input_x - w_r0_ie @ r0_i[:, time_i]

        input_r0_e = tf.reshape(input_x, np.product(input_x.shape))  # - w_r0_ie @ r0_i[:, time_i]
        # r0_inh = w_EI @ r0_exc
        input_r0_i = tf.einsum('ij, j -> i', w_r0_ei, r0_e[:, time_i])

        # r1_exc = pPE - r1_inh (nPE) + self loop
        input_r1_e = tf.einsum('ij, j -> i', w_0pe_1e, r_ppe_e[:, time_i]) - r1_i[:, time_i]
        # r1_inh = nPE + r1_exc
        input_r1_i = tf.einsum('ij, j -> i', w_0ne_1i, r_npe_e[:, time_i])
        # r1 rep = r1_exc + self loop
        input_r1_r = r1_e[:, time_i]

        # r0_pPe_exc = dampened input - PV - SST + self loop
        input_ppe_e = r0_e[:, time_i] - r_ppe_pv[:, time_i] - r_ppe_sst[:, time_i]
        # r0_pPE_PV = pred + r0_pPE_exc - self loop
        input_ppe_pv = tf.einsum('ij, j -> i', tf.transpose(w_0pe_1e), r1_r[:, time_i]) + r_ppe_e[:, time_i]
        # r0_pPE_SST = pred + r0_pPE_exc - VIP
        input_ppe_sst = tf.einsum('ij, j -> i', tf.transpose(w_0pe_1e), r1_r[:, time_i]) + r_ppe_e[:, time_i] - r_ppe_vip[:, time_i]
        # r0_pPE_VIP = pred + r0_pPE_exc
        input_ppe_vip = tf.einsum('ij, j -> i', tf.transpose(w_0pe_1e), r1_r[:, time_i]) + r_ppe_e[:, time_i]

        # r0_nPe_exc = dampened input + pred - PV - SST + self loop
        input_npe_e = r0_e[:, time_i] + tf.einsum('ij, j -> i', tf.transpose(w_0ne_1i), r1_r[:, time_i]) - r_npe_pv[:, time_i] - r_npe_sst[:, time_i]
        # r0_nPe_PV = dampened input + r_nPE_exc - self loop
        input_npe_pv = r0_e[:, time_i] + r_npe_e[:, time_i]
        # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
        input_npe_sst = r0_e[:, time_i] + tf.einsum('ij, j -> i', tf.transpose(w_0ne_1i), r1_r[:, time_i]) + r_npe_e[:, time_i] - r_npe_vip[:, time_i]
        # r0_nPe_VIP = pred + r0_nPE_exc
        input_npe_vip = tf.einsum('ij, j -> i', tf.transpose(w_0ne_1i), r1_r[:, time_i]) + r_npe_e[:, time_i]

        r0_e[:, time_i + 1].assign(dr(r=r0_e[:, time_i], inputs=input_r0_e, func=func_type, ei_type='exc'))
        r0_i[:, time_i + 1].assign(dr(r=r0_i[:, time_i], inputs=input_r0_i, func=func_type))

        r1_e[:, time_i + 1].assign(dr(r=r1_e[:, time_i], inputs=input_r1_e, func=func_type, ei_type='exc'))
        r1_i[:, time_i + 1].assign(dr(r=r1_i[:, time_i], inputs=input_r1_i, func=func_type))
        r1_r[:, time_i + 1].assign(dr(r=r1_r[:, time_i], inputs=input_r1_r, func=func_type, ei_type='exc'))

        r_ppe_e[:, time_i + 1].assign(dr(r=r_ppe_e[:, time_i], inputs=input_ppe_e, func=func_type, ei_type='exc'))
        r_ppe_pv[:, time_i + 1].assign(dr(r=r_ppe_pv[:, time_i], inputs=input_ppe_pv, func=func_type))
        r_ppe_sst[:, time_i + 1].assign(dr(r=r_ppe_sst[:, time_i], inputs=input_ppe_sst, func=func_type))
        r_ppe_vip[:, time_i + 1].assign(dr(r=r_ppe_vip[:, time_i], inputs=input_ppe_vip, func=func_type))

        r_npe_e[:, time_i + 1].assign(dr(r=r_npe_e[:, time_i], inputs=input_npe_e, func=func_type, ei_type='exc'))
        r_npe_pv[:, time_i + 1].assign(dr(r=r_npe_pv[:, time_i], inputs=input_npe_pv, func=func_type))
        r_npe_sst[:, time_i + 1].assign(dr(r=r_npe_sst[:, time_i], inputs=input_npe_sst, func=func_type))
        r_npe_vip[:, time_i + 1].assign(dr(r=r_npe_vip[:, time_i], inputs=input_npe_vip, func=func_type))

        if (time_i + 1) % hebb_window == 0:
            exc1 = tf.reduce_mean(r1_e[:, time_i + 1 - hebb_window:time_i], axis=1)
            inh1 = tf.reduce_mean(r1_i[:, time_i + 1 - hebb_window:time_i], axis=1)
            rep1 = tf.reduce_mean(r1_r[:, time_i + 1 - hebb_window:time_i], axis=1)

            dw_0pe_1e = -lr * (tf.einsum(
                'i,j->ij', exc1, tf.reduce_mean(r_ppe_e[:, time_i + 1 - hebb_window:time_i], axis=1)))
            dw_0ne_1i = -lr * (tf.einsum(
                'i,j->ij', inh1, tf.reduce_mean(r_npe_e[:, time_i + 1 - hebb_window:time_i], axis=1)))

            w_0pe_1e = np.maximum(w_0pe_1e - dw_0pe_1e + dw_0ne_1i, 0.0)
            w_0ne_1i = w_0pe_1e

            reconstruction_ppe.append(
                tf.einsum(
                    'ij,j-> i',
                    tf.transpose(w_0pe_1e),
                    tf.reduce_mean(r1_r[:, time_i + 1 - hebb_window:time_i], axis=1)
                )
            )

            reconstruction_npe.append(
                tf.einsum(
                    'ij,j-> i',
                    tf.transpose(w_0ne_1i),
                    tf.reduce_mean(r1_r[:, time_i + 1 - hebb_window:time_i], axis=1)
                )
            )

    psse.append(
        tf.reduce_sum(
            tf.reduce_mean(
                r_ppe_e[:, time_i + 1 - hebb_window:time_i], axis=1
            )
        )
    )
    nsse.append(
        tf.reduce_sum(
            tf.reduce_mean(
                r_npe_e[:, time_i + 1 - hebb_window:time_i], axis=1
            )
        )
    )

# summary plot
sum_fig, sum_axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 15))
# input
inp_img = tf.reshape(tf.reduce_mean(r0_e[:, -hebb_window:], axis=1), shape=(img_xy, img_xy))
inp_plt = sum_axs[0].imshow(inp_img, cmap='gray')
sum_fig.colorbar(inp_plt, ax=sum_axs[0], shrink=0.6)
sum_axs[0].axis('off')
sum_axs[0].set_title('R0')
# pPE
ppe_img = tf.reshape(tf.reduce_mean(r_ppe_e[:, -hebb_window:], axis=1), shape=(img_xy, img_xy))
ppe_plt = sum_axs[1].imshow(ppe_img, cmap='Reds')
sum_fig.colorbar(ppe_plt, ax=sum_axs[1], shrink=0.6)
sum_axs[1].axis('off')
sum_axs[1].set_title('pPE')
# nPE
npe_img = tf.reshape(tf.reduce_mean(r_npe_e[:, -hebb_window:], axis=1), shape=(img_xy, img_xy))
npe_plt = sum_axs[2].imshow(npe_img, cmap='Blues')
sum_fig.colorbar(npe_plt, ax=sum_axs[2], shrink=0.6)
sum_axs[2].axis('off')
sum_axs[2].set_title('nPE')
# prediction to pPE circuit
ppe_pred = sum_axs[3].imshow(tf.reshape(reconstruction_ppe[-1], shape=(img_xy, img_xy)), cmap='Reds')
sum_fig.colorbar(ppe_pred, ax=sum_axs[3], shrink=0.6)
sum_axs[3].axis('off')
sum_axs[3].set_title('Prediction to pPE')
# prediction to nPE circuit
npe_pred = sum_axs[4].imshow(tf.reshape(reconstruction_npe[-1], shape=(img_xy, img_xy)), cmap='Reds')
sum_fig.colorbar(npe_pred, ax=sum_axs[4], shrink=0.6)
sum_axs[4].axis('off')
sum_axs[4].set_title('Prediction to nPE')

sum_fig.tight_layout()
sum_fig.show()

# error plot
pe_fig, pe_axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
pe_axs[0].plot(psse, c='r')
pe_axs[0].set_title('pPE')
pe_axs[1].plot(nsse, c='b')
pe_axs[1].set_title('nPE')

pe_fig.tight_layout()
pe_fig.show()

# # reconsturction plot over training epoch
# progress_by = int(n_epoch / 10)
# progress_fig, progress_axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 5))
# for i, ax in enumerate(progress_axs.flatten()):
#     img = ax.imshow(reconstruction_ppe[::progress_by][i].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
#     progress_fig.colorbar(img, ax=ax, shrink=0.3)
#     ax.set_title(f'iter #{(i + 1) * progress_by}')
#     ax.set_axis_off()
# progress_fig.tight_layout()
# progress_fig.show()

# create prediction, pPE and nPE plots in time series (e.g., every 100 ms)
time_int = 100
ppred_prg = plot_error_over_time(
    w_0pe_1e.T_total @ r1_r, time_int, 'Reds', plt_name='prediction to pPE', plot_xy=img_xy
)
ppred_prg.show()

# npred_prg = plot_error_over_time(
#     w_01n_e @ r1_r, time_int, 'Reds', plt_name='prediction to nPE', plot_xy=img_xy
# )
# npred_prg.show()
# ppe_prg = plot_error_over_time(r_ppe_e, time_int, 'Reds', plt_name='pPE', plot_xy=img_xy)
# ppe_prg.show()
# npe_prg = plot_error_over_time(r_npe_e, time_int, 'Blues', plt_name='nPE', plot_xy=img_xy)
# npe_prg.show()

# why does pPE increase across time? nPE decreases
# check on the pc of scalar value
# study the gradient descent
