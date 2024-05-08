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
        raise 'func not implemented yet!'

    return r + (-r + fx + bg) * (dt / tau)


def oddball_input(x, y, n_repeat):
    norm_img_repeat = tf.reshape(tf.tile(x, [n_repeat, 1]), (n_repeat, 32, 32))

    return tf.concat([norm_img_repeat, tf.expand_dims(y, axis=0)], 0)


n_img = 3
T = 9.0
T_steps = int(T / dt)

# input
max_fr = 30.0
# # X (3x3)
# input_x = np.zeros(9)
# input_x[::2] = max_fr
# plt_title = 'example X shape'

# mnist: lr = 5.0
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_idx = np.random.choice(np.arange(len(x_train)), n_img)
input_x = tf.constant(x_train[x_idx] / 255.0 * max_fr, dtype=tf.float32)
plt_title = 'mnist digit'
#
# # fashion mnist: lr = 5.0, nepoch=30
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)), n_img)
# input_x = tf.constant(x_train[x_idx] / 255.0 * max_fr, dtype=tf.float32)
# plt_title = 'fashion mnist sample'

# # grayscale cifar-10
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# # x_idx = []
# # for cl_i in np.unique(y_train):
# #     x_idx.append(np.random.choice(np.where(y_train==[cl_i])[0], 1, replace=False)[0])
# # x_idx = np.random.choice(np.arange(len(x_train)), n_img)
# x_idx =[38900, 13523, 32792]
# input_x = tf.squeeze(tf.image.rgb_to_grayscale(x_train[x_idx]), -1) / tf.reduce_max(x_train) * max_fr
# plt_title = 'cifar-10 sample'

oddball_paradigm = False
if oddball_paradigm:
    input_x = oddball_input(input_x[0], input_x[1], n_img - 1)
else:
    pass

img_xy = input_x.shape[-1]

inp_fig, inp_axs = plt.subplots(n_img, 1, figsize=(3, 3 * n_img))
for i, ax in enumerate(inp_axs.flatten()):
    inp_img = ax.imshow(input_x[i], cmap='gray')
    inp_fig.colorbar(inp_img, ax=ax, shrink=0.6)
    ax.axis('off')
    ax.set_title(f'input #{i + 1}')
inp_fig.suptitle(plt_title)
inp_fig.tight_layout()
inp_fig.show()

# network size
n0 = np.product(input_x.shape[1:])
n0i = int(n0 / 4)
n1 = 625#(int(np.sqrt(n0)) + 4) ** 2
n1i = n1
n_inh = n0

# R0 circuit: n0 = nPixel, n0i = 3
r0_e = tf.Variable(tf.zeros(shape=(n0, ), dtype=tf.float32))
r0_i = tf.Variable(tf.zeros(shape=(n0i, ), dtype=tf.float32))

# R1 circuit
r1_e = tf.Variable(tf.zeros(shape=(n1, ), dtype=tf.float32))
r1_i = tf.Variable(tf.zeros(shape=(n1i, ), dtype=tf.float32))
r1_r = tf.Variable(tf.zeros(shape=(n1, ), dtype=tf.float32))

# error circuits
r_pe_r = tf.Variable(tf.zeros(shape=(n0, ), dtype=tf.float32))

# initialize pPE circuit
r_ppe_e = tf.Variable(tf.zeros(shape=(n0, ), dtype=tf.float32))
r_ppe_pv = tf.Variable(tf.zeros(shape=(n_inh, ), dtype=tf.float32))
r_ppe_sst = tf.Variable(tf.zeros(shape=(n_inh, ), dtype=tf.float32))
r_ppe_vip = tf.Variable(tf.zeros(shape=(n_inh, ), dtype=tf.float32))

# initialize nPE circuit
r_npe_e = tf.Variable(tf.zeros(shape=(n0, ), dtype=tf.float32))
r_npe_pv = tf.Variable(tf.zeros(shape=(n_inh, ), dtype=tf.float32))
r_npe_sst = tf.Variable(tf.zeros(shape=(n_inh, ), dtype=tf.float32))
r_npe_vip = tf.Variable(tf.zeros(shape=(n_inh, ), dtype=tf.float32))

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

lr = 5e-5
n_epoch = 1
hebb_window = 200

psse = []
nsse = []
reconstruction_ppe = []
# reconstruction_npe = []

func_type = 'jorge'

for epoch_i in range(n_epoch):

    for time_i in trange(T_steps):

        # series of images witout isi
        input_idx = (time_i) // (T_steps // n_img)
        input_r0_e = tf.reshape(input_x[input_idx], np.product(input_x[input_idx].shape))

        # r1_exc = pPE - r1_inh (nPE) + self loop
        input_r1_e = tf.einsum('ij, j -> i', w_0pe_1e, r_ppe_e) - r1_i
        # r1_inh = nPE + r1_exc
        input_r1_i = tf.einsum('ij, j -> i', w_0ne_1i, r_npe_e)
        # r1 rep = r1_exc + self loop
        input_r1_r = r1_e

        # r0_pPe_exc = dampened input - PV - SST + self loop
        input_ppe_e = r0_e - r_ppe_pv - r_ppe_sst
        # r0_pPE_PV = pred + r0_pPE_exc - self loop
        input_ppe_pv = tf.einsum('ij, j -> i', tf.transpose(w_0pe_1e), r1_r) + r_ppe_e
        # r0_pPE_SST = pred + r0_pPE_exc - VIP
        input_ppe_sst = tf.einsum('ij, j -> i', tf.transpose(w_0pe_1e), r1_r) + r_ppe_e - r_ppe_vip
        # r0_pPE_VIP = pred + r0_pPE_exc
        input_ppe_vip = tf.einsum('ij, j -> i', tf.transpose(w_0pe_1e), r1_r) + r_ppe_e

        # r0_nPe_exc = dampened input + pred - PV - SST + self loop
        input_npe_e = r0_e + tf.einsum('ij, j -> i', tf.transpose(w_0ne_1i), r1_r) - r_npe_pv - r_npe_sst
        # r0_nPe_PV = dampened input + r_nPE_exc - self loop
        input_npe_pv = r0_e + r_npe_e
        # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
        input_npe_sst = r0_e + tf.einsum('ij, j -> i', tf.transpose(w_0ne_1i), r1_r) + r_npe_e - r_npe_vip
        # r0_nPe_VIP = pred + r0_nPE_exc
        input_npe_vip = tf.einsum('ij, j -> i', tf.transpose(w_0ne_1i), r1_r) + r_npe_e

        r0_e.assign(dr(r=r0_e, inputs=input_r0_e, func=func_type, ei_type='exc'))

        r1_e.assign(dr(r=r1_e, inputs=input_r1_e, func=func_type, ei_type='exc'))
        r1_i.assign(dr(r=r1_i, inputs=input_r1_i, func=func_type))
        r1_r.assign(dr(r=r1_r, inputs=input_r1_r, func=func_type, ei_type='exc'))

        r_ppe_e.assign(dr(r=r_ppe_e, inputs=input_ppe_e, func=func_type, ei_type='exc'))
        r_ppe_pv.assign(dr(r=r_ppe_pv, inputs=input_ppe_pv, func=func_type))
        r_ppe_sst.assign(dr(r=r_ppe_sst, inputs=input_ppe_sst, func=func_type))
        r_ppe_vip.assign(dr(r=r_ppe_vip, inputs=input_ppe_vip, func=func_type))

        r_npe_e.assign(dr(r=r_npe_e, inputs=input_npe_e, func=func_type, ei_type='exc'))
        r_npe_pv.assign(dr(r=r_npe_pv, inputs=input_npe_pv, func=func_type))
        r_npe_sst.assign(dr(r=r_npe_sst, inputs=input_npe_sst, func=func_type))
        r_npe_vip.assign(dr(r=r_npe_vip, inputs=input_npe_vip, func=func_type))

        if (time_i + 1) % hebb_window == 0:

            dw_0pe_1e = -lr * tf.einsum('i,j->ij', r1_e, r_ppe_e)
            dw_0ne_1i = -lr * tf.einsum('i,j->ij', r1_i, r_npe_e)

            w_0pe_1e = tf.maximum(w_0pe_1e - dw_0pe_1e + dw_0ne_1i, 0.0)
            w_0ne_1i = w_0pe_1e

            reconstruction_ppe.append(
                tf.einsum(
                    'ij,j-> i',
                    tf.transpose(w_0pe_1e),
                    r1_r
                )
            )

            # reconstruction_npe.append(
            #     tf.einsum(
            #         'ij,j-> i',
            #         tf.transpose(w_0ne_1i),
            #         r1_r
            #     )
            # )

            psse.append(
                r_ppe_e.numpy()
            )
            nsse.append(
                r_npe_e.numpy()
            )

        # psse.append(
        #     tf.reduce_mean(
        #         r_ppe_e ** 2
        #         ).numpy()
        #     )
        # nsse.append(
        #     tf.reduce_mean(
        #         r_npe_e ** 2
        #         ).numpy()
        #     )

        if time_i in [(i+1) * (T_steps // n_img) - 1 for i in range(n_img)]:

            # summary plot
            sum_fig, sum_axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 15))
            # input
            inp_img = tf.reshape(r0_e, shape=(img_xy, img_xy))
            inp_plt = sum_axs[0].imshow(inp_img, cmap='gray')
            sum_fig.colorbar(inp_plt, ax=sum_axs[0], shrink=0.6)
            sum_axs[0].axis('off')
            sum_axs[0].set_title('R0')
            # pPE
            ppe_img = tf.reshape(r_ppe_e, shape=(img_xy, img_xy))
            ppe_plt = sum_axs[1].imshow(ppe_img, cmap='Reds')
            sum_fig.colorbar(ppe_plt, ax=sum_axs[1], shrink=0.6)
            sum_axs[1].axis('off')
            sum_axs[1].set_title('pPE')
            # nPE
            npe_img = tf.reshape(r_npe_e, shape=(img_xy, img_xy))
            npe_plt = sum_axs[2].imshow(npe_img, cmap='Blues')
            sum_fig.colorbar(npe_plt, ax=sum_axs[2], shrink=0.6)
            sum_axs[2].axis('off')
            sum_axs[2].set_title('nPE')
            # prediction to pPE circuit
            pred_sig = tf.einsum('ij,j-> i', tf.transpose(w_0pe_1e), r1_r)
            ppe_pred = sum_axs[3].imshow(tf.reshape(pred_sig, shape=(img_xy, img_xy)), cmap='gray')
            sum_fig.colorbar(ppe_pred, ax=sum_axs[3], shrink=0.6)
            sum_axs[3].axis('off')
            sum_axs[3].set_title('Prediction to pPE')
            # error-corrected
            corrected = sum_axs[4].imshow(tf.reshape(pred_sig + r_ppe_e - r_npe_e,
                                                     shape=(img_xy, img_xy)), cmap='Greens')
            sum_fig.colorbar(corrected, ax=sum_axs[4], shrink=0.6)
            sum_axs[4].axis('off')
            sum_axs[4].set_title('corrected')

            sum_fig.suptitle(f'Epoch #{epoch_i + 1}/{n_epoch}, img #{int((time_i + 1) / (T_steps / 3))}/{n_img}')
            sum_fig.tight_layout()
            sum_fig.show()

# error plot
pe_fig, pe_axs = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(15, 8))
for ax_j in range(1):
    pe_axs[0, ax_j].plot(psse[int(T_steps) * ax_j : int(T_steps) * (ax_j + 1)], c='r')
    pe_axs[0, ax_j].set_title(f'pPE: e{ax_j + 1}/{n_epoch}')
    pe_axs[1, ax_j].plot(nsse[int(T_steps) * ax_j: int(T_steps) * (ax_j + 1)], c='b')
    pe_axs[1, ax_j].set_title(f'nPE: e{ax_j + 1}/{n_epoch}')

for pe_ax in pe_axs.flatten():
    for i in range(n_img):
        pe_ax.axvline(x=i * int(T_steps / n_img), c='k', linestyle='--')

pe_fig.tight_layout()
pe_fig.show()

# reconstruction progress plot
reduce_by = 1
n_col = T_steps // n_img // hebb_window // reduce_by
# n_plt = n_col * n_epoch

fig, axs = plt.subplots(nrows=n_img, ncols=n_col, sharex='all', sharey='all', figsize=(n_col * 1, n_img))
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.imshow(tf.reshape(reconstruction_ppe[::reduce_by][i], shape=(32,32)), vmin=0, vmax=30, cmap='gray')
    ax.axis('off')
    ax.set_title(f'{(i + 1) * hebb_window} ms')
fig.tight_layout()
fig.show()
# fig.savefig('/home/kwangjun/Pictures/progress.png', dpi=300)


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

# # create prediction, pPE and nPE plots in time series (e.g., every 100 ms)
# time_int = 100
# ppred_prg = plot_error_over_time(
#     w_0pe_1e.T @ r1_r, time_int, 'Reds', plt_name='prediction to pPE', plot_xy=img_xy
# )
# ppred_prg.show()

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

