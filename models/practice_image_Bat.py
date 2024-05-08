import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from alive_progress import alive_bar
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
        img = tf.reshape(pes, shape=(plot_xy, plot_xy))
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
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)))
# input_x = x_train[x_idx].flatten() / 255.0 * max_fr
# plt_title = 'fashion mnist sample'

def pick_samples(n_class, n_sample, img_set):

    if img_set == 'cifar':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x = tf.squeeze(tf.image.rgb_to_grayscale(x_train), [3]) / tf.reduce_max(x_train) * max_fr
    else:
        if img_set == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        elif img_set == 'fashion':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            raise('data_type not implemented yet!')
        x = x_train / tf.reduce_max(x_train) * max_fr

    idx = []
    for ci in range(n_class):
        idx.append((tf.where(y_train == ci)[:n_sample, 0]).numpy())
    idcs = np.array(idx).flatten()
    np.random.shuffle(idcs)

    return tf.gather(x, idcs), tf.reshape(tf.gather(y_train, idcs), shape=(len(idcs),))

# data type
dataset = 'mnist'
n_class = 3
n_sample = 256
input_x, input_y = pick_samples(n_class, n_sample, img_set=dataset)
plt_title = f'{dataset} sample'

img_xy = input_x.shape[-1]
cmap_type = 'gray'

# input plot
show_sample = np.minimum(n_sample, 10)
sample_idcs = np.ravel([np.argwhere(input_y.numpy() == cl_i)[:10] for cl_i in np.unique(input_y)])

input_fig, input_axs = plt.subplots(
    nrows=n_class, ncols=show_sample, figsize=(3 * show_sample, 3 * n_class)
)

# iterate over input_axs
input_axs = input_axs.flatten()
for i, axs in enumerate(input_axs):
    # plt_idx = (i // 10) * n_sample + i % 10
    im = axs.imshow(input_x[sample_idcs[i]], cmap=cmap_type)
    axs.axis("off")
    axs.set_title(f'class {input_y[sample_idcs[i]]}, sample #{i % 10 + 1}')

input_fig.suptitle(plt_title)
input_fig.subplots_adjust(right=0.8)
cbar_ax = input_fig.add_axes([0.85, 0.15, 0.03, 0.7])
input_fig.colorbar(im, cax=cbar_ax)
input_fig.show()

# network size
n0 = np.product(input_x.shape[1:])
n0i = int(n0 / 4)
n1 = (int(np.sqrt(n0)) + 4) ** 2
n1i = n1
n_inh = n0

batch_size = 128

# R0 circuit: n0 = nPixel, n0i = 3
r0_e = tf.Variable(tf.zeros(shape=(n0, batch_size), dtype=tf.float32))
# r0_i = tf.Variable(tf.zeros(shape=(n0i, batch_size), dtype=tf.float32))

# R1 circuit
r1_e = tf.Variable(tf.zeros(shape=(n1, batch_size), dtype=tf.float32))
r1_i = tf.Variable(tf.zeros(shape=(n1i, batch_size), dtype=tf.float32))
r1_r = tf.Variable(tf.zeros(shape=(n1, batch_size), dtype=tf.float32))

# error circuits
r_pe_r = tf.Variable(tf.zeros(shape=(n0, batch_size), dtype=tf.float32))

# initialize pPE circuit
r_ppe_e = tf.Variable(tf.zeros(shape=(n0, batch_size), dtype=tf.float32))
r_ppe_pv = tf.Variable(tf.zeros(shape=(n_inh, batch_size), dtype=tf.float32))
r_ppe_sst = tf.Variable(tf.zeros(shape=(n_inh, batch_size), dtype=tf.float32))
r_ppe_vip = tf.Variable(tf.zeros(shape=(n_inh, batch_size), dtype=tf.float32))

# initialize nPE circuit
r_npe_e = tf.Variable(tf.zeros(shape=(n0, batch_size), dtype=tf.float32))
r_npe_pv = tf.Variable(tf.zeros(shape=(n_inh, batch_size), dtype=tf.float32))
r_npe_sst = tf.Variable(tf.zeros(shape=(n_inh, batch_size), dtype=tf.float32))
r_npe_vip = tf.Variable(tf.zeros(shape=(n_inh, batch_size), dtype=tf.float32))

# initialize weights

# bottom layer: rep E-I
w_r0_ei = create_weights(n0i, n0)
# w_r0_ie = 0
# top layer: rep E-I
w_r1_ei = create_weights(n1i, n1)
w_r1_ie = 1
# bottom-up: pPE pyr -> rep
w_0pe_1e = create_weights(n1, n0)
w_0ne_1i = w_0pe_1e

# self loops
w_self = 1.0

T = 0.5
T_steps = int(T / dt)

lr = 1e-4
n_epoch = 3
hebb_window = 250

psse = []
nsse = []
reconstruction_over_epoch = []
# reconstruction_npe = []
# prg_idx = [n_sample * i + n_sample - 1 for i in range(n_class)]
prg_idx = np.ravel([np.argwhere(input_y.numpy()[-batch_size:] == cl_i)[0] for cl_i in np.unique(input_y)])
func_type = 'jorge'


for epoch_i in trange(n_epoch, desc='epoch', position=0):

    reconstruction_ppe = []
    for batch_i in trange(0, len(input_x), batch_size, desc='batch', position=1, leave=True):

        x_batch_i = tf.transpose(
            tf.reshape(
                input_x[batch_i:batch_i + batch_size], shape=(batch_size, np.product(input_x.shape[1:]))
            )
        )

        for time_i in range(T_steps):

            # if i < int(T_steps * 0.1):
            #     # r0_exc = X - r0_inh
            #     input_r0_e = np.zeros(input_x.shape) - w_r0_ie @ r0_i
            # else:
            #     # r0_exc = X - r0_inh
            #     input_r0_e = input_x - w_r0_ie @ r0_i

            input_r0_e = x_batch_i  # - w_r0_ie @ r0_i
            # # r0_inh = w_EI @ r0_exc
            # input_r0_i = tf.einsum('ij, jk -> ik', w_r0_ei, r0_e)

            # r1_exc = pPE - r1_inh (nPE) + self loop
            input_r1_e = tf.einsum('ij, jk -> ik', w_0pe_1e, r_ppe_e) - r1_i
            # r1_inh = nPE + r1_exc
            input_r1_i = tf.einsum('ij, jk -> ik', w_0ne_1i, r_npe_e)
            # r1 rep = r1_exc + self loop
            input_r1_r = r1_e

            # r0_pPe_exc = dampened input - PV - SST + self loop
            input_ppe_e = r0_e - r_ppe_pv - r_ppe_sst
            # r0_pPE_PV = pred + r0_pPE_exc - self loop
            input_ppe_pv = tf.einsum('ij, jk -> ik', tf.transpose(w_0pe_1e), r1_r) + r_ppe_e
            # r0_pPE_SST = pred + r0_pPE_exc - VIP
            input_ppe_sst = tf.einsum('ij, jk -> ik', tf.transpose(w_0pe_1e), r1_r) + r_ppe_e
            # r0_pPE_VIP = pred + r0_pPE_exc
            input_ppe_vip = tf.einsum('ij, jk -> ik', tf.transpose(w_0pe_1e), r1_r) + r_ppe_e

            # r0_nPe_exc = dampened input + pred - PV - SST + self loop
            input_npe_e = r0_e + tf.einsum('ij, jk -> ik', tf.transpose(w_0ne_1i), r1_r) - r_npe_pv - r_npe_sst
            # r0_nPe_PV = dampened input + r_nPE_exc - self loop
            input_npe_pv = r0_e + r_npe_e
            # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
            input_npe_sst = r0_e + tf.einsum('ij, jk -> ik', tf.transpose(w_0ne_1i), r1_r) + r_npe_e - r_npe_vip
            # r0_nPe_VIP = pred + r0_nPE_exc
            input_npe_vip = tf.einsum('ij, jk -> ik', tf.transpose(w_0ne_1i), r1_r) + r_npe_e

            r0_e.assign(dr(r=r0_e, inputs=input_r0_e, func=func_type, ei_type='exc'))
            # r0_i.assign(dr(r=r0_i, inputs=input_r0_i, func=func_type))

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

                r1e_mean = tf.reduce_mean(r1_e, axis=1)
                r1i_mean = tf.reduce_mean(r1_i, axis=1)
                r_ppe_e_mean = tf.reduce_mean(r_ppe_e, axis=1)
                r_npe_e_mean = tf.reduce_mean(r_npe_e, axis=1)

                dw_0pe_1e = -lr * tf.einsum('i,j->ij', r1e_mean, r_ppe_e_mean)
                dw_0ne_1i = -lr * tf.einsum('i,j->ij', r1i_mean, r_npe_e_mean)

                w_0pe_1e = np.maximum(w_0pe_1e - dw_0pe_1e + dw_0ne_1i, 0.0)
                w_0ne_1i = w_0pe_1e

                if batch_i + batch_size == len(input_x):
                    # plot progress
                    # (1024, 3)
                    reconstruction_ppe.append(
                        tf.gather(
                            tf.einsum(
                                'ij, jk -> ik', tf.transpose(w_0pe_1e), r1_r
                            ), indices=prg_idx, axis=1
                        )
                    )

    # plot learning progress in the current epoch
    nrow = len(prg_idx)
    ncol = int(T_steps / hebb_window)
    prg_pe_fig, prg_pe_axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(2 * ncol, 2 * nrow))
    # row: class
    for i in range(nrow):
        # col: time
        for j in range(ncol):
            prd_img = prg_pe_axs[i, j].imshow(tf.reshape(reconstruction_ppe[j][:, i], (img_xy, img_xy)), cmap='gray')
            prg_pe_axs[i, j].axis('off')

    prg_pe_fig.subplots_adjust(right=0.8)
    cbar_ax = prg_pe_fig.add_axes([0.85, 0.35, 0.02, 0.3])
    prg_pe_fig.colorbar(prd_img, cax=cbar_ax)

    prg_pe_fig.suptitle(f'TD Prediction @ ep{epoch_i + 1}/{n_epoch}')
    prg_pe_fig.show()

    # save the last reconstruction_ppe to reconstruction_over_epoch
    reconstruction_over_epoch.append(reconstruction_ppe[-1])

    # save SSEs
    psse.append(
        tf.reduce_sum(
            tf.reduce_mean(
                r_ppe_e, axis=1
            ) ** 2
        ).numpy()
    )
    nsse.append(
        tf.reduce_sum(
            tf.reduce_mean(
                r_npe_e, axis=1
            ) ** 2
        ).numpy()
    )

# summary plot

def summarize(fig_handle, fig_ax, ax_row, fr, prg_idx, label, cmap_type, vmin_fig=0, vmax_fig=30):

    img_xy = np.sqrt(fr.shape[0]).astype(int)
    n_img = len(prg_idx)

    img = tf.reshape(tf.gather(fr, prg_idx, axis=1), shape=(img_xy, img_xy, n_img))
    for i in range(n_img):
        inp_plt = fig_ax[ax_row, i].imshow(
            img[:, :, i], cmap=cmap_type, vmin=vmin_fig, vmax=vmax_fig, aspect='auto'
        )
        fig_handle.colorbar(inp_plt, ax=fig_ax[ax_row, i], shrink=0.6)
        fig_ax[ax_row, i].axis('off')
        fig_ax[ax_row, i].set_title(label)

def plot_summary(n_class):

    fig_labels = ['R0', 'pPE', 'nPE', 'Prediction']
    data = [r0_e, r_ppe_e, r_npe_e, tf.transpose(w_0pe_1e) @ r1_r]
    cmaps = ['gray', 'Reds', 'Blues', 'gray']
    sum_fig, sum_axs = plt.subplots(nrows=4, ncols=n_class, figsize=(n_class * 3, 12))
    for ax_i in range(len(data)):
        summarize(sum_fig, sum_axs, ax_i, data[ax_i], prg_idx, fig_labels[ax_i], cmaps[ax_i])

    return sum_fig

sum_fig = plot_summary(n_class)
sum_fig.show()
#
# error plot
pe_fig, pe_axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
pe_axs[0].plot(psse, c='r')
pe_axs[0].set_title('pPE')
pe_axs[1].plot(nsse, c='b')
pe_axs[1].set_title('nPE')

pe_fig.tight_layout()
pe_fig.show()
#
# # # reconsturction plot over training epoch
# # progress_by = int(n_epoch / 10)
# # progress_fig, progress_axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 5))
# # for i, ax in enumerate(progress_axs.flatten()):
# #     img = ax.imshow(reconstruction_ppe[::progress_by][i].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
# #     progress_fig.colorbar(img, ax=ax, shrink=0.3)
# #     ax.set_title(f'iter #{(i + 1) * progress_by}')
# #     ax.set_axis_off()
# # progress_fig.tight_layout()
# # progress_fig.show()
#
# # npred_prg = plot_error_over_time(
# #     w_01n_e @ r1_r, time_int, 'Reds', plt_name='prediction to nPE', plot_xy=img_xy
# # )
# # npred_prg.show()
# # ppe_prg = plot_error_over_time(r_ppe_e, time_int, 'Reds', plt_name='pPE', plot_xy=img_xy)
# # ppe_prg.show()
# # npe_prg = plot_error_over_time(r_npe_e, time_int, 'Blues', plt_name='nPE', plot_xy=img_xy)
# # npe_prg.show()
#
# # why does pPE increase across time? nPE decreases
# # check on the pc of scalar value
# # study the gradient descent