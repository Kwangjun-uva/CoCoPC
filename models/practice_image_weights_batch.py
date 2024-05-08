import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from parameters import dt
from practice_scalar import dr
import tensorflow as tf
from scipy import sparse
from practice_image_series_isi import oddball_input
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


def oddball_input_generator(dataset, labels, seq_len, fr_max):

    dat_len = len(labels)
    n_repeat = seq_len - 2

    # randomly select two indices of samples of different integer values from labels
    idx1, idx2 = np.random.choice(np.arange(dat_len), 2, replace=False)
    # select the corresponding samples from dataset
    imgs = tf.gather(dataset, [idx1, idx2])
    # convert the images to grayscale
    imgs = tf.image.rgb_to_grayscale(imgs)
    # flatten the images to one-dim
    imgs = tf.reshape(imgs, (imgs.shape[0], imgs.shape[1] * imgs.shape[2]))
    # scale imgs to range [0, fr_max]
    imgs = tf.cast(imgs, dtype=tf.float32) / 255.0 * fr_max
    # create a tensor that repeats imgs[0] n_repeat times
    norm_img_repeat = tf.reshape(tf.tile(imgs[0], [n_repeat]), (n_repeat, imgs.shape[1]))
    # create a tensor named img_seq with norm_img_repeat, imgs[1], and imgs[0] in order
    img_seq = tf.concat([norm_img_repeat, tf.expand_dims(imgs[1], axis=0), tf.expand_dims(imgs[0], axis=0)], 0)

    # plot seq_len images of img_seq in a single plot
    img_xy = int(np.sqrt(imgs.shape[1]))
    fig, axs = plt.subplots(nrows=1, ncols=seq_len, figsize=(seq_len * 3, 3))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(img_seq[i].numpy().reshape(img_xy, img_xy), cmap='gray')
        ax.axis('off')
        ax.set_title(f'sequence #{i + 1}')
    # figure suptitle and tight layout
    fig.suptitle(f'oddball sequence of {seq_len + 1} images')
    fig.show()

    return img_seq, img_xy

T = 3.0
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
# # fashion mnist: lr = 5.0, nepoch=30
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)))
# input_x = x_train[x_idx].flatten() / 255.0 * max_fr
# plt_title = 'fashion mnist sample'

# grayscale cifar-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_idx = np.random.choice(np.arange(len(x_train)), 1)
input_x = tf.image.rgb_to_grayscale(x_train[x_idx]).numpy().flatten() / 255.0 * max_fr
plt_title = 'cifar-10 sample'

img_xy = int(np.sqrt(len(input_x)))
plt.imshow(input_x.reshape(img_xy, img_xy), cmap='gray')
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.title(plt_title)
plt.show()

# sequence_length = 6
# input_x, img_xy = oddball_input_generator(x_train, y_train, seq_len=sequence_length, fr_max=max_fr)
# isi_time = 0.2
# T_steps += (sequence_length - 1) * T_steps + (sequence_length + 1) * int(isi_time / dt)

# network size
n0 = input_x.shape[-1]
n0i = int(n0 / 4)
n1 = (int(np.sqrt(n0)) + 4) ** 2
n1i = n1
n_inh = n0

# R0 circuit: n0 = nPixel, n0i = 3
r0_e = np.zeros((n0,))
r0_i = np.zeros((n0i,))

# R1 circuit
# r1_e = np.zeros((n1,))
# r1_i = np.zeros((n1i,))
r1_r = np.zeros((n1,))
r1_e = sparse.random(n1, 1, density=0.2).toarray().flatten() * max_fr
r1_i = sparse.random(n1i, 1, density=0.2).toarray().flatten() * max_fr
# error circuits
r_pe_r = np.zeros((n0,))

# initialize pPE circuit
r_ppe_e = np.zeros((n0,))
r_ppe_pv = np.zeros((n_inh,))
r_ppe_sst = np.zeros((n_inh,))
r_ppe_vip = np.zeros((n_inh,))

# initialize nPE circuit
r_npe_e = np.zeros((n0,))
r_npe_pv = np.zeros((n_inh,))
r_npe_sst = np.zeros((n_inh,))
r_npe_vip = np.zeros((n_inh,))

# initialize weights

w_r0_pyr = 1.0


weights = {
    '0pe_1e': create_weights(n1, n0),
    '0ne_1i': create_weights(n1, n0),
    '01p_pv': create_weights(n_inh, n1),
    '01p_sst': create_weights(n_inh, n1),
    '01p_vip': create_weights(n_inh, n1),
    '01n_e': create_weights(n0, n1),
    '01n_sst': create_weights(n_inh, n1),
    '01n_vip': create_weights(n_inh, n1)
}

# bottom layer: rep E-I
w_r0_ei = create_weights(n0i, n0)
w_r0_ie = 0
# top layer: rep E-I
w_r1_ei = 1
w_r1_ie = 1  # create_weights(n1, n1i)

# self loops
w_self = 1.0

#
w_r0_pv = 1.0

lr = {'pos': 1e-5, 'neg': 1e-5}
n_epoch = 1
hebb_window = 100
plot_interval = 500

pmse = []
nmse = []

ppred = []
npred = []

func_type = 'relu'

for epoch_i in range(n_epoch):

    reconstruction_ppe = {'pv': [], 'sst': [], 'vip': []}
    reconstruction_npe = {'e': [], 'sst': [], 'vip': []}

    for time_i in trange(T_steps - 1):

        # # of len(input_x), every time time_i is a multiple of int((T + isi_time) / dt), increase value of input_idx.
        # input_idx = int(time_i / int((T + isi_time) / dt)) if (T_steps - int(isi_time / dt)) > time_i else 0
        # # If the value is greater or equal to len(input_x), set input_idx to 0.
        # input_idx = input_idx if input_idx < len(input_x) else 0
        # # isi_switch is 1 for int(isi_time / dt) time steps and 0 for the next int(T / dt) time steps. Repeat.
        # isi_switch = int((time_i % int((T + isi_time) / dt)) >= int(isi_time / dt))
        # input_r0_e = input_x[input_idx] * isi_switch

        input_r0_e = input_x  # - w_r0_ie @ r0_i[:, time_i]
        # r0_inh = w_EI @ r0_exc
        input_r0_i = w_r0_ei @ r0_e

        # r1_exc = pPE - r1_inh (nPE) + self loop
        input_r1_e = weights['0pe_1e'] @ r_ppe_e - r1_i  # + w_r1_ee * r1_e[:, i]
        # r1_inh = nPE + r1_exc
        input_r1_i = weights['0ne_1i'] @ r_npe_e #+ w_r1_ei * r1_e - r1_i
        # r1 rep = r1_exc + self loop
        input_r1_r = r1_e  # + w_r1_rr * r1_r[:, i]

        # r0_pPe_exc = dampened input - PV - SST + self loop
        input_ppe_e = r0_e - r_ppe_pv - r_ppe_sst #+ w_r0_pyr * r_ppe_e
        # r0_pPE_PV = pred + r0_pPE_exc - self loop
        input_ppe_pv = weights['01p_pv'] @ r1_r #+ w_r0_pyr * r_ppe_e - w_r0_pv * r_ppe_pv
        # r0_pPE_SST = pred + r0_pPE_exc - VIP
        input_ppe_sst = weights['01p_sst'] @ r1_r - r_ppe_vip #+ w_r0_pyr * r_ppe_e
        # r0_pPE_VIP = pred + r0_pPE_exc
        input_ppe_vip = weights['01p_vip'] @ r1_r #+ w_r0_pyr * r_ppe_e

        # r0_nPe_exc = dampened input + pred - PV - SST + self loop
        input_npe_e = r0_e + weights['01n_e'] @ r1_r - r_npe_pv - r_npe_sst #+ w_r0_pyr * r_npe_e
        # r0_nPe_PV = dampened input + r_nPE_exc - self loop
        input_npe_pv = r0_e #+ w_r0_pyr * r_npe_e - w_r0_pv * r_npe_pv
        # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
        input_npe_sst = r0_e + weights['01n_sst'] @ r1_r - r_npe_vip #+ w_r0_pyr * r_npe_e
        # r0_nPe_VIP = pred + r0_nPE_exc
        input_npe_vip = weights['01n_vip'] @ r1_r #+ w_r0_pyr * r_npe_e

        # r + (-r + fx + bg) * (dt / tau)
        r0_e = dr(r=r0_e, inputs=input_r0_e, func=func_type, ei_type='exc')
        r0_i = dr(r=r0_i, inputs=input_r0_i, func=func_type)

        r1_e = dr(r=r1_e, inputs=input_r1_e, func=func_type, ei_type='exc')
        r1_i = dr(r=r1_i, inputs=input_r1_i/2, func=func_type)
        r1_r = dr(r=r1_r, inputs=input_r1_r, func=func_type, ei_type='exc')

        r_ppe_e = dr(r=r_ppe_e, inputs=input_ppe_e, func=func_type, ei_type='exc')
        r_ppe_pv = dr(r=r_ppe_pv, inputs=input_ppe_pv, func=func_type)
        r_ppe_sst = dr(r=r_ppe_sst, inputs=input_ppe_sst, func=func_type)
        r_ppe_vip = dr(r=r_ppe_vip, inputs=input_ppe_vip, func=func_type)

        r_npe_e = dr(r=r_npe_e, inputs=input_npe_e, func=func_type, ei_type='exc')
        r_npe_pv = dr(r=r_npe_pv, inputs=input_npe_pv, func=func_type)
        r_npe_sst = dr(r=r_npe_sst, inputs=input_npe_sst, func=func_type)
        r_npe_vip = dr(r=r_npe_vip, inputs=input_npe_vip, func=func_type)

        # record error
        pmse.append(np.mean(r_ppe_e ** 2))
        nmse.append(np.mean(r_npe_e ** 2))

        # if np.mean(pmse[-10:]) < np.mean(pmse[-20:-10]):
        #     lr['pos'] *= 0.1
        # elif np.mean(nmse[-10:]) < np.mean(nmse[-20:-10]):
        #     lr['neg'] *= 0.1
        # else:
        #     pass
        # # input sequence test
        # if (time_i + 1) % 1200 == 0:
        #     plt.imshow(r0_e[:, time_i-1000:time_i].mean(axis=1).reshape(img_xy, img_xy), cmap='gray')
        #     plt.axis('off')
        #     plt.title('RO at time: ' + str(time_i + 1))
        #     plt.show()

        if (time_i + 1) % hebb_window == 0:

            exc1 = r1_e
            inh1 = r1_i
            rep1 = r1_r

            dw_0pe_1e = lr['pos'] * (np.einsum('i,j->ij', exc1, r_ppe_e.T))
            dw_0ne_1i = lr['neg'] * (np.einsum('i,j->ij', inh1, r_npe_e.T))

            dw_01p_pv = lr['pos'] * (np.einsum('i,j->ij', r_ppe_pv.T, rep1))
            dw_01p_sst = lr['pos'] * (np.einsum('i,j->ij', r_ppe_sst.T, rep1))
            dw_01p_vip = lr['pos'] * (np.einsum('i,j->ij', r_ppe_vip.T, rep1))

            dw_01n_e = lr['neg'] * (np.einsum('i,j->ij', r_npe_e.T, rep1))
            dw_01n_sst = lr['neg'] * (np.einsum('i,j->ij', r_npe_sst.T, rep1))
            dw_01n_vip = lr['neg'] * (np.einsum('i,j->ij', r_npe_vip.T, rep1))

            # w_0pe_1e = np.maximum(w_0pe_1e - dw_0pe_1e + dw_0ne_1i, 0.0)
            # w_0ne_1i = w_0pe_1e
            weights['0pe_1e'] = np.maximum(weights['0pe_1e'] - dw_0pe_1e, 0.0)
            weights['0ne_1i'] = np.maximum(weights['0ne_1i'] - dw_0ne_1i, 0.0)

            weights['01p_pv'] = np.maximum(weights['01p_pv'] - dw_01p_pv, 0.0)
            weights['01p_sst'] = np.maximum(weights['01p_sst'] - dw_01p_sst, 0.0)
            weights['01p_vip'] = np.maximum(weights['01p_vip'] - dw_01p_vip, 0.0)

            weights['01n_e'] = np.maximum(weights['01n_e'] - dw_01n_e, 0.0)
            weights['01n_sst'] = np.maximum(weights['01n_sst'] - dw_01n_sst, 0.0)
            weights['01n_vip'] = np.maximum(weights['01n_vip'] - dw_01n_vip, 0.0)

            # w_fig, w_axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
            # w_axs = w_axs.flatten()
            # for item_i, (key, grp) in enumerate(weights.items()):
            #     w_img = w_axs[item_i].imshow(grp)
            #     w_fig.colorbar(w_img, ax=w_axs[item_i], shrink=0.6)
            #     w_axs[item_i].set_title(f'{key}')
            #
            # w_fig.suptitle(f't={time_i + 1}')
            # w_fig.tight_layout()
            # w_fig.show()

        if (time_i + 1) % plot_interval == 0:

            ppred = -r_ppe_pv - r_ppe_sst
            npred = weights['01n_e'] @ r1_r - r_npe_pv - r_npe_sst

            pPE = r_ppe_e
            nPE = r_npe_e

            ppv_pred = weights['01p_pv'] @ r1_r
            psst_pred = weights['01p_sst'] @ r1_r
            pvip_pred = weights['01p_vip'] @ r1_r

            ne_pred = weights['01n_e'] @ r1_r
            nsst_pred = weights['01n_sst'] @ r1_r
            nvip_pred = weights['01n_vip'] @ r1_r

            # summary plot
            sum_fig, sum_axs = plt.subplots(nrows=4, ncols=3, figsize=(20, 15))
            # [0, 0] input
            inp_plt = sum_axs[0, 0].imshow(r0_e.reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(inp_plt, ax=sum_axs[0, 0], shrink=0.6)
            sum_axs[0, 0].set_title('R0')

            # prediction
            # [0, 1] prediction to pPE
            ppred_plot = sum_axs[0, 1].imshow(ppred.reshape(img_xy, img_xy), cmap='gray_r')
            sum_fig.colorbar(ppred_plot, ax=sum_axs[0, 1], shrink=0.6)
            sum_axs[0, 1].set_title('prediction to pPE')
            # [0, 2] prediction to nPE
            npred_plot = sum_axs[0, 2].imshow(npred.reshape(img_xy, img_xy), cmap='gray_r')
            sum_fig.colorbar(npred_plot, ax=sum_axs[0, 2], shrink=0.6)
            sum_axs[0, 2].set_title('prediction to nPE')

            # [1, 0] SSE
            sum_axs[1, 0].plot(pmse, color='red')
            sum_axs[1, 0].plot(nmse, color='blue')
            for axvi in range(len(pmse)):

            sum_axs[1, 0].axvline()
            sum_axs[1, 0].set_title('PE')

            # [1, 1] pPE
            ppe_plt = sum_axs[1, 1].imshow(pPE.reshape(img_xy, img_xy), cmap='Reds')
            sum_fig.colorbar(ppe_plt, ax=sum_axs[1, 1], shrink=0.6)
            sum_axs[1, 1].set_title('pPE')
            # [1, 2] nPE
            npe_plt = sum_axs[1, 2].imshow(nPE.reshape(img_xy, img_xy), cmap='Blues')
            sum_fig.colorbar(npe_plt, ax=sum_axs[1, 2], shrink=0.6)
            sum_axs[1, 2].set_title('nPE')

            # prediction to pPE: PV
            ppv_pred_plot = sum_axs[2, 0].imshow(ppv_pred.reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(ppv_pred_plot, ax=sum_axs[2, 0], shrink=0.6)
            sum_axs[2, 0].set_title('Prediction to pv+')
            # prediction to pPE: SST
            psst_pred_plot = sum_axs[2, 1].imshow(psst_pred.reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(psst_pred_plot, ax=sum_axs[2, 1], shrink=0.6)
            sum_axs[2, 1].set_title('Prediction to sst+')
            # prediction to pPE circuit: VIP
            pvip_pred_plot = sum_axs[2, 2].imshow(pvip_pred.reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(pvip_pred_plot, ax=sum_axs[2, 2], shrink=0.6)
            sum_axs[2, 2].set_title('Prediction to vip+')

            # prediction to nPE circuit: e
            ne_pred_plot = sum_axs[3, 0].imshow(ne_pred.reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(ne_pred_plot, ax=sum_axs[3, 0], shrink=0.6)
            sum_axs[3, 0].set_title('Prediction to e-')
            # prediction to nPE circuit: SST
            nsst_pred_plot = sum_axs[3, 1].imshow(nsst_pred.reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(nsst_pred_plot, ax=sum_axs[3, 1], shrink=0.6)
            sum_axs[3, 1].set_title('Prediction to sst-')
            # prediction to nPE circuit: VIP
            nvip_pred_plot = sum_axs[3, 2].imshow(nvip_pred.reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(nvip_pred_plot, ax=sum_axs[3, 2], shrink=0.6)
            sum_axs[3, 2].set_title('Prediction to vip-')

            for ax_i in sum_axs.flatten():
                if ax_i is not sum_axs[1, 0]:
                    ax_i.axis('off')

            sum_fig.suptitle(f'fig{int(time_i + 1)//100: 02d}')
            sum_fig.tight_layout()
            # # save figure with f't{int(time_i + 1)/hebb_window}' as figname
            # sum_fig.savefig(f'/home/kwangjun/PycharmProjects/si_pc/fig{int(time_i + 1)/hebb_window}.png')
            sum_fig.show()


# error plot
pe_fig, pe_axs = plt.subplots(nrows=1, ncols=1, sharex='all', sharey='all')
pe_axs.plot(pmse, c='r', label='pPE')
pe_axs.plot(nmse, c='b', label='nPE')

pe_fig.legend()
pe_fig.tight_layout()
# pe_fig.savefig(f'/home/kwangjun/PycharmProjects/si_pc/pe_summary.png')
pe_fig.show()