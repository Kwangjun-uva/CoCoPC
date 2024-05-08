import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from parameters import dt, tau_exc, tau_inh, dt, bg_exc, bg_inh
from practice_scalar import jorge, sigmoid, ReLu
import tensorflow as tf
from scipy import sparse
from practice_image_series_isi import oddball_input
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

def update_weight(weight, dw, alpha):

    return np.maximum(weight - dw - alpha * np.maximum(0.0, weight), 0.0)


T = 1.0
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
x_idx = np.random.choice(np.arange(len(x_train)))
input_x = x_train[x_idx].flatten() / 255.0 * max_fr
plt_title = 'fashion mnist sample'

# # grayscale cifar-10
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)), 1)
# input_x = tf.image.rgb_to_grayscale(x_train[x_idx]).numpy().flatten() / 255.0 * max_fr
# plt_title = 'cifar-10 sample'

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
n1 = (int(np.sqrt(n0)) - 8) ** 2
n1i = n1
n_inh = n0

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
# w_r1_ei = create_weights(n1i, n1)
w_r1_ie = 1  # create_weights(n1, n1i)

# self loops
w_self = 1.0

lr = 1e-3
alpha_var = lr * 0.01
n_epoch = 50
hebb_window = T_steps

pmse = []
nmse = []

ppred = []
npred = []

func_type = 'relu'

for epoch_i in range(n_epoch):

    # R0 circuit: n0 = nPixel, n0i = 3
    r0_e = np.zeros((n0, T_steps))
    r0_i = np.zeros((n0i, T_steps))

    # R1 circuit
    r1_e = np.zeros((n1, T_steps))
    r1_i = np.zeros((n1i, T_steps))
    r1_r = np.zeros((n1, T_steps))
    # r1_e[:, 0] = sparse.random(n1, 1, density=0.2).toarray().flatten() * max_fr
    # r1_i[:, 0] = sparse.random(n1i, 1, density=0.2).toarray().flatten() * max_fr
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

    # reconstruction_ppe = {'pv': [], 'sst': [], 'vip': []}
    # reconstruction_npe = {'e': [], 'sst': [], 'vip': []}

    for time_i in trange(T_steps - 1, desc=f'{epoch_i + 1}/{n_epoch}', leave=False):

        # # of len(input_x), every time time_i is a multiple of int((T + isi_time) / dt), increase value of input_idx.
        # input_idx = int(time_i / int((T + isi_time) / dt)) if (T_steps - int(isi_time / dt)) > time_i else 0
        # # If the value is greater or equal to len(input_x), set input_idx to 0.
        # input_idx = input_idx if input_idx < len(input_x) else 0
        # # isi_switch is 1 for int(isi_time / dt) time steps and 0 for the next int(T / dt) time steps. Repeat.
        # isi_switch = int((time_i % int((T + isi_time) / dt)) >= int(isi_time / dt))
        # input_r0_e = input_x[input_idx] * isi_switch

        input_r0_e = input_x  # - w_r0_ie @ r0_i[:, time_i]
        # r0_inh = w_EI @ r0_exc
        input_r0_i = w_r0_ei @ r0_e[:, time_i]

        # r1_exc = pPE - r1_inh (nPE) + self loop
        input_r1_e = weights['0pe_1e'] @ r_ppe_e[:, time_i]  - r1_i[:, time_i] # + r1_e[:, time_i]
        # r1_inh = nPE + r1_exc
        input_r1_i = weights['0ne_1i'] @ r_npe_e[:, time_i]  + r1_e[:, time_i]
        # r1 rep = r1_exc + self loop
        input_r1_r = 1.0 * r1_e[:, time_i] #- 1.0 * r1_i[:, time_i]  # + w_r1_rr * r1_r[:, i]

        # r0_pPe_exc = dampened input - PV - SST + self loop
        input_ppe_e = r0_e[:, time_i] - r_ppe_pv[:, time_i] - r_ppe_sst[:, time_i] #+ r_ppe_e[:, time_i]
        # r0_pPE_PV = pred + r0_pPE_exc - self loop
        input_ppe_pv = weights['01p_pv'] @ r1_r[:, time_i] + w_r0_pyr * r_ppe_e[:, time_i]  # - w_r0_pv * r_ppe_pv[:, i]
        # r0_pPE_SST = pred + r0_pPE_exc - VIP
        input_ppe_sst = weights['01p_sst'] @ r1_r[:, time_i] + w_r0_pyr * r_ppe_e[:, time_i] - r_ppe_vip[:, time_i]
        # r0_pPE_VIP = pred + r0_pPE_exc
        input_ppe_vip = weights['01p_vip'] @ r1_r[:, time_i] + w_r0_pyr * r_ppe_e[:, time_i]

        # r0_nPe_exc = dampened input + pred - PV - SST + self loop
        input_npe_e = r0_e[:, time_i] + weights['01n_e'] @ r1_r[:, time_i] - r_npe_pv[:, time_i] \
                      - r_npe_sst[:, time_i] #+ r_npe_e[:, time_i]
        # r0_nPe_PV = dampened input + r_nPE_exc - self loop
        input_npe_pv = r0_e[:, time_i] + w_r0_pyr * r_npe_e[:, time_i]  # - w_r0_pv * r_npe_pv[:, i]
        # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
        input_npe_sst = r0_e[:, time_i] + weights['01n_sst'] @ r1_r[:, time_i] - r_npe_vip[:, time_i] \
                        + w_r0_pyr * r_npe_e[:, time_i]
        # r0_nPe_VIP = pred + r0_nPE_exc
        input_npe_vip = weights['01n_vip'] @ r1_r[:, time_i] + w_r0_pyr * r_npe_e[:, time_i]

        # r + (-r + fx + bg) * (dt / tau)
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
        pmse.append(np.mean(r_ppe_e[:, time_i + 1]))
        nmse.append(np.mean(r_npe_e[:, time_i + 1]))

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

        if (time_i + 2) % hebb_window == 0:

            time_slice = (time_i + 1 - hebb_window, time_i)

            exc1 = r1_e[:, time_slice].mean(axis=1)
            inh1 = r1_i[:, time_slice].mean(axis=1)
            rep1 = r1_r[:, time_slice].mean(axis=1)

            # pPE, pyr -> rep, e
            dw_0pe_1e = -lr * (np.einsum('i,j->ij', exc1, r_ppe_e[:, time_slice].mean(axis=1).T))
            # rep, r -> pPE, interneurons
            dw_01p_pv = -lr * (np.einsum('i,j->ij', r_ppe_pv[:, time_slice].mean(axis=1).T, rep1))
            dw_01p_sst = -lr * (np.einsum('i,j->ij', r_ppe_sst[:, time_slice].mean(axis=1).T, rep1))
            dw_01p_vip = lr * (np.einsum('i,j->ij', r_ppe_vip[:, time_slice].mean(axis=1).T, rep1))
            # nPE, pyr -> rep, i
            dw_0ne_1i = -lr * (np.einsum('i,j->ij', inh1, r_npe_e[:, time_slice].mean(axis=1).T))
            # rep, r -> nPE, interneurons
            dw_01n_e = lr * (np.einsum('i,j->ij', r_npe_e[:, time_slice].mean(axis=1).T, rep1))
            dw_01n_sst = -lr * (np.einsum('i,j->ij', r_npe_sst[:, time_slice].mean(axis=1).T, rep1))
            dw_01n_vip = lr * (np.einsum('i,j->ij', r_npe_vip[:, time_slice].mean(axis=1).T, rep1))

            # w_0pe_1e = np.maximum(w_0pe_1e - dw_0pe_1e + dw_0ne_1i, 0.0)
            # w_0ne_1i = w_0pe_1e

            weights['0pe_1e'] = update_weight(weight=weights['0pe_1e'], dw=dw_0pe_1e, alpha=alpha_var)
            weights['0ne_1i'] = update_weight(weight=weights['0ne_1i'], dw=dw_0ne_1i, alpha=alpha_var)

            weights['01p_pv'] = update_weight(weight=weights['01p_pv'], dw=dw_01p_pv, alpha=alpha_var)
            weights['01p_sst'] = update_weight(weight=weights['01p_sst'], dw=dw_01p_sst, alpha=alpha_var)
            weights['01p_vip'] = update_weight(weight=weights['01p_vip'], dw=dw_01p_vip, alpha=alpha_var)


            weights['01n_e'] = update_weight(weight=weights['01n_e'], dw=dw_01n_e, alpha=alpha_var)
            weights['01n_sst'] = update_weight(weight=weights['01n_sst'], dw=dw_01n_sst, alpha=alpha_var)
            weights['01n_vip'] = update_weight(weight=weights['01n_vip'], dw=dw_01n_vip, alpha=alpha_var)

            # reconstruction_ppe['pv'].append((w_01p_pv @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            # reconstruction_ppe['sst'].append((w_01p_sst @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            # reconstruction_ppe['vip'].append((w_01p_vip @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            # reconstruction_npe['e'].append((w_01n_e @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            # reconstruction_npe['sst'].append((w_01n_sst @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))
            # reconstruction_npe['vip'].append((w_01n_vip @ r1_r)[:, time_i + 1 - hebb_window:time_i].mean(axis=1))

            # reconstruction_ppe['pv'].append(w_01p_pv @ r1_r[:, time_i + 1])
            # reconstruction_ppe['sst'].append(w_01p_sst @ r1_r[:, time_i + 1])
            # reconstruction_ppe['vip'].append(w_01p_vip @ r1_r[:, time_i + 1])
            # reconstruction_npe['e'].append(w_01n_e @ r1_r[:, time_i + 1])
            # reconstruction_npe['sst'].append(w_01n_sst @ r1_r[:, time_i + 1])
            # reconstruction_npe['vip'].append(w_01n_vip @ r1_r[:, time_i + 1])

            # ppred.append(r_ppe_pv[:, time_i + 1] + r_ppe_sst[:, time_i + 1])
            # npred.append(-w_01n_e @ r1_r[:, time_i + 1] + r_npe_pv[:, time_i + 1] + r_npe_sst[:, time_i + 1])
            #
            # weight plot
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

        if ((epoch_i + 1) % 10 == 0) and (time_i + 2) % T_steps == 0:

            ppred = r_ppe_pv[:, time_i + 1] + r_ppe_sst[:, time_i + 1]
            npred = -weights['01n_e'] @ r1_r[:, time_i + 1] + r_npe_pv[:, time_i + 1] + r_npe_sst[:, time_i + 1]

            pPE = r_ppe_e[:, time_i + 1]
            nPE = r_npe_e[:, time_i + 1]

            ppv_pred = weights['01p_pv'] @ r1_r[:, time_i + 1]
            psst_pred = weights['01p_sst'] @ r1_r[:, time_i + 1]
            pvip_pred = weights['01p_vip'] @ r1_r[:, time_i + 1]

            ne_pred = weights['01n_e'] @ r1_r[:, time_i + 1]
            nsst_pred = weights['01n_sst'] @ r1_r[:, time_i + 1]
            nvip_pred = weights['01n_vip'] @ r1_r[:, time_i + 1]

            # summary plot
            sum_fig, sum_axs = plt.subplots(nrows=4, ncols=3, figsize=(20, 15))
            # [0, 0] input
            inp_plt = sum_axs[0, 0].imshow(r0_e[:, time_i + 1].reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(inp_plt, ax=sum_axs[0, 0], shrink=0.6)
            sum_axs[0, 0].set_title('R0')

            # prediction
            # [0, 1] prediction to pPE
            ppred_plot = sum_axs[0, 1].imshow(ppred.reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(ppred_plot, ax=sum_axs[0, 1], shrink=0.6)
            sum_axs[0, 1].set_title('prediction to pPE')
            # [0, 2] prediction to nPE
            npred_plot = sum_axs[0, 2].imshow(npred.reshape(img_xy, img_xy), cmap='gray')
            sum_fig.colorbar(npred_plot, ax=sum_axs[0, 2], shrink=0.6)
            sum_axs[0, 2].set_title('prediction to nPE')

            # [1, 0] SSE
            err_slice = slice(epoch_i * T_steps, (epoch_i + 1) * T_steps)
            sum_axs[1, 0].plot(pmse[err_slice], color='red', label='PE+')
            sum_axs[1, 0].plot(nmse[err_slice], color='blue', label='PE-')
            sum_axs[1, 0].set_xticks(np.arange(0, len(pmse[err_slice]) + 2, 100))
            sum_axs[1, 0].set_xticklabels(np.arange(0, len(pmse[err_slice]) + 2, 100) + T_steps * epoch_i)
            sum_axs[1, 0].set_ylim([0, 1])
            sum_axs[1, 0].set_title('PE')
            sum_axs[1, 0].legend()

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

            sum_fig.suptitle(f'time = {epoch_i + 1: 02d}')
            sum_fig.tight_layout()
            # # save figure with f't{int(time_i + 1)/hebb_window}' as figname
            # sum_fig.savefig(f'/home/kwangjun/PycharmProjects/si_pc/fig{int(time_i + 1)/hebb_window}.png')
            sum_fig.show()

    if (epoch_i == 0) or ((epoch_i + 1) % 10 == 0):

        plt.close('all')
        plt.subplot(241)
        plt.imshow(np.mean(r_ppe_vip[:, -100:], axis=1).reshape(img_xy, img_xy), cmap='gray', vmin=0, vmax=1)
        plt.title('VIP+')
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.subplot(242)
        plt.imshow(np.mean(r_ppe_e[:, -100:], axis=1).reshape(img_xy, img_xy), cmap='gray', vmin=0, vmax=1)
        plt.title('Pyr +')
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.subplot(243)
        plt.imshow(np.mean(r_ppe_pv[:, -100:], axis=1).reshape(img_xy, img_xy), cmap='gray', vmin=0, vmax=1)
        plt.title('PV+')
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.subplot(244)
        plt.imshow(np.mean(r_ppe_pv[:, -100:], axis=1).reshape(img_xy, img_xy), cmap='gray', vmin=0, vmax=1)
        plt.title('SST+')
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.subplot(245)
        plt.imshow(np.mean(r_npe_vip[:, -100:], axis=1).reshape(img_xy, img_xy), cmap='gray', vmin=0, vmax=1)
        plt.title('VIP-')
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.subplot(246)
        plt.imshow(np.mean(r_npe_e[:, -100:], axis=1).reshape(img_xy, img_xy), cmap='gray', vmin=0, vmax=1)
        plt.title('Pyr-')
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.subplot(247)
        plt.imshow(np.mean(r_npe_pv[:, -100:], axis=1).reshape(img_xy, img_xy), cmap='gray', vmin=0, vmax=1)
        plt.title('PV-')
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.subplot(248)
        plt.imshow(np.mean(r_npe_pv[:, -100:], axis=1).reshape(img_xy, img_xy), cmap='gray', vmin=0, vmax=1)
        plt.title('SST-')
        plt.axis('off')
        plt.colorbar(shrink=0.3)
        plt.suptitle(f'epoch #{epoch_i + 1} / {n_epoch}')
        plt.show()

# error plot
pe_fig, pe_axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
pe_axs[0].plot(pmse, c='r', label='pPE')
pe_axs[1].plot(nmse, c='b', label='nPE')

pe_fig.legend()
pe_fig.tight_layout()
# pe_fig.savefig(f'/home/kwangjun/PycharmProjects/si_pc/pe_summary.png')
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

# # create prediction, pPE and nPE plots in time series (e.g., every 100 ms)
# time_int = 1000
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

# n_plots = T_steps // hebb_window - 1
# n_pred_target = len(reconstruction_ppe) + len(reconstruction_npe)
#
# prg_fig, prg_axs = plt.subplots(nrows=n_pred_target, ncols=n_plots, figsize=(2 * n_plots, 6))
# prg_axs = prg_axs.flatten()
# for j in range(n_plots):
#     for i, (key, grp) in enumerate(reconstruction_ppe.items()):
#         img = prg_axs[(i * n_plots) + j].imshow(grp[j].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=30)
#         prg_fig.colorbar(img, ax=prg_axs[(i * n_plots) + j], shrink=0.3)
#         prg_axs[(i * n_plots) + j].set_title(f'{(j + 1) * hebb_window} ms')
#     for k, (key, grp) in enumerate(reconstruction_npe.items()):
#         img2 = prg_axs[(k + 3) * n_plots + j].imshow(grp[j].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=30)
#         prg_fig.colorbar(img2, ax=prg_axs[(k + 3) * n_plots + j], shrink=0.3)
#         prg_axs[(k + 3) * n_plots + j].set_title(f'{(j + 1) * hebb_window} ms')
#
# for ax in prg_axs:
#     ax.set_axis_off()
#
# prg_fig.tight_layout()
# prg_fig.suptitle('Prediction over time', fontsize=16, y=1.05)
# prg_fig.show()