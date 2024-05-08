import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from parameters import tau_exc, tau_inh, bg_exc, bg_inh, dt
import tensorflow as tf
import os
from scipy.sparse import random as sprandn
from scipy.stats import norm

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
        return tf.random.uniform((target, source)) / np.sqrt(source)


def jorge(x, d=10.0, theta=dt * 0.1):
    return (x - theta) / (1 - tf.exp(-d * (x - theta)))

def ReLu(x, theta=0):
    new_x = x - theta

    return np.maximum(new_x, 0)

def dr(r, inputs, func, ei_type='inh'):
    if ei_type == 'exc':
        tau = tau_exc
        bg = bg_exc
    elif ei_type == 'inh':
        tau = tau_inh
        bg = bg_inh

    if func == 'jorge':
        fx = jorge(x=inputs)
    elif func == 'relu':
        fx = ReLu(x=inputs)
    else:
        raise 'func not implemented yet!'

    return r + (-r + fx + bg) * (dt / tau)


# def oddball_input(x, y, n_repeat):
#     norm_img_repeat = tf.reshape(tf.tile(x, [n_repeat, 1]), (n_repeat, 32, 32))
#
#     return tf.concat([norm_img_repeat, tf.expand_dims(y, axis=0)], 0)

def oddball_input(img1, img2, n_repeat):

    oddball_inputs = tf.constant(np.array([img1] * n_repeat + [img2] + [img1]), dtype=tf.float32)

    n_imgs, nx, ny = oddball_inputs.shape
    # img_dim = int(np.sqrt(n_pixel))

    odd_fig, odd_axs = plt.subplots(nrows=1, ncols=n_imgs, sharex='all', sharey='all')
    for seq_i, ax in enumerate(odd_axs.flatten()):
        ax.imshow(oddball_inputs[seq_i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'seq #{seq_i + 1}/{n_repeat + 2}')
    odd_fig.suptitle('Oddball stimulus sequence')
    odd_fig.tight_layout()

    return oddball_inputs, odd_fig

if __name__ == '__main__':

    n_img = 4
    T_per_img = 3.0
    T_per_isi = 0.1
    steps_per_img = int(T_per_img / dt)
    steps_per_isi = int(T_per_isi / dt)
    steps_per_seq = steps_per_img + steps_per_isi
    steps_total = steps_per_seq * n_img
    # T_img = n_img * T_per_img
    # n_isi = n_img
    # T_isi = n_isi * T_per_isi

    # T_total = n_img * (T_per_img + T_per_isi)
    # T_steps = int(T_total / dt)
    # T_steps_per_img = int((T_per_img + T_per_isi) / dt)

    # input
    max_fr = 1.0
    # # X (3x3)
    # input_x = np.zeros(9)
    # input_x[::2] = max_fr
    # plt_title = 'example X shape'

    # # mnist: lr = 5.0
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_idx = np.random.choice(np.arange(len(x_train)), n_img)
    # input_x = tf.constant(x_train[x_idx] / 255.0 * max_fr, dtype=tf.float32)
    # plt_title = 'mnist digit'
    #
    # # fashion mnist: lr = 5.0, nepoch=30
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # x_idx = np.random.choice(np.arange(len(x_train)), n_img)
    # input_x = x_train[x_idx] / 255.0 * max_fr
    # plt_title = 'fashion mnist sample'

    # grayscale cifar-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    ran_cls = np.random.choice(np.arange(10), n_img, replace=False)
    x_idx = np.zeros(n_img).astype(int)

    for i, val in enumerate(ran_cls):
        x_idx[i] = np.where(y_train == val)[0][0]
    # x_idx = np.random.choice(np.arange(len(x_train)), n_img) #[25020, 21551]
    input_x = tf.squeeze(tf.image.rgb_to_grayscale(x_train[x_idx]), -1) / tf.reduce_max(x_train) * max_fr
    plt_title = 'cifar-10 sample'

    oddball_paradigm = False
    img_xy = input_x.shape[-1]
    if oddball_paradigm:
        # input_x = oddball_input(input_x[0], input_x[1], n_img - 1)
        input_x, inp_fig = oddball_input(img1=input_x[0], img2=input_x[1], n_repeat=n_img - 2)
    else:
        inp_fig, inp_axs = plt.subplots(1, 4)
        for i, ax in enumerate(inp_axs.flat):
            ax.imshow(input_x[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f'img #{i + 1}')

    # img_xy = input_x.shape[-1]
    #
    # inp_fig, inp_axs = plt.subplots(n_img, 1, figsize=(3, 3 * n_img))
    # for i, ax in enumerate(inp_axs.flatten()):
    #     inp_img = ax.imshow(input_x[i], cmap='gray')
    #     inp_fig.colorbar(inp_img, ax=ax, shrink=0.6)
    #     ax.axis('off')
    #     ax.set_title(f'input #{i + 1}')
    # inp_fig.suptitle(plt_title)
    # inp_fig.tight_layout()
    inp_fig.show()

    # network size
    n0 = np.product(input_x.shape[1:])
    n0i = int(n0 / 4)
    n1 = 28 ** 2
    n1i = n1
    n_inh = n0

    # R0 circuit: n0 = nPixel, n0i = 3
    r0_e = tf.Variable(tf.zeros(shape=(n0, ), dtype=tf.float32))
    r0_i = tf.Variable(tf.zeros(shape=(n0i, ), dtype=tf.float32))

    # R1 circuit
    # rvs_e = tf.constant(sprandn(n1, 1, density=0.2, data_rvs=norm(loc=10, scale=5).rvs).A.flatten(), dtype=tf.float32)
    # rvs_i = tf.constant(sprandn(n1, 1, density=0.2, data_rvs=norm(loc=10, scale=5).rvs).A.flatten(), dtype=tf.float32)
    # r1_e = tf.Variable(rvs_e, dtype=tf.float32)
    # r1_i = tf.Variable(rvs_i, dtype=tf.float32)
    r1_e = tf.Variable(tf.zeros(shape=(n1, ), dtype=tf.float32))
    r1_i = tf.Variable(tf.zeros(shape=(n1i, ), dtype=tf.float32))
    r1_r = tf.Variable(tf.zeros(shape=(n1, ), dtype=tf.float32))

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

    lr = 1e-1
    n_epoch = 10
    hebb_window = 200

    psse = []
    nsse = []
    inputs = []
    reconstruction = []
    ppe = []
    npe = []

    func_type = 'relu'

    # ##
    # psse2 = []
    # nsse2 = []
    # inputs2 = []
    # reconstruction2 = []
    #
    # n2 = 20 ** 2
    #
    # # R2 circuit
    # e2 = tf.constant(sprandn(n2, 1, density=0.2, data_rvs=norm(loc=10, scale=5).rvs).A.flatten(), dtype=tf.float32)
    # i2 = tf.constant(sprandn(n2, 1, density=0.2, data_rvs=norm(loc=10, scale=5).rvs).A.flatten(), dtype=tf.float32)
    # r2_e = tf.Variable(e2, dtype=tf.float32)
    # r2_i = tf.Variable(i2, dtype=tf.float32)
    # r2_r = tf.Variable(tf.zeros(shape=(n2,), dtype=tf.float32))
    #
    # # initialize pPE circuit
    # r1_ppe_e = tf.Variable(tf.zeros(shape=(n1,), dtype=tf.float32))
    # r1_ppe_pv = tf.Variable(tf.zeros(shape=(n1i,), dtype=tf.float32))
    # r1_ppe_sst = tf.Variable(tf.zeros(shape=(n1i,), dtype=tf.float32))
    # r1_ppe_vip = tf.Variable(tf.zeros(shape=(n1i,), dtype=tf.float32))
    #
    # # initialize nPE circuit
    # r1_npe_e = tf.Variable(tf.zeros(shape=(n1,), dtype=tf.float32))
    # r1_npe_pv = tf.Variable(tf.zeros(shape=(n1i,), dtype=tf.float32))
    # r1_npe_sst = tf.Variable(tf.zeros(shape=(n1i,), dtype=tf.float32))
    # r1_npe_vip = tf.Variable(tf.zeros(shape=(n1i,), dtype=tf.float32))
    #
    # # bottom-up: pPE pyr -> rep
    # w_1pe_2e = create_weights(n2, n1)
    # w_1ne_2i = w_1pe_2e

    for epoch_i in range(n_epoch):

        for time_i in trange(steps_total):

            # series of images with isi

            input_idx = time_i // steps_per_seq if time_i < (steps_total - steps_per_isi) else n_img - 1
            isi_var = 1 if time_i % steps_per_seq >= steps_per_isi else 0
            input_r0_e = tf.reshape(input_x[input_idx], np.product(input_x[input_idx].shape)) * isi_var

            # r1_exc = pPE - r1_inh (nPE) + self loop
            input_r1_e = tf.einsum('ij, j -> i', w_0pe_1e, r_ppe_e) - r1_i #+ r1_npe_e
            # r1_inh = nPE + r1_exc
            input_r1_i = tf.einsum('ij, j -> i', w_0ne_1i, r_npe_e) #+ r1_ppe_e
            # r1 rep = r1_exc + self loop
            input_r1_r = r1_e

            # r0_pPe_exc = dampened input + pyr (self loop) - PV - SST
            input_ppe_e = r0_e + r_ppe_e - r_ppe_pv - r_ppe_sst
            # r0_pPE_PV = pred + r0_pPE_exc + pyr
            input_ppe_pv = tf.einsum('ij, j -> i', tf.transpose(w_0pe_1e), r1_r) + r_ppe_e
            # r0_pPE_SST = pred + pyr - VIP
            input_ppe_sst = tf.einsum('ij, j -> i', tf.transpose(w_0pe_1e), r1_r) + r_ppe_e - r_ppe_vip
            # r0_pPE_VIP = pred + pyr
            input_ppe_vip = tf.einsum('ij, j -> i', tf.transpose(w_0pe_1e), r1_r) + r_ppe_e

            # r0_nPe_exc = dampened input + pyr (self loop) + pred - PV - SST
            input_npe_e = r0_e + r_npe_e + tf.einsum('ij, j -> i', tf.transpose(w_0ne_1i), r1_r) - r_npe_pv - r_npe_sst
            # r0_nPe_PV = dampened input + pyr
            input_npe_pv = r0_e + r_npe_e
            # r0_nPe_SST = dampened input + pred + pyr - VIP
            input_npe_sst = r0_e + tf.einsum('ij, j -> i', tf.transpose(w_0ne_1i), r1_r) + r_npe_e - r_npe_vip
            # r0_nPe_VIP = pred + pyr
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

            psse.append(
                tf.reduce_mean(
                    r_ppe_e.numpy()
                ).numpy()
            )
            nsse.append(
                tf.reduce_mean(
                    r_npe_e.numpy()
                ).numpy()
            )

            # ##
            # # r1_exc = pPE - r1_inh (nPE) + self loop
            # input_r2_e = tf.einsum('ij, j -> i', w_1pe_2e, r1_ppe_e) - r2_i
            # # r1_inh = nPE + r1_exc
            # input_r2_i = tf.einsum('ij, j -> i', w_1ne_2i, r1_npe_e)
            # # r1 rep = r1_exc + self loop
            # input_r2_r = r2_e
            #
            # # r0_pPe_exc = dampened input - PV - SST + self loop
            # input_r1_ppe_e = r1_e + r1_ppe_e - r1_ppe_pv - r1_ppe_sst
            # # r0_pPE_PV = pred + r0_pPE_exc - self loop
            # input_r1_ppe_pv = tf.einsum('ij, j -> i', tf.transpose(w_1pe_2e), r2_r) + r1_ppe_e
            # # r0_pPE_SST = pred + r0_pPE_exc - VIP
            # input_r1_ppe_sst = tf.einsum('ij, j -> i', tf.transpose(w_1pe_2e), r2_r) + r1_ppe_e - r1_ppe_vip
            # # r0_pPE_VIP = pred + r0_pPE_exc
            # input_r1_ppe_vip = tf.einsum('ij, j -> i', tf.transpose(w_1pe_2e), r2_r) + r1_ppe_e
            #
            # # r0_nPe_exc = dampened input + pred - PV - SST + self loop
            # input_r1_npe_e = r1_e + r1_npe_e + tf.einsum('ij, j -> i', tf.transpose(w_1ne_2i), r2_r) - r1_npe_pv - r1_npe_sst
            # # r0_nPe_PV = dampened input + r_nPE_exc - self loop
            # input_r1_npe_pv = r1_e + r1_npe_e
            # # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
            # input_r1_npe_sst = r1_e + tf.einsum('ij, j -> i', tf.transpose(w_1ne_2i), r2_r) + r1_npe_e - r1_npe_vip
            # # r0_nPe_VIP = pred + r0_nPE_exc
            # input_r1_npe_vip = tf.einsum('ij, j -> i', tf.transpose(w_1ne_2i), r2_r) + r1_npe_e
            #
            # r2_e.assign(dr(r=r2_e, inputs=input_r2_e, func=func_type, ei_type='exc'))
            # r2_i.assign(dr(r=r2_i, inputs=input_r2_i, func=func_type))
            # r2_r.assign(dr(r=r2_r, inputs=input_r2_r, func=func_type, ei_type='exc'))
            #
            # r1_ppe_e.assign(dr(r=r1_ppe_e, inputs=input_r1_ppe_e, func=func_type, ei_type='exc'))
            # r1_ppe_pv.assign(dr(r=r1_ppe_pv, inputs=input_r1_ppe_pv, func=func_type))
            # r1_ppe_sst.assign(dr(r=r1_ppe_sst, inputs=input_r1_ppe_sst, func=func_type))
            # r1_ppe_vip.assign(dr(r=r1_ppe_vip, inputs=input_r1_ppe_vip, func=func_type))
            #
            # r1_npe_e.assign(dr(r=r1_npe_e, inputs=input_r1_npe_e, func=func_type, ei_type='exc'))
            # r1_npe_pv.assign(dr(r=r1_npe_pv, inputs=input_r1_npe_pv, func=func_type))
            # r1_npe_sst.assign(dr(r=r1_npe_sst, inputs=input_r1_npe_sst, func=func_type))
            # r1_npe_vip.assign(dr(r=r1_npe_vip, inputs=input_r1_npe_vip, func=func_type))
            #
            # psse2.append(
            #     tf.reduce_mean(
            #         r1_ppe_e ** 2
            #     ).numpy()
            # )
            # nsse2.append(
            #     tf.reduce_mean(
            #         r1_npe_e ** 2
            #     ).numpy()
            # )

            if (time_i + 1) % hebb_window == 0:
                # print(f'time={time_i + 1} ms, input={input_idx}, isi_switch={isi_var}')

                dw_0pe_1e = -lr * tf.einsum('i,j->ij', r1_r, r_ppe_e) * isi_var
                dw_0ne_1i = -lr * tf.einsum('i,j->ij', r1_r, r_npe_e) * isi_var

                w_0pe_1e = tf.maximum(w_0pe_1e - dw_0pe_1e + dw_0ne_1i, 0.0)
                w_0ne_1i = w_0pe_1e

                reconstruction.append(
                    tf.einsum(
                        'ij,j-> i',
                        tf.transpose(w_0pe_1e),
                        r1_r
                    )
                )

                inputs.append(
                    tf.reshape(r0_e, shape=(img_xy, img_xy))
                )

                ppe.append(
                    r_ppe_e
                )
                npe.append(
                    r_npe_e
                )

                # ##
                # dw_1pe_2e = -lr * tf.einsum('i,j->ij', r2_r, r1_ppe_e) * isi_var
                # dw_1ne_2i = -lr * tf.einsum('i,j->ij', r2_r, r1_npe_e) * isi_var
                #
                # w_1pe_2e = tf.maximum(w_1pe_2e - dw_1pe_2e + dw_1ne_2i, 0.0)
                # w_1ne_2i = w_1pe_2e
                #
                # reconstruction2.append(
                #     tf.einsum(
                #         'ij,j-> i',
                #         tf.transpose(w_1pe_2e),
                #         r2_r
                #     )
                # )
                # inputs2.append(
                #     tf.reshape(r1_e, shape=(25, 25))
                # )


            if time_i in [(i+1) * steps_per_seq - 1 for i in range(n_img)]:

                # print(f'input={input_idx}, isi_switch={isi_var}')

                # dw_0pe_1e = -lr * tf.einsum('i,j->ij', r1_e, r_ppe_e)
                # dw_0ne_1i = -lr * tf.einsum('i,j->ij', r1_i, r_npe_e)
                #
                # w_0pe_1e = tf.maximum(w_0pe_1e - dw_0pe_1e + dw_0ne_1i, 0.0)
                # w_0ne_1i = w_0pe_1e

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

                sum_fig.suptitle(f'Epoch #{epoch_i + 1}/{n_epoch}, img #{input_idx}/{n_img}')
                sum_fig.tight_layout()
                sum_fig.show()

                # ##
                # # summary plot
                # sum_fig, sum_axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 15))
                # # input
                # r1_dim = np.sqrt(r1_e.shape[0]).astype(int)
                # inp_img = tf.reshape(r1_e, shape=(r1_dim, r1_dim))
                # inp_plt = sum_axs[0].imshow(inp_img, cmap='gray')
                # sum_fig.colorbar(inp_plt, ax=sum_axs[0], shrink=0.6)
                # sum_axs[0].axis('off')
                # sum_axs[0].set_title('R0')
                # # pPE
                # ppe_img = tf.reshape(r1_ppe_e, shape=(r1_dim, r1_dim))
                # ppe_plt = sum_axs[1].imshow(ppe_img, cmap='Reds')
                # sum_fig.colorbar(ppe_plt, ax=sum_axs[1], shrink=0.6)
                # sum_axs[1].axis('off')
                # sum_axs[1].set_title('pPE')
                # # nPE
                # npe_img = tf.reshape(r1_npe_e, shape=(r1_dim, r1_dim))
                # npe_plt = sum_axs[2].imshow(npe_img, cmap='Blues')
                # sum_fig.colorbar(npe_plt, ax=sum_axs[2], shrink=0.6)
                # sum_axs[2].axis('off')
                # sum_axs[2].set_title('nPE')
                # # prediction to pPE circuit
                # pred_sig = tf.einsum('ij,j-> i', tf.transpose(w_1pe_2e), r2_r)
                # ppe_pred = sum_axs[3].imshow(tf.reshape(pred_sig, shape=(r1_dim, r1_dim)), cmap='gray')
                # sum_fig.colorbar(ppe_pred, ax=sum_axs[3], shrink=0.6)
                # sum_axs[3].axis('off')
                # sum_axs[3].set_title('Prediction to pPE')
                # # error-corrected
                # corrected = sum_axs[4].imshow(tf.reshape(pred_sig + r1_ppe_e - r1_npe_e,
                #                                          shape=(r1_dim, r1_dim)), cmap='Greens')
                # sum_fig.colorbar(corrected, ax=sum_axs[4], shrink=0.6)
                # sum_axs[4].axis('off')
                # sum_axs[4].set_title('corrected')
                #
                # sum_fig.suptitle(
                #     f'Epoch #{epoch_i + 1}/{n_epoch}, img #{int((time_i + 1) / (T_steps / 3)) + 1}/{n_img}')
                # sum_fig.tight_layout()
                # sum_fig.show()

    # error plot
    mse = np.array(psse).mean(axis=1) + np.array(nsse).mean(axis=1)
    isi_steps = int(T_per_isi / dt)
    steps_per_img = int((T_per_isi + T_per_img) / dt)
    plastic_t = np.array([i * steps_per_img + j
                          for i in range(n_img)
                          for j in np.arange(isi_steps, steps_per_img + 1, hebb_window)][1:])
    plt.figure(figsize=(15, 2.5))
    plt.plot(mse)
    for i in range(n_img + 1):
        img_x = i * steps_per_img
        isi_x = img_x + steps_per_isi
        plt.axvline(x=img_x, ls='--', c='black')
        plt.axvline(x=isi_x, ls='--', c='black')
    plt.scatter(plastic_t, np.zeros(plastic_t.shape), marker='o', c='r', s=1)
    plt.show()

    # reconstruction progress plot
    reduce_by = 1
    n_col = int((steps_total // hebb_window) / reduce_by)
    # n_plt = n_col * n_epoch

    fig, axs = plt.subplots(nrows=2, ncols=n_col, sharex='all', sharey='all', figsize= (n_col * 2, 4))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i < n_col:
            ax.imshow(inputs[::reduce_by][i], vmin=0, vmax=max_fr, cmap='gray')
            ax.set_title(f'{(i + 1) * hebb_window * reduce_by} ms')
        else:
            ax.imshow(tf.reshape(reconstruction[::reduce_by][i - n_col], shape=(img_xy, img_xy)), vmin=0, vmax=max_fr, cmap='gray')
            ax.set_title(f'{(i - n_col + 1) * hebb_window * reduce_by} ms')
        ax.axis('off')

    fig.tight_layout()
    fig.show()
    # fig.savefig('/home/kwangjun/Pictures/progress.png', dpi=300)

    # ##
    # # error plot
    # mse2 = np.array(psse2) + np.array(nsse2)
    # plt.figure(figsize=(15, 2.5))
    # plt.plot(mse2)
    # for i in range(n_img):
    #     img_x = i * 1000
    #     isi_x = img_x + 300
    #     plt.axvline(x=img_x, ls='--', c='black')
    #     plt.axvline(x=isi_x, ls='--', c='black')
    # plt.scatter(plastic_t, np.zeros(plastic_t.shape), marker='o', c='r', s=1)
    # plt.show()

    # # reconstruction progress plot
    # reduce_by = 2
    # n_col = int((T_steps // hebb_window) / reduce_by)
    # # n_plt = n_col * n_epoch
    #
    # fig, axs = plt.subplots(nrows=2, ncols=n_col, sharex='all', sharey='all', figsize=(n_col * 2, 4))
    # axs = axs.flatten()
    # for i, ax in enumerate(axs):
    #     if i < n_col:
    #         ax.imshow(inputs2[::reduce_by][i], vmin=0, vmax=max_fr, cmap='gray')
    #         ax.set_title(f'{(i + 1) * hebb_window * reduce_by} ms')
    #     else:
    #         ax.imshow(tf.reshape(reconstruction2[::reduce_by][i - n_col], shape=(32, 32)), vmin=0, vmax=max_fr,
    #                   cmap='gray')
    #         ax.set_title(f'{(i - n_col + 1) * hebb_window * reduce_by} ms')
    #     ax.axis('off')
    #
    # fig.tight_layout()
    # fig.show()


    # why does pPE increase across time? nPE decreases
    # check on the pc of scalar value
    # study the gradient descent

    fig, axs = plt.subplots(n_img, 4)
    for i in range(n_img):
        axs[i, 0].imshow()