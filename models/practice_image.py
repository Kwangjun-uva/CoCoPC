import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from parameters import dt
from practice_scalar import dr, ReLu
import tensorflow as tf

def drelu(x):
    return np.maximum(np.sign(x), 0)

T = 1.0
T_steps = int(T / dt)
hebb_window = 1 #int(T_steps * 0.5)

# input
max_fr = 2.0
n_img = 1
# # X (3x3)
# input_x = np.zeros(9)
# input_x[::2] = max_fr
# plt_title = 'example X shape'

# # mnist: lr = 0.01, n1=400
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)), n_img)
# input_x = x_train[x_idx].reshape(len(x_idx), np.product(x_train.shape[1:])) / 255.0 * max_fr
# plt_title = 'mnist digit'

# # fashion mnist: lr = 0.01, n1=400
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# x_idx = np.random.choice(np.arange(len(x_train)), n_img)
# # input_x = x_train[x_idx].flatten() / 255.0 * max_fr
# input_x = x_train[x_idx].reshape(len(x_idx), np.product(x_train.shape[1:])) / 255.0 * max_fr
# plt_title = 'fashion mnist sample'

# grayscale cifar-10: lr = 0.01, n1=784
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_idx = np.random.choice(np.arange(len(x_train)), n_img)
input_x = tf.image.rgb_to_grayscale(x_train[x_idx]).numpy().reshape(n_img, np.product(x_train.shape[1:-1])) / 255.0 * max_fr
plt_title = 'cifar-10 sample'

# input_x = (n_img, n_pixel)
img_xy = int(np.sqrt(input_x.shape[1]))

# plot
input_fig, input_axs = plt.subplots(len(input_x), 1)
if n_img == 1:
    input_axs = [input_axs]
for img_i, input_ax in enumerate(input_axs):
    img = input_ax.imshow(input_x[img_i].reshape(img_xy, img_xy), cmap='gray')
    input_ax.axis("off")
    input_fig.colorbar(img, ax=input_ax, shrink=0.6)
input_fig.suptitle(plt_title)
input_fig.show()

# network size
n0 = input_x.shape[1]
n0i = int(n0 / 4)
n1 = 784
n1i = n1
n_inh = n0

# initialize weights
w_r0_ei = np.random.random((n0i, n0))
w_r0_ie = np.random.random((n0, n0i)) * 1 / (n0 * n0i)

w_01_p = np.random.random((n1, n0)) / (n1 * n0)
w_01_n = w_01_p
# w_01_n = np.random.random((n1, n0)) / (n1 * n0)

w_ppe_e = np.random.random((n0, n0)) #* 1 / (n0 * n0)
w_npe_e = w_ppe_e # np.random.random((n0, n0)) #* 1 / (n0 * n0)

lr = 0.01
n_epoch = 10
psse = []
nsse = []
reconstruction = []
reconstruction_npe = []

plt.close('all')

for epoch_i in trange(n_epoch):

    for img_i in range(n_img):

        # R0 circuit: n0 = nPixel, n0i = 3
        r0_e = np.zeros((n0, T_steps))
        r0_i = np.zeros((n0i, T_steps))

        # R1 circuit
        r1_e = np.zeros((n1, T_steps))
        r1_i = np.zeros((n1i, T_steps))
        r1_r = np.ones((n1, T_steps)) #* 0.1

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

        curr_input = input_x[img_i]

        for i in range(T_steps - 1):

            # r0_exc = X - r0_inh
            input_r0_e = curr_input

            # r0_pPe_exc = dampened input + self loop - PV - SST
            input_ppe_e = r0_e[:, i] + r_ppe_e[:, i] - r_ppe_pv[:, i] - r_ppe_sst[:, i]
            # r0_pPE_PV = pred + r0_pPE_exc
            input_ppe_pv = w_01_p.T @ r1_r[:, i] + r_ppe_e[:, i]
            # r0_pPE_SST = pred + r0_pPE_exc - VIP
            input_ppe_sst = w_01_p.T @ r1_r[:, i] + r_ppe_e[:, i] - r_ppe_vip[:, i]
            # r0_pPE_VIP = pred + r0_pPE_exc
            input_ppe_vip = w_01_p.T @ r1_r[:, i] + r_ppe_e[:, i]

            # r0_nPe_exc = dampened input + pred + self loop - PV - SST
            input_npe_e = (
                    r0_e[:, i] + w_01_n.T @ r1_r[:, i] + r_npe_e[:, i]
                    - r_npe_pv[:, i] - r_npe_sst[:, i]
                           )
            # r0_nPe_PV = dampened input + r_nPE_exc
            input_npe_pv = r0_e[:, i] + r_npe_e[:, i]
            # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
            input_npe_sst = r0_e[:, i] + w_01_n.T @ r1_r[:, i] + r_npe_e[:, i] - r_npe_vip[:, i]
            # r0_nPe_VIP = pred + r0_nPE_exc
            input_npe_vip = w_01_n.T @ r1_r[:, i] + r_npe_e[:, i]

            # r1_exc = pPE - r1_inh (nPE)
            # r1 rep = r1_exc
            input_r1_r = r1_e[:, i] - r1_i[:, i] + r1_r[:, i]
            # input_r1_e = w_01_p @ r_ppe_e[:, i]
            input_r1_e = w_01_p @ r_ppe_e[:, i]
            # r1_inh = nPE
            # input_r1_i = w_01_n @ r_npe_e[:, i] #+ r1_e[:, i]
            input_r1_i = w_01_n @ r_npe_e[:, i]


            r0_e[:, i + 1] = dr(r=r0_e[:, i], inputs=input_r0_e, ei_type='exc')
            # r0_i[:, i + 1] = dr(r=r0_i[:, i], inputs=input_r0_i)

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

            # r0_e[:, i + 1] = r0_e[:, i] + (-r0_e[:, i] / 0.02 + input_r0_e / 0.01) * dt
            #
            # r1_e[:, i + 1] = r1_e[:, i] + (-r1_e[:, i] / 0.02 + input_r1_e) * dt
            # r1_i[:, i + 1] = r1_i[:, i] + (-r1_i[:, i] / 0.02 + input_r1_i / 0.01) * dt
            # r1_r[:, i + 1] = r1_r[:, i] + (-r1_r[:, i] / 0.02 + input_r1_r / 0.01) * dt
            #
            # r_ppe_e[:, i + 1] = r_ppe_e[:, i] + (-r_ppe_e[:, i] / 0.02 + input_ppe_e / 0.01) * dt
            # r_ppe_pv[:, i + 1] = r_ppe_pv[:, i] + (-r_ppe_pv[:, i] / 0.02 + input_ppe_pv / 0.01) * dt
            # r_ppe_sst[:, i + 1] = r_ppe_sst[:, i] + (-r_ppe_sst[:, i] / 0.02 + input_ppe_sst / 0.01) * dt
            # r_ppe_vip[:, i + 1] = r_ppe_vip[:, i] + (-r_ppe_vip[:, i] / 0.02 + input_ppe_vip / 0.01) * dt
            #
            # r_npe_e[:, i + 1] = r_npe_e[:, i] + (-r_npe_e[:, i] / 0.02 + input_npe_e / 0.01) * dt
            # r_npe_pv[:, i + 1] = r_npe_pv[:, i] + (-r_npe_pv[:, i] / 0.02 + input_npe_pv / 0.01) * dt
            # r_npe_sst[:, i + 1] = r_npe_sst[:, i] + (-r_npe_sst[:, i] / 0.02 + input_npe_sst / 0.01) * dt
            # r_npe_vip[:, i + 1] = r_npe_vip[:, i] + (-r_npe_vip[:, i] / 0.02 + input_npe_vip / 0.01) * dt

            # decay = -r / params_dict[ei_type]['tau_d']
            # rise = (fx + bg_r) / params_dict[ei_type]['tau_r']
            # return r + dt * (decay + rise)

        rep1 = r1_r[:, -hebb_window:].mean(axis=1)
        ppe0 = r_ppe_e[:, -hebb_window:].mean(axis=1).T
        npe0 = r_npe_e[:, -hebb_window:].mean(axis=1).T

        # rep1 = input_r1_r
        # ppe0 = input_ppe_e
        # npe0 = input_npe_e

        dw_01_p = lr * (np.einsum('i,j->ij', rep1, ppe0)) / 2
        dw_01_n = lr * (np.einsum('i,j->ij', rep1, npe0)) / 2
        # dw_01_p = lr * (np.einsum('i,j->ij', ReLu(rep1), ReLu(ppe0) * drelu(ppe0))) / 4
        # dw_01_n = lr * (np.einsum('i,j->ij', ReLu(rep1), ReLu(npe0) * drelu(npe0))) / 2

        # w_01_p += dw_01_p
        # w_01_n -= dw_01_n
        # w_01_p = np.maximum(w_01_p + dw_01_p, 0.0)
        # w_01_n = np.maximum(w_01_n - dw_01_n, 0.0)
        w_01_p = np.maximum(w_01_p + dw_01_p - dw_01_n, 0.0)
        w_01_n = w_01_p

        # ppe
        psse.append(np.sum(r_ppe_e[:, -hebb_window:]))
        # npe
        nsse.append(np.sum(r_npe_e[:, -hebb_window:]))
        # pred to ppe
        # reconstruction.append((r_ppe_pv + r_ppe_sst)[:, -hebb_window:].mean(axis=1))
        reconstruction.append((w_01_p.T @ ReLu(r1_r))[:, -hebb_window:].mean(axis=1))
        # pred to npeprogress_by = int(n_epoch / 10)
        reconstruction_npe.append((w_01_n.T @ ReLu(r1_r))[:, -hebb_window:].mean(axis=1))

    if (epoch_i + 1) % 2 == 0:

        vmax = r0_e.mean(axis=1).max()
        plt.figure(figsize=(4, 15))
        #
        plt.subplot(511)
        plt.imshow(r0_e[:, -hebb_window:].mean(axis=1).reshape(img_xy, img_xy), cmap='gray')#, vmin=0, vmax=vmax)
        plt.colorbar(shrink=0.6)
        plt.axis('off')
        plt.title('R0')
        plt.subplot(512)
        plt.imshow(r_ppe_e[:, -hebb_window:].mean(axis=1).reshape(img_xy, img_xy), cmap='Reds')#, vmin=0, vmax=vmax)
        plt.colorbar(shrink=0.6)
        plt.axis('off')
        plt.title('pPE')
        plt.subplot(513)
        plt.imshow(r_npe_e[:, -hebb_window:].mean(axis=1).reshape(img_xy, img_xy), cmap='Blues')#, vmin=0, vmax=vmax)
        plt.colorbar(shrink=0.6)
        plt.axis('off')
        plt.title('nPE')
        plt.subplot(514)
        plt.imshow(reconstruction[-1].reshape(img_xy, img_xy), cmap='gray')#, vmin=0, vmax=vmax)
        plt.colorbar(shrink=0.6)
        plt.axis('off')
        plt.title('Prediction to pPE')
        plt.subplot(515)
        plt.imshow(reconstruction_npe[-1].reshape(img_xy, img_xy), cmap='gray')#, vmin=0, vmax=vmax)
        plt.colorbar(shrink=0.6)
        plt.axis('off')
        plt.title('Prediction to nPE')
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.subplot(211)
        plt.plot(psse, c='r', label='pPE')
        plt.title('PE+')
        plt.subplot(212)
        plt.plot(nsse, c='b', label='nPE')
        plt.title('PE-')
        plt.show()

progress_by = int(n_epoch / 10)
progress_fig, progress_axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 5))
for i, ax in enumerate(progress_axs.flatten()):
    img = ax.imshow(reconstruction[::progress_by][i].reshape(img_xy, img_xy), cmap='gray')#, vmin=0, vmax=vmax)
    progress_fig.colorbar(img, ax=ax, shrink=0.3)
    ax.set_title(f'iter #{(i+1) * progress_by}')
    ax.set_axis_off()
progress_fig.tight_layout()
progress_fig.show()