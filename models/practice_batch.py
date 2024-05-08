import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from parameters import dt
from practice_scalar import dr
import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


T = 0.5
T_steps = int(T / dt)
hebb_window = int(T_steps * 0.1)


def pick_samples(label, nclass, nsample):
    all_class = np.unique(label)
    if nclass == len(all_class):
        class_list = all_class
    else:
        class_list = np.random.choice(all_class, nclass)

    sample_idcs = []
    for i in class_list:
        idcs = np.where(label == i)[0]
        sample_idcs.append(np.random.choice(idcs, nsample))

    return np.ravel(sample_idcs).astype(int)


def create_input_data(max_fr, img_set, nclass, nsample):
    if img_set == 'mnist':
        # mnist: lr = 5.0
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        plt_title = 'mnist digit'

    elif img_set == 'fmnist':
        # fashion mnist: lr = 5.0, nepoch=30
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        plt_title = 'fashion mnist sample'

    elif img_set == 'cifar':
        # grayscale cifar-10
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = tf.squeeze(tf.image.rgb_to_grayscale(x_train), -1).numpy()
        x_test = tf.squeeze(tf.image.rgb_to_grayscale(x_test), -1).numpy()
        plt_title = 'cifar-10 sample'

    # pick samples
    x_idx = pick_samples(y_train, nclass, nsample)
    test_idx = pick_samples(y_test, nclass, int(nsample / 4))
    # preprocessing
    input_x = x_train[x_idx].reshape(len(x_idx), np.product(x_train.shape[1:])) / np.max(x_train) * max_fr
    test_x = x_test[test_idx].reshape(len(test_idx), np.product(x_test.shape[1:])) / np.max(x_test) * max_fr

    # compute img dimensions
    img_xy = int(np.sqrt(input_x.shape[-1]))

    #
    plot_img_idcs = pick_samples(y_train[x_idx], nclass, 1)
    # plot
    input_fig, input_axs = plt.subplots(nrows=1, ncols=nclass, figsize=(3*nclass, 3))
    for img_input, ax_input in enumerate(input_axs):
        img = ax_input.imshow(input_x[plot_img_idcs[img_input]].reshape(img_xy, img_xy), cmap='Reds')
        ax_input.axis("off")
        ax_input.set_title(y_train[x_idx][plot_img_idcs[img_input]][0])
        plt.colorbar(img, ax=ax_input, shrink=0.6)

    input_fig.suptitle(plt_title)
    input_fig.tight_layout()

    return input_fig, input_x, y_train[x_idx], test_x, y_test[test_idx]


input_fig, x_train, y_train, x_test, y_test = create_input_data(
    max_fr=30, img_set='cifar', nclass=10, nsample=128)
input_fig.show()

def create_neuron_group(n_neuron, size_batch):
    return tf.zeros(shape=(n_neuron, size_batch), dtype=tf.float32)

def create_layer(net_size, size_batch, top=False, bot=False):
    '''
    :param net_size: [n0_rep_exc, n0_rep_inh, n0_rep_fb, n0_err_pyr, n0_err_inh]
    :return:
    '''

    layer_labels = [
        'rep_exc', 'rep_inh', 'rep_fb',
        'pe_pyr', 'pe_vp', 'pe_sst', 'pe_vip',
        'ne_pyr', 'ne_vp', 'ne_sst', 'ne_vip'
    ]

    # top layer
    if top:
        layer_labels = layer_labels[:3]
        layer_size = net_size[:3]
    # bottom layer
    elif bot:
        layer_labels.remove('rep_fb')
        layer_size = net_size[:2] + [net_size[3]] + [net_size[4]] * 3 + [net_size[3]] + [net_size[4]] * 3
    else:
        pass

    return {label_i: create_neuron_group(layer_size[i], size_batch) for i, label_i in enumerate(layer_labels)}

def initialize_network(n_layer, net_size, size_batch):

    # create layer
    network = {}
    for layer_i in range(n_layer):
        if layer_i == 0:
            network[f'area_{i}'] = create_layer(net_size, size_batch, top=False, bot=True)
        elif layer_i == n_layer - 1:
            network[f'area_{i}'] = create_layer(net_size, size_batch, top=True, bot=False)
        else:
            network[f'area_{i}'] = create_layer(net_size, size_batch, top=False, bot=False)

    return network

def initialzie_weights(network):

    # bot layer: rep E-I
    w_r0_ei = np.random.random((net_size[1], net_size[0]))
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

batch_size = 1

# network size
n0 = len(input_x)
n0i = int(n0 / 4)

n1 = (int(np.sqrt(n0)) + 4) ** 2
n1i = n1

n_inh = n0

# # R0 circuit: n0 = nPixel, n0i = 3
# r0_e = np.zeros((n0, batch_size))
# r0_i = np.zeros((n0i, batch_size))
#
# # R1 circuit
# r1_e = np.zeros((n1, batch_size))
# r1_i = np.zeros((n1i, batch_size))
# r1_r = np.zeros((n1, batch_size))
#
# # error circuits
# r_pe_r = np.zeros((n0, batch_size))
#
# # initialize pPE circuit
# r_ppe_e = np.zeros((n0, batch_size))
# r_ppe_pv = np.zeros((n_inh, batch_size))
# r_ppe_sst = np.zeros((n_inh, batch_size))
# r_ppe_vip = np.zeros((n_inh, batch_size))
#
# # initialize nPE circuit
# r_npe_e = np.zeros((n0, batch_size))
# r_npe_pv = np.zeros((n_inh, batch_size))
# r_npe_sst = np.zeros((n_inh, batch_size))
# r_npe_vip = np.zeros((n_inh, batch_size))

# initialize weights
w_r0_ei = np.random.random((n0i, n0))
w_r0_ie = np.random.random((n0, n0i)) * 1 / (n0 * n0i)

w_0pe_1e = np.random.random((n1, n0))
w_0ne_1i = np.random.random((n1, n0))

w_01p_pv = np.random.random((n_inh, n1))
w_01p_sst = np.random.random((n_inh, n1))
w_01p_vip = np.random.random((n_inh, n1))

w_01n_e = np.random.random((n0, n1))
w_01n_sst = np.random.random((n_inh, n1))
w_01n_vip = np.random.random((n_inh, n1))

w_e_self = 1.0
w_pv_self = 1.0

lr = 0.1
n_epoch = 10
psse = []
nsse = []
reconstruction = []
reconstruction_npe = []
#
# for epoch_i in trange(n_epoch):
#
#     exc1 = np.zeros((n1, batch_size))
#     inh1 = np.zeros((n1i, batch_size))
#     rep1 = np.zeros((n1, batch_size))
#
#     p_e = np.zeros((n0, batch_size))
#     p_pv = np.zeros((n_inh, batch_size))
#     p_sst = np.zeros((n_inh, batch_size))
#     p_vip = np.zeros((n_inh, batch_size))
#
#     n_e = np.zeros((n0, batch_size))
#     n_pv = np.zeros((n_inh, batch_size))
#     n_sst = np.zeros((n_inh, batch_size))
#     n_vip = np.zeros((n_inh, batch_size))
#
#     for i in range(T_steps):
#
#         # r0_exc = X - r0_inh
#         input_r0_e = input_x.reshape(len(input_x), batch_size) - w_r0_ie @ r0_i
#         # r0_inh = w_EI @ r0_exc
#         input_r0_i = w_r0_ei @ r0_e
#
#         # r1_exc = pPE - r1_inh (nPE)
#         input_r1_e = w_0pe_1e @ r_ppe_e + r1_e - r1_i
#         # r1_inh = nPE
#         input_r1_i = w_0ne_1i @ r_npe_e - r1_i
#         # r1 rep = r1_exc
#         input_r1_r = r1_e + r1_r
#
#         # r0_pPe_exc = dampened input + self loop + - PV - SST
#         input_ppe_e = r0_e + w_e_self * r_ppe_e - r_ppe_pv - r_ppe_sst
#         # r0_pPE_PV = pred - self loop + r0_pPE_exc
#         input_ppe_pv = w_01p_pv @ r1_r - w_pv_self * r_ppe_pv + w_e_self * r_ppe_e
#         # r0_pPE_SST = pred + r0_pPE_exc - VIP
#         input_ppe_sst = w_01p_sst @ r1_r + r_ppe_e - r_ppe_vip
#         # r0_pPE_VIP = pred + r0_pPE_exc
#         input_ppe_vip = w_01p_vip @ r1_r + r_ppe_e
#
#         # r0_nPe_exc = dampened input + pred + self loop - PV - SST
#         input_npe_e = r0_e + w_01n_e @ r1_r + w_e_self * r_npe_e - r_npe_pv - r_npe_sst
#         # r0_nPe_PV = dampened input - self loop + r_nPE_exc
#         input_npe_pv = r0_e - w_pv_self * r_npe_pv + r_npe_e
#         # r0_nPe_SST = dampened input + pred + r_nPE_exc - VIP
#         input_npe_sst = r0_e + w_01n_sst @ r1_r + r_npe_e - r_npe_vip
#         # r0_nPe_VIP = pred + r0_nPE_exc
#         input_npe_vip = w_01n_vip @ r1_r + r_npe_e
#
#         # if i == T_steps - 1:
#         #     print(input_r0_e.shape, input_r0_i.shape, input_r1_e.shape, input_r1_i.shape, input_r1_r.shape)
#         #     print(input_ppe_e.shape, input_ppe_pv.shape, input_ppe_sst.shape, input_ppe_vip.shape)
#         #     print(input_ppe_e.shape, input_ppe_pv.shape, input_ppe_sst.shape, input_ppe_vip.shape)
#
#         r0_e = dr(r=r0_e, inputs=input_r0_e, ei_type='exc')
#         r0_i = dr(r=r0_i, inputs=input_r0_i)
#
#         r1_e = dr(r=r1_e, inputs=input_r1_e, ei_type='exc')
#         r1_i = dr(r=r1_i, inputs=input_r1_i)
#         r1_r = dr(r=r1_r, inputs=input_r1_r, ei_type='exc')
#
#         r_ppe_e = dr(r=r_ppe_e, inputs=input_ppe_e, ei_type='exc')
#         r_ppe_pv = dr(r=r_ppe_pv, inputs=input_ppe_pv)
#         r_ppe_sst = dr(r=r_ppe_sst, inputs=input_ppe_sst)
#         r_ppe_vip = dr(r=r_ppe_vip, inputs=input_ppe_vip)
#
#         r_npe_e = dr(r=r_npe_e, inputs=input_npe_e, ei_type='exc')
#         r_npe_pv = dr(r=r_npe_pv, inputs=input_npe_pv)
#         r_npe_sst = dr(r=r_npe_sst, inputs=input_npe_sst)
#         r_npe_vip = dr(r=r_npe_vip, inputs=input_npe_vip)
#
#         if i > hebb_window:
#             exc1 += r1_e
#             inh1 += r1_i
#             rep1 += r1_r
#
#             p_e += r_ppe_e
#             p_pv += r_ppe_pv
#             p_sst += r_ppe_sst
#             p_vip += r_ppe_vip
#
#             n_e += r_npe_e
#             n_pv += r_npe_pv
#             n_sst += r_npe_sst
#             n_vip += r_npe_vip
#
#     # print(r0_e.shape, r0_i.shape, r1_e.shape, r1_i.shape, r1_r.shape)
#     # print(r_ppe_e.shape, r_ppe_pv.shape, r_ppe_sst.shape, r_ppe_vip.shape)
#     # print(r_npe_e.shape, r_npe_pv.shape, r_npe_sst.shape, r_npe_vip.shape)
#
#     # compute gradient
#     dw_0pe_1e = -lr * (np.einsum('ij,kj->ik', exc1 / hebb_window, p_e / hebb_window))
#     dw_0ne_1i = -lr * (np.einsum('ij,kj->ik', inh1 / hebb_window, n_e / hebb_window))
#
#     dw_01p_pv = -lr * (np.einsum('ij,kj->ik', p_pv / hebb_window, rep1 / hebb_window))
#     dw_01p_sst = -lr * (np.einsum('ij,kj->ik', p_sst / hebb_window, rep1 / hebb_window))
#     dw_01p_vip = -lr * (np.einsum('ij,kj->ik', p_vip / hebb_window, rep1 / hebb_window))
#
#     dw_01n_e = -lr * (np.einsum('ij,kj->ik', n_e / hebb_window, rep1 / hebb_window))
#     dw_01n_sst = -lr * (np.einsum('ij,kj->ik', n_sst / hebb_window, rep1 / hebb_window))
#     dw_01n_vip = -lr * (np.einsum('ij,kj->ik', n_vip / hebb_window, rep1 / hebb_window))
#
#     # weight update
#     w_0pe_1e = np.maximum(w_0pe_1e - dw_0pe_1e, 0.0)
#     w_0ne_1i = np.maximum(w_0ne_1i - dw_0ne_1i, 0.0)
#
#     w_01p_pv = np.maximum(w_01p_pv - dw_01p_pv, 0.0)
#     w_01p_sst = np.maximum(w_01p_sst - dw_01p_sst, 0.0)
#     w_01p_vip = np.maximum(w_01p_vip - dw_01p_vip, 0.0)
#
#     w_01n_e = np.maximum(w_01n_e - dw_01n_e, 0.0)
#     w_01n_sst = np.maximum(w_01n_sst - dw_01n_sst, 0.0)
#     w_01n_vip = np.maximum(w_01n_vip - dw_01n_vip, 0.0)
#
#     # append error
#     psse.append(np.sum((p_e / hebb_window) ** 2))
#     nsse.append(np.sum((n_e / hebb_window) ** 2))
#
#     # append reconstruction
#     reconstruction.append(p_pv / hebb_window + p_sst / hebb_window)
#     reconstruction_npe.append(-w_01n_e @ (rep1 / hebb_window) + n_pv / hebb_window + n_sst / hebb_window)
#
# vmax = r0_e.max()
# plt.figure(figsize=(4, 15))
# plt.subplot(511)
# plt.imshow(r0_e.reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
# plt.colorbar(shrink=0.6)
# plt.axis('off')
# plt.title('R0')
# plt.subplot(512)
# plt.imshow(r_ppe_e.reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
# plt.colorbar(shrink=0.6)
# plt.axis('off')
# plt.title('pPE')
# plt.subplot(513)
# plt.imshow(r_npe_e.reshape(img_xy, img_xy), cmap='Blues', vmin=0, vmax=vmax)
# plt.colorbar(shrink=0.6)
# plt.axis('off')
# plt.title('nPE')
# plt.subplot(514)
# plt.imshow(reconstruction[-1].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
# plt.colorbar(shrink=0.6)
# plt.axis('off')
# plt.title('Prediction to pPE')
# plt.subplot(515)
# plt.imshow(reconstruction_npe[-1].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
# plt.colorbar(shrink=0.6)
# plt.axis('off')
# plt.title('Prediction to nPE')
# plt.tight_layout()
# plt.show()
#
# plt.figure()
# plt.plot(psse, c='r', label='pPE')
# plt.plot(nsse, c='b', label='nPE')
# plt.title('error')
# plt.legend()
# plt.show()
#
# progress_by = int(n_epoch / 10)
# progress_fig, progress_axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 5))
# for i, ax in enumerate(progress_axs.flatten()):
#     img = ax.imshow(reconstruction[::progress_by][i].reshape(img_xy, img_xy), cmap='Reds', vmin=0, vmax=vmax)
#     progress_fig.colorbar(img, ax=ax, shrink=0.3)
#     ax.set_title(f'iter #{(i + 1) * progress_by}')
#     ax.set_axis_off()
# progress_fig.tight_layout()
# progress_fig.show()
