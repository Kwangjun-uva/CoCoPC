import pickle5 as pickle
import numpy as np
from script.pc_network import network
import matplotlib.pyplot as plt
from tqdm import trange

# specify the model dir
model_dir = '/home/kwangjun/PycharmProjects/si_pc/cifar10/trial03/'

# load simulation parameters, weights, and dataset
with open(model_dir + 'sim_params.pkl', 'rb') as f:
    sim_params = pickle.load(f)
with open(model_dir + 'weights.pkl', 'rb') as f:
    pretrained_weights = pickle.load(f)
with open(model_dir + 'dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# initialize network
test_net = network(
        neurons_per_layer=sim_params['net_size'],
        bu_rate=sim_params['bu_rate'], td_rate=sim_params['td_rate'],
        tau_exc=sim_params['tau_exc'], tau_inh=sim_params['tau_inh'],
        symm_w=sim_params['symmetric_weight'], pretrained_weights=pretrained_weights,
        bg_exc=0.0, bg_inh=0.0, jitter_lvl=0.0, jitter_type='constant'
    )

# test_net.add_noise()
test_net.initialize_network(batch_size=dataset['train_x'][:16].shape[0])
test_net.initialize_error()

steps_isi = int(sim_params['isi_time'] / sim_params['dt'])
steps_sim = int(sim_params['sim_time'] / sim_params['dt'] * 5)

print(sim_params['tau_exc'], sim_params['tau_inh'], steps_sim)

for _ in trange(steps_isi):
    test_net.compute(inputs=np.zeros(dataset['train_x'][:16].T.shape), record=None)

# # rep saver
# rep_save = np.zeros((*test_net.network['layer_1']['rep_r'].shape, steps_sim))
# # err saver
# ppe_save = np.zeros((*test_net.network['layer_0']['ppe_pyr'].shape, steps_sim))
# npe_save = np.zeros((*test_net.network['layer_0']['npe_pyr'].shape, steps_sim))

# img_idx = 0
# plt.subplot(311)
# plt.plot(rep_save[:, img_idx, :].mean(axis=0), c='purple')
# plt.subplot(312)
# plt.plot(ppe_save[:, img_idx, :].mean(axis=0), c='r')
# plt.subplot(313)
# plt.plot(npe_save[:, img_idx, :].mean(axis=0), c='b')
# plt.show()


pe_circuit_neuron_groups = list(test_net.network['layer_0'].keys())[3:]
interneuron_dict = {
    key: np.zeros((*grp.shape, steps_sim))
    for key, grp in test_net.network['layer_0'].items()
    if ('pe' in key) and ('pyr') not in key
}

# # sim
# for t_step in trange(steps_sim):
#     test_net.compute(inputs=dataset['train_x'][:16].T, record=None)
#     # rep_save += model.network['layer_1']['rep_r']
#     rep_save[:, :, t_step] = test_net.network['layer_1']['rep_r']
#     ppe_save[:, :, t_step] = test_net.network['layer_0']['ppe_pyr']
#     npe_save[:, :, t_step] = test_net.network['layer_0']['npe_pyr']
#
# test_net.initialize_network(batch_size=dataset['train_x'][:16].shape[0])
# test_net.initialize_error()
# for _ in trange(steps_isi):
#     test_net.compute(inputs=np.zeros(dataset['train_x'][:16].T.shape), record=None)
#
for t_step in trange(steps_sim):
    test_net.compute(inputs=dataset['train_x'][:16].T, record=None)
    for key, grp in interneuron_dict.items():
        grp[:, :, t_step] += test_net.network['layer_0'][key]
#
# n_colors = rep_save.shape[1]
# cmap_variables = ['Blues', 'Greens', 'Wistia']
#
# fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
# for i, (key, grp) in enumerate(interneuron_dict.items()):
#     line_colors = plt.get_cmap(cmap_variables[i % 3])(np.linspace(0, 1, n_colors))
#     for img_i in range(n_colors):
#         axs.flat[i].plot(grp[:, img_i, :].mean(axis=0), c=line_colors[img_i])
#         axs.flat[i].set_title(" ".join(key.split('_')))
#         axs.flat[i].spines['right'].set_visible(False)
#         axs.flat[i].spines['top'].set_visible(False)
# fig.tight_layout()
# fig.show()
# # fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/figures/fig4b_variance.png', dpi=300, bbox_inches='tight')
#
# fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
# for i, (key, grp) in enumerate(interneuron_dict.items()):
#     line_colors = plt.get_cmap(cmap_variables[i % 3])(np.linspace(0, 1, n_colors))
#
#     axs.flat[i].plot(grp.mean(axis=1).mean(axis=0), c=line_colors[img_i])
#     axs.flat[i].set_title(" ".join(key.split('_')))
    #     axs.flat[i].spines['right'].set_visible(False)
    #     axs.flat[i].spines['top'].set_visible(False)
    # fig.tight_layout()
    # fig.show()
    # # fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/figures/fig4b.png', dpi=300, bbox_inches='tight')
    #
# rep_colors = plt.cm.Purples(np.linspace(0, 1, n_colors))
# ppe_colors = plt.cm.Reds(np.linspace(0, 1, n_colors))
# npe_colors = plt.cm.Blues(np.linspace(0, 1, n_colors))
#
# titles = ['Representation L5 pyramidal activity', 'pPE pyramidal activity', 'nPE pyramidal activity']
#
# plt.close('all')
# fig, axs = plt.subplots(nrows=3, sharex=True)
#
# for i in range(n_colors):
#     axs[0].plot(rep_save[:, i, :].mean(axis=0), linewidth=2, c=rep_colors[i])
#     axs[1].plot(ppe_save[:, i, :].mean(axis=0), linewidth=2, c=ppe_colors[i])
#     axs[2].plot(npe_save[:, i, :].mean(axis=0), linewidth=2, c=npe_colors[i])
# for i in range(n_colors):
#     axs[0].plot(rep_save.mean(axis=0).mean(axis=0), linewidth=2, c=rep_colors[i])
#     axs[1].plot(ppe_save.mean(axis=0).mean(axis=0), linewidth=2, c=ppe_colors[i])
#     axs[2].plot(npe_save.mean(axis=0).mean(axis=0), linewidth=2, c=npe_colors[i])
#
# for ax_i, ax in enumerate(axs.flat):
#     ax.set_title(titles[ax_i])
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#
# axs[1].set_ylabel('firing rates (a.u.)')
# axs[2].set_xlabel('time (ms)')
#
# fig.tight_layout()
# plt.show()
# # fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/figures/fig4a.png', dpi=300, bbox_inches='tight')

# # frequency
# from scipy import signal
#
# r_data = [rep_save.mean(axis=0).mean(axis=0), ppe_save.mean(axis=0).mean(axis=0), npe_save.mean(axis=0).mean(axis=0)]
# periodogram_titles = ['Rep', 'pPE', 'nPE']
# periodogram_colors = [rep_colors[-1], ppe_colors[-1], npe_colors[-1]]
# fig_pdg, axs_pdg = plt.subplots(3, 1, sharex=True, sharey=True)
# for ax_i, ax_pdg in enumerate(axs_pdg.flat):
#     f, Sk = signal.periodogram(r_data[ax_i], fs = 1/0.001, return_onesided=True, scaling = "spectrum")
#     ax_pdg.semilogy(f, Sk, c=periodogram_colors[ax_i], lw=2)
#     ax_pdg.axvline(f[np.argmax(Sk)], c='k', ls='--')
#     if ax_i == 2:
#         ax_pdg.set_xlabel('frequency [Hz]')
#     if ax_i == 1:
#         ax_pdg.set_ylabel('PSD [V**2/Hz]')
#     ax_pdg.set_title(f'{periodogram_titles[ax_i]}: max f = {f[np.argmax(Sk)]}')
#     ax_pdg.spines['top'].set_visible(False)
#     ax_pdg.spines['right'].set_visible(False)
#     ax_pdg.set_ylim([10e-12, 10e-3])
# fig_pdg.tight_layout()
# fig_pdg.show()
#
# tau = [2, 4, 20, 100, 200]
# osc_freq = [0.6]
# pick the first two crests and troughs to show waxing and waning of reconstruction

# plt.close('all')
# fig, axs = plt.subplots(nrows=3, sharex=True)
#
#
# axs[0].plot(rep_save.mean(axis=1).mean(axis=0)[:3000], linewidth=2, c=rep_colors[-1])
# axs[1].plot(ppe_save.mean(axis=1).mean(axis=0)[:3000], linewidth=2, c=ppe_colors[-1])
# axs[2].plot(npe_save.mean(axis=1).mean(axis=0)[:3000], linewidth=2, c=npe_colors[-1])
#
# for ax_i, ax in enumerate(axs.flat):
#     ax.set_title(titles[ax_i])
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#
# for v_val in [44, 99+44, 100+44+68, 100+44+161]:
#     for ax_i, ax in enumerate(axs.flat):
#         ax.axvline(x=v_val, ls='--', c='k')
#
# axs[1].set_ylabel('firing rates (a.u.)')
# axs[2].set_xlabel('time (ms)')
#
# fig.tight_layout()
# plt.show()
#
# plt.close('all')
# idcs = [44, 99 + 44, 100 + 44 + 68, 100 + 44 + 161, -5000]
# fig, axs = plt.subplots(3, len(idcs), sharex=True, sharey=True)
# for idx_i, idx in enumerate(idcs):
#     for img_i in range(3):
#         imgs = test_net.weights['01'].T @ rep_save[:, :, idx]
#         axs[img_i, idx_i].imshow(imgs[:, img_i].reshape(32, 32), cmap='gray', vmin=0, vmax=1)
#         axs[img_i, idx_i].axis('off')
#         axs[img_i, idx_i].set_title(f'img#{img_i + 1}, time #{idx_i + 1}')
# fig.tight_layout()
# fig.show()

# rep, ppe, and npe
# plt.close('all')
# idcs = [44, 99 + 44, 100 + 44 + 68, 100 + 44 + 161, -5000]
# fig, axs = plt.subplots(3, len(idcs), sharex=True, sharey=True)
# fig_ppe, axs_ppe = plt.subplots(3, len(idcs), sharex=True, sharey=True)
# fig_npe, axs_npe = plt.subplots(3, len(idcs), sharex=True, sharey=True)
# for idx_i, idx in enumerate(idcs):
#     for img_i in range(3):
#         imgs = test_net.weights['01'].T @ rep_save[:, :, idx]
#         axs[img_i, idx_i].imshow(imgs[:, img_i].reshape(32, 32), cmap='gray', vmin=0, vmax=1)
#         axs[img_i, idx_i].axis('off')
#         axs[img_i, idx_i].set_title(f'img#{img_i + 1}, time #{idx_i + 1}')
#
#         ppes = ppe_save[:, :, idx]
#         axs_ppe[img_i, idx_i].imshow(ppes[:, img_i].reshape(32, 32), cmap='Reds', vmin=0, vmax=1)
#         axs_ppe[img_i, idx_i].axis('off')
#         axs_ppe[img_i, idx_i].set_title(f'img#{img_i + 1}, time #{idx_i + 1}')
#         npes = npe_save[:, :, idx]
#         axs_npe[img_i, idx_i].imshow(npes[:, img_i].reshape(32, 32), cmap='Blues', vmin=0, vmax=1)
#         axs_npe[img_i, idx_i].axis('off')
#         axs_npe[img_i, idx_i].set_title(f'img#{img_i + 1}, time #{idx_i + 1}')
#
# fig.tight_layout()
# fig.show()
# fig_ppe.tight_layout()
# fig_ppe.show()
# fig_npe.tight_layout()
# fig_npe.show()

# get an example img oscillations
# plt.close('all')
# fig, axs = plt.subplots(nrows=3, sharex=True)
#
# axs[0].plot(rep_save[:, 10, :].mean(axis=0)[:3000], linewidth=2, c=rep_colors[-1])
# axs[1].plot(ppe_save[:, 10, :].mean(axis=0)[:3000], linewidth=2, c=ppe_colors[-1])
# axs[2].plot(npe_save[:, 10, :].mean(axis=0)[:3000], linewidth=2, c=npe_colors[-1])
#
# for v_val in [44, 99+44, 100+44+68, 100+44+161]:
#     for ax_i, ax in enumerate(axs.flat):
#         if ax_i == 0:
#             cc = rep_colors[-1]
#         elif ax_i == 1:
#             cc = ppe_colors[-1]
#         elif ax_i == 2:
#             cc = npe_colors[-1]
#         ax.axvline(x=v_val, ls='--', c=cc, alpha=0.5)
#         ax.set_title(titles[ax_i], c=cc)
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#
# axs[1].set_ylabel('firing rates (a.u.)')
# axs[2].set_xlabel('time (ms)')
#
# fig.tight_layout()
# plt.show()
# fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/figures/fig4/fig4a_exampleImg_oscillations.png', dpi=300, bbox_inches='tight')


# # get recon, ppe, and npe on example img
#
# idcs = [44, 99 + 44, 100 + 44 + 68, 100 + 44 + 161, -5000]
# titles = ['c1', 't1', 'c2', 't2', 'final']
# for idx_i, idx in enumerate(idcs):
#     fig, axs = plt.subplots(1, 1)
#     imgs = npe_save[:, 10, idx]
#     # imgs = ppe_save[:, 10, idx]
#     # imgs = test_net.weights['01'].T @ rep_save[:, 10, idx]
#     axs.imshow(imgs.reshape(32, 32), cmap='Blues', vmin=0, vmax=1)
#     axs.axis('off')
#     # axs.set_title()
#
#     fig.tight_layout()
#     fig.show()
#     fig.savefig(f'/home/kwangjun/PycharmProjects/si_pc/cifar10/figures/fig4/exampleImg_npe_{titles[idx_i]}.png', dpi=300, bbox_inches='tight')


# # curve fitting f vs tau
# from scipy.optimize import curve_fit
#
# def func(x, a, b, c):
#
#     return a * np.exp(-b * x) + c
#
# x = [2, 4, 10, 20, 40, 100, 200]
# y = [70, 30, 12, 6, 3, 1.2, 0.6]
#
# popt, _ = curve_fit(func, x, y)
# fig, axs = plt.subplots(1,1)
# axs.scatter(x, y, c='r', s=50)
# axs.plot(x, func(np.array(x), *popt), c='k', lw=2)
# axs.spines['top'].set_visible(False)
# axs.spines['right'].set_visible(False)
# axs.set_xlabel(r'$\tau$')
# axs.set_ylabel(r'$f$ (Hz)')
# fig.show()

# # supp fig: interneurons oscillations
# interneuron_colors = ['#FF2F92', '#F90000', '#FF6666', '#00B0F0', '#0432FF', '#76ADEE']
# interneuron_color_dict = {key: interneuron_colors[i] for i, key in enumerate(interneuron_dict)}
# fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(10, 5))
# for i, (key, grp) in enumerate(interneuron_dict.items()):
#     for img_i in range(16):
#         axs.flat[i].plot(grp[:, img_i, :].mean(axis=0), c=interneuron_color_dict[key], alpha=0.5)
#     axs.flat[i].set_title(" ".join(key.split('_')))
#     axs.flat[i].spines['right'].set_visible(False)
#     axs.flat[i].spines['top'].set_visible(False)
# fig.tight_layout()
# fig.show()
#
# import matplotlib.gridspec as gridspec
#
# interneuron_keys = [key for key in interneuron_dict]
# ranges = [slice(0, 2000), slice(9000, 10000)]
# fig, axs = plt.subplots(nrows=2, ncols=6, sharex=True, figsize=(10, 5))
# for sub_ax_i, sub_ax in enumerate(axs.flat):
#     key_i = interneuron_keys[sub_ax_i // 2]
#     range_i = ranges[sub_ax_i % 2]
#     curr_interneuron = interneuron_dict[key_i]
#     for img_i in range(16):
#         sub_ax.plot(curr_interneuron[:, img_i, :].mean(axis=0)[range_i], c=interneuron_color_dict[key_i], alpha=0.5)
#         sub_ax.spines['right'].set_visible(False)
#         sub_ax.spines['top'].set_visible(False)
#         if sub_ax_i % 2 == 1:
#             sub_ax.spines['left'].set_visible(False)
#             sub_ax.set_yticks([])
#         sub_ax.set_ylim([0, 1.0])
# fig.tight_layout()
# fig.show()
#
# # seperate panels
# plt.close('all')
# interneuron_keys = [key for key in interneuron_dict]
# ranges = [slice(0, 2000), slice(9000, 10000)]
# gs_kw = dict(width_ratios=[2, 1])
#
# for key in interneuron_keys:
#     fig, axs = plt.subplots(ncols=2, nrows=1, gridspec_kw=gs_kw)
#     for sub_ax_i, sub_ax in enumerate(axs.flat):
#         range_i = ranges[sub_ax_i % 2]
#         curr_interneuron = interneuron_dict[key]
#         for img_i in range(16):
#             sub_ax.plot(curr_interneuron[:, img_i, :].mean(axis=0)[range_i], c=interneuron_color_dict[key], alpha=0.5)
#             sub_ax.spines['right'].set_visible(False)
#             sub_ax.spines['top'].set_visible(False)
#             if sub_ax_i == 1:
#                 sub_ax.spines['left'].set_visible(False)
#                 sub_ax.set_yticks([])
#             sub_ax.set_ylim([0, 1.2])
# # fig.subplots_adjust(wspace=0.2, hspace=0.5)
#     fig.show()
