import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from tqdm import trange
from script.pc_network import pickle_load, network

# dir
project_dir = '/home/kwangjun/PycharmProjects/si_pc/'
model_dir = project_dir + 'gamma_test/'

# load data
dataset = pickle_load(model_dir, 'dataset.pkl')
sim_params = pickle_load(model_dir, 'sim_params.pkl')
weights = pickle_load(model_dir, 'weights.pkl')

n_sample, n_neuron = dataset['train_x'].shape

# simulation
# create network
net = network(
    neurons_per_layer=sim_params['net_size'],
    bu_rate=sim_params['bu_rate'], td_rate=sim_params['td_rate'],
    tau_exc=sim_params['tau_exc'], tau_inh=sim_params['tau_inh'],
    symm_w=sim_params['symmetric_weight'], pretrained_weights=weights,
    bg_exc=0.0, bg_inh=0.0, jitter_lvl=0.0, jitter_type='constant'
)
# initialize network
net.initialize_network(batch_size=n_sample)
# reset error
net.initialize_error()
# set simulation time
t_infer = int(sim_params['sim_time'] / 1e-3)
# create template for activity
rep_r_save = np.zeros((*net.network['layer_1']['rep_r'].shape, t_infer))
rep_ppe_save = np.zeros((*net.network['layer_0']['ppe_pyr'].shape, t_infer))
rep_npe_save = np.zeros((*net.network['layer_0']['npe_pyr'].shape, t_infer))
recon_save = np.zeros((n_neuron, t_infer))

# pick a sample
sample_idx = np.random.choice(np.arange(n_sample))

for t in trange(t_infer):
    net.compute(dataset['train_x'].T)
    rep_r_save[:, :, t] = net.network['layer_1']['rep_r']
    rep_ppe_save[:, :, t] = net.network['layer_0']['ppe_pyr']
    rep_npe_save[:, :, t] = net.network['layer_0']['npe_pyr']
    recon_save[:, t] = (net.weights['01'].T @ net.network['layer_1']['rep_r'])[:, sample_idx]
    plt.show()

plt.close('all')
# activity plot
recon_idcs = np.arange(0, 24, 5.5)
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
for i in range(1):
    axs[0].plot(rep_r_save[:, i, :].mean(axis=0), c='purple', label='rep')
    axs[1].plot(rep_ppe_save[:, i, :].mean(axis=0), c='r', label=r'PE$+$')
    axs[2].plot(rep_npe_save[:, i, :].mean(axis=0), c='b', label=r'PE$-$')
for ax_i, ax in enumerate(axs.flat):
    for idx in recon_idcs:
        ax.axvline(x=idx, ls='--', c='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax_i < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('time (ms)')
    if ax_i == 1:
        ax.set_ylabel('firing rate (a.u.)')
fig.legend()
fig.show()

# # if you want it as individual figures
# plt.close('all')
# recon_idcs = np.arange(0, 24, 5.5)
# vals = [rep_r_save[:, i, :], rep_ppe_save[:, i, :], rep_npe_save[:, i, :]]
# cs = ['purple', 'r', 'b']
# labels = ['rep', r'PE$+$', r'PE$-$']
#
# for i in range(3):
#     fig, axs = plt.subplots(1, 1, figsize=(10, 2))
#     axs.plot(vals[i].mean(axis=0), c=cs[i], label=labels[i])
#     axs.spines['top'].set_visible(False)
#     axs.spines['right'].set_visible(False)
#     for idx in recon_idcs:
#         axs.axvline(x=idx, ls='--', c='black')
#     fig.legend()

# recon plot
recon_fig, recon_axs = plt.subplots(nrows=2, ncols=10)
for i, ax in enumerate(recon_axs.flat):
    ax.imshow(recon_save[:, recon_idcs[i]].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
recon_fig.show()

# ppe plot
ppe_ex = rep_ppe_save[:, sample_idx, :]
ppe_fig, ppe_axs = plt.subplots(nrows=2, ncols=10)
for i, ax in enumerate(ppe_axs.flat):
    ax.imshow(ppe_ex[:, recon_idcs[i]].reshape(28, 28), cmap='Reds', vmin=0, vmax=1)
    ax.axis('off')
ppe_fig.show()

# npe plot
npe_ex = rep_npe_save[:, sample_idx, :]
npe_fig, npe_axs = plt.subplots(nrows=2, ncols=10)
for i, ax in enumerate(ppe_axs.flat):
    ax.imshow(npe_ex[:, recon_idcs[i]].reshape(28, 28), cmap='Blues', vmin=0, vmax=1)
    ax.axis('off')
ppe_fig.show()

# plot
plt.close('all')
fig = plt.figure(figsize=(10, 8))
spec = gridspec.GridSpec(nrows=3, ncols=4, figure=fig)
f_ax_rep_response = fig.add_subplot(spec[0, :3])
f_ax_rep_response.set_title('Rep')
f_ax_rep_img = fig.add_subplot(spec[0, 3])

f_ax_ppe_response = fig.add_subplot(spec[1, :3])
f_ax_ppe_response.set_title('PE+')
f_ax_ppe_response.set_ylabel('population response', rotation=90)
f_ax_ppe_img = fig.add_subplot(spec[1, 3])

f_ax_npe_response = fig.add_subplot(spec[2, :3])
f_ax_npe_response.set_title('PE-')
f_ax_npe_img = fig.add_subplot(spec[2, 3])

f_ax_rep_response.axes.spines['top'].set_visible(False)
f_ax_rep_response.spines['right'].set_visible(False)
f_ax_rep_response.set_xticks([])
f_ax_rep_response.set_xticklabels('')

f_ax_ppe_response.spines['top'].set_visible(False)
f_ax_ppe_response.spines['right'].set_visible(False)
f_ax_ppe_response.set_xticks([])
f_ax_ppe_response.set_xticklabels('')

f_ax_npe_response.spines['top'].set_visible(False)
f_ax_npe_response.spines['right'].set_visible(False)
f_ax_npe_response.set_xlabel('time (ms)')

f_ax_rep_img.axis('off')
f_ax_ppe_img.axis('off')
f_ax_npe_img.axis('off')

y1 = rep_npe_save[:, 0, :].mean(axis=0)
y2 = rep_ppe_save[:, 0, :].mean(axis=0)
y3 = rep_npe_save[:, 0, :].mean(axis=0)
x = np.arange(len(y1))

ims = []
for time in np.arange(0, 200, 1):
    line1, = f_ax_rep_response.plot(y1[0:time], c='purple')
    line2, = f_ax_ppe_response.plot(y2[0:time], c='red')
    line3, = f_ax_npe_response.plot(y3[0:time], c='blue')

    im1 = f_ax_rep_img.imshow(recon_save[:, time].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    im2 = f_ax_ppe_img.imshow(ppe_ex[:, time].reshape(28, 28), cmap='Reds', vmin=0, vmax=1)
    im3 = f_ax_npe_img.imshow(npe_ex[:, time].reshape(28, 28), cmap='Blues', vmin=0, vmax=1)

    ims.append([line1, line2, line3, im1, im2, im3])

ani = animation.ArtistAnimation(fig, ims, interval=10, blit=False)
ani.save('animation_drawing_full.gif', writer='imagemagick', fps=10)
