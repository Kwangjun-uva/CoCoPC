import numpy as np
import matplotlib.pyplot as plt
import rsatoolbox
from script.pc_network import network, generate_reconstruction_pad, pickle_load, sample_imgs
from tqdm import trange

def test_opto(
        simul_params, learned_weights,
        img_data, img_label,
        silence_target,
        record_var,
        rdm=False
):
    n_sample, n_neuron = img_data.shape
    t_sim = int(simul_params['sim_time'] / 1e-3)
    t_isi = int(simul_params['isi_time'] / 1e-3)

    opto_network = network(
        neurons_per_layer=simul_params['net_size'],
        pretrained_weights=learned_weights,
        bu_rate=simul_params['bu_rate'], td_rate=simul_params['td_rate'],
        tau_exc=simul_params['tau_exc'], tau_inh=simul_params['tau_inh'],
        symm_w=simul_params['symmetric_weight'],
        bg_exc=0.0, bg_inh=0.0, jitter_lvl=0.0, jitter_type='constant'
    )

    # set silence target
    # target_layer = 'layer_0'
    if silence_target == None:
        pass
    else:
        for target in silence_target:
            target_layer, target_neuron = target
            opto_network.activate_silencer(target_layer=target_layer, target_neuron=target_neuron)

    # set computor
    computor = opto_network.compute

    # initialize network
    opto_network.initialize_network(batch_size=n_sample)
    opto_network.initialize_error()

    # isi
    for _ in trange(t_isi):
        computor(inputs=np.zeros(img_data.T.shape), record=record_var)

    # rep_save = np.zeros((*opto_network.network['layer_1']['rep_r'].shape, t_sim))
    # stimulus presentation
    for _ in trange(t_sim):
        computor(inputs=img_data.T, record=record_var)
        # rep_save += model.network['layer_1']['rep_r']
        # rep_save[:, :, t_step] = opto_network.network['layer_1']['rep_r']

    layer = 0
    input_fr = opto_network.network[f'layer_{layer}']['rep_e']
    pred_fr = opto_network.weights[f'{layer}{layer + 1}'].T @ opto_network.network[f'layer_{layer + 1}']['rep_r']

    # plot reconstruction
    _, _, input_pad = generate_reconstruction_pad(img_mat=input_fr, nx=2)
    _, _, pred_pad = generate_reconstruction_pad(img_mat=pred_fr, nx=2)

    reconst_fig, recon_axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
    plt_min = np.min(input_fr)
    plt_max = np.max(input_fr)
    recon_min = np.min(pred_fr)
    recon_max = np.max(pred_fr)
    input_imgs = recon_axs[0].imshow(input_pad, cmap='gray', vmin=plt_min, vmax=plt_max)
    reconst_fig.colorbar(input_imgs, ax=recon_axs[0], shrink=0.4)
    recon_axs[0].set_title('input')
    pred_imgs = recon_axs[1].imshow(pred_pad, cmap='gray', vmin=recon_min, vmax=recon_max)
    reconst_fig.colorbar(pred_imgs, ax=recon_axs[1], shrink=0.4)
    recon_axs[1].set_title('prediction')
    for ax in recon_axs.flatten():
        ax.axis('off')

    reconst_fig.suptitle(f'reconstruction at layer {layer}')
    reconst_fig.tight_layout()

    total_t = len(opto_network.errors['layer_0']['ppe_pyr'])
    if record_var == 'error':
        # plot errors
        response_fig, response_axs = plt.subplots(1, 1)
        response_axs.plot(opto_network.errors['layer_0']['ppe_pyr'], c='red', label='PE+')
        response_axs.plot(opto_network.errors['layer_0']['npe_pyr'], c='blue', label='PE-')
        response_axs.spines['top'].set_visible(False)
        response_axs.spines['right'].set_visible(False)
        for yi in np.arange(0, total_t, t_sim + t_isi):
            ax.axvline(x=yi, ls='--', c='black')
            ax.axvline(x=yi + t_isi, ls='--', c='black')
        leg = response_axs.legend()
        for h, t in zip(leg.legendHandles, leg.get_texts()):
            t.set_color(h.get_facecolor()[0])

    elif record_var == 'all':
        # plot rep + errors
        response_fig, response_axs = plt.subplots(3, 1)
        response_axs[0].plot(opto_network.errors['layer_1']['rep_r'], c='purple', label='rep')
        response_axs[1].plot(opto_network.errors['layer_0']['ppe_pyr'], c='red', label='PE+')
        response_axs[2].plot(opto_network.errors['layer_0']['npe_pyr'], c='blue', label='PE-')
        for ax_i, ax in enumerate(response_axs.flat):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for yi in np.arange(0, total_t, t_sim + t_isi):
                ax.axvline(x=yi, ls='--', c='black')
                ax.axvline(x=yi + t_isi, ls='--', c='black')
            if ax_i == 2:
                ax.set_xlabel('time (ms)')
            else:
                ax.set_xlabel('')
            if ax_i == 1:
                ax.set_ylabel('Firing rate (Hz)')
            else:
                ax.set_ylabel('')
        leg = response_fig.legend()
        for h, t in zip(leg.legendHandles, leg.get_texts()):
            t.set_color(h.get_color())
    else:
        response_fig = None

    # recon_fig.savefig(model_dir + f'opto/{target_neuron}_recon_scaled.png', dpi=300)

    if rdm:
        reordering_idx = []
        for yi in np.unique(img_label):
            reordering_idx.append(np.argwhere(img_label == yi).flatten())

        # label
        opto_reps = opto_network.network['layer_1']['rep_r'][:, reordering_idx]
        opto_inputs = opto_network.network['layer_0']['rep_e'][:, reordering_idx]
        opto_labels = img_label[reordering_idx]

        rdm_list = []

        n_neuron, n_sample = opto_reps.shape
        obs_desc = {'digits': np.array([str(x) for x in opto_labels])}
        # construct dataset for RDM
        chn_desc = {'neurons': np.array([f'neuron_{str(x + 1)}' for x in np.arange(n_neuron)])}

        rdm_list.append(
            rsatoolbox.data.Dataset(
                measurements=opto_inputs.T,
                descriptors={'Area': 'input'},
                obs_descriptors=obs_desc,
                channel_descriptors=chn_desc
            )
        )

        rdm_list.append(
            rsatoolbox.data.Dataset(
                measurements=opto_reps.T,
                descriptors={r'Area': 'Area 1, $r_{VIP-} = 0$'},
                obs_descriptors=obs_desc,
                channel_descriptors=chn_desc
            )
        )

        rdm_corr = rsatoolbox.rdm.calc_rdm(
            rdm_list,
            method='correlation'
        )

        # RDM plot
        opto_rdms, rdm_axs, rdm_dict = rsatoolbox.vis.show_rdm(rdm_corr,
                                                               cmap='RdBu', show_colorbar='figure',
                                                               rdm_descriptor='Area',
                                                               n_column=2)
    else:
        opto_rdms = None

    return opto_network, reconst_fig, response_fig, opto_rdms, input_pad, pred_pad


projet_dir = '/home/kwangjun/PycharmProjects/si_pc/'
model_dir = projet_dir + 'cifar10/trial05/'
# model_dir = projet_dir + 'fmnist/trial01/'

dataset = pickle_load(model_dir, 'dataset.pkl')
sim_params = pickle_load(model_dir, 'sim_params.pkl')
weights = pickle_load(model_dir, 'weights.pkl')

sample_idcs = sample_imgs(dataset['test_y'], n_class=10, n_sample=3, class_choice=None)
np.random.shuffle(sample_idcs)
test_x = dataset['test_x'][sample_idcs]
test_y = dataset['test_y'][sample_idcs]

fig_save_dir = model_dir + 'opto/'
# neuron_types = [(None, None), ('ppe_pyr', 'npe_pyr'), ('ppe_pv', 'npe_pv'), ('ppe_sst', 'npe_sst'), ('ppe_vip', 'npe_vip')]
# for n_pairs in neuron_types:
#     opto_net, recon_fig, response_fig, _, input_pad, pred_pad = test_opto(
#         simul_params=sim_params, learned_weights=weights,
#         img_data=test_x, img_label=test_y,
#         silence_target=[('layer_0', n_pairs[0]), ('layer_0', n_pairs[1])],
#         record_var='all',
#         rdm=False
#     )
#
#     if n_pairs[0] == None:
#         fig_name = 'none'
#     else:
#         fig_name = n_pairs[0].split('_')[1]
    # recon_fig.savefig(fig_save_dir + f'{fig_name}_recon.png', dpi=300, bbox_inches='tight')
    # response_fig.savefig(fig_save_dir + f'{fig_name}_response.png', dpi=300, bbox_inches='tight')

# work on rep pop response!

target_neuron_type = 'pv'
if target_neuron_type == 'none':
    ppe_target = None
    npe_target = None
else:
    ppe_target = f'ppe_{target_neuron_type}'
    npe_target = f'npe_{target_neuron_type}'

opto_net, recon_fig, rep_fig, _, input_img, pred_img = test_opto(
        simul_params=sim_params, learned_weights=weights,
        img_data=test_x, img_label=test_y,
        # silence_target=[('layer_0', 'ppe_vip'), ('layer_0', 'npe_vip')],
        silence_target=[('layer_0', ppe_target), ('layer_0', npe_target)],
        record_var='all',
        rdm=False
    )
rep_fig.show()


total_t = len(opto_net.errors['layer_0']['ppe_pyr'])
t_sim = int(sim_params['sim_time'] / 1e-3)
t_isi = int(sim_params['isi_time'] / 1e-3)

font = {'family': 'Arial',
        'color':  'darkred',
        'weight': 'normal',
        'size': 20,
        'color': 'black'
        }

plt.close('all')
response_fig, response_axs = plt.subplots(1,1, figsize=(6, 3))
response_axs.plot(opto_net.errors['layer_0']['ppe_pyr'][t_isi:], lw=2.0, c='r')
response_axs.plot(opto_net.errors['layer_0']['npe_pyr'][t_isi:], lw=2.0, c='b')
response_axs.plot(np.arange(0, 101), -0.1 * np.ones(101), linewidth=5, c='black')
response_axs.spines['top'].set_visible(False)
response_axs.spines['right'].set_visible(False)
response_axs.spines['left'].set_visible(False)
response_axs.spines['bottom'].set_visible(False)
response_axs.text(-60, -0.1, '100 ms', fontdict=font)
response_axs.set_yticks([])
response_axs.set_xticks([])
# for yi in np.arange(0, total_t, t_sim + t_isi):
#     ax.axvline(x=yi, ls='--', c='black')
#     ax.axvline(x=yi + t_isi, ls='--', c='black')
# response_axs.legend()
response_fig.tight_layout()
# response_fig.savefig(f'/home/kwangjun/PycharmProjects/si_pc/cifar10/fig6_{target_neuron_type}_osc.png', dpi=300, bbox_inches='tight')
response_fig.show()

recon_none = opto_net.weights['01'].T @ opto_net.network['layer_1']['rep_r']
_, _, pred_pad = generate_reconstruction_pad(img_mat=recon_none, nx=6)
fig, axs = plt.subplots(1, 1)
axs.imshow(pred_pad, cmap='gray', vmin=0, vmax=1)
axs.axis('off')
# fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/fig6_recon_{target_neuron_type}.png', dpi=300, bbox_inches='tight')
fig.show()