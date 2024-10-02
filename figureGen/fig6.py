from script.opto_expt import test_opto, run_opto
from script.tools import load_sim_data, sample_imgs
import numpy as np
import os


model_dir = '../results/trial03/'
sim_params, weights, dataset = load_sim_data(model_dir)

sample_idcs = sample_imgs(dataset['test_y'], n_class=10, n_sample=3, class_choice=None)
np.random.shuffle(sample_idcs)
test_x = dataset['test_x'][sample_idcs]
test_y = dataset['test_y'][sample_idcs]

# specify output directory
save_dir = model_dir + 'opto/'
# create the output directory, if it does not already exist
if os.path.exists(save_dir):
    pass
else:
    os.mkdir(save_dir)


# silence a particular type of neuron: e.g., PV
response_fig, recon_fig = run_opto(target_neuronType='pv', simParams=sim_params, weights=weights, imgData=test_x)

# loop over all neuron types
neuron_types = [(None, None), ('ppe_pyr', 'npe_pyr'), ('ppe_pv', 'npe_pv'), ('ppe_sst', 'npe_sst'), ('ppe_vip', 'npe_vip')]
for n_pairs in neuron_types:
    opto_net, recon_fig, response_fig, _, input_pad, pred_pad = test_opto(
        simul_params=sim_params, learned_weights=weights,
        img_data=test_x, img_label=test_y,
        silence_target=[('layer_0', n_pairs[0]), ('layer_0', n_pairs[1])],
        record_var='all',
        rdm=False
    )

    if n_pairs[0] == None:
        fig_name = 'none'
    else:
        fig_name = n_pairs[0].split('_')[1]
recon_fig.savefig(save_dir + f'{fig_name}_recon.png', dpi=300, bbox_inches='tight')
response_fig.savefig(save_dir + f'{fig_name}_response.png', dpi=300, bbox_inches='tight')