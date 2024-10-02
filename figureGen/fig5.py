from script.tools import create_dir, load_sim_data
# from script.oddball_expt import oddball_simulation, vary_devLoc_and_get_amplitudes, compute_mmn
from script.oddball_expt import oddball_expt


# define model directory
model_dir = '../results/trial03/'
# create output directory
save_dir = model_dir + 'oddball/'
create_dir(save_dir)

# load simulation results
sim_params, weights, dataset = load_sim_data(model_path=model_dir)

# initialize network for an oddball experiment
sequence_length = 5
deviant_stim_loc = 3
oddball_net = oddball_expt(
    len_seq=sequence_length, loc_dev=sequence_length,
    simParams=sim_params, pretrained_weights=weights, dataset=dataset
)

# fig 5A and B: oddball sequence and response of pyramidal neurons
sequence_fig, pyr_response_fig = oddball_net.simulate(
    sim_type='test_different', record='all'
)
sequence_fig.show()
pyr_response_fig.show()

# supp fig: mismatch negativity
mmn_fig = oddball_net.compute_mmn(signal_type=('layer_0', 'ppe_pyr'))
mmn_fig.show()

# fig 5C:
max_amplitudes, amp_fig, dev_response_fig = oddball_net.vary_devLoc_and_get_amplitudes()
# reset dev loc
oddball_net.loc_deviant = deviant_stim_loc

# fig 5D: response of interneurons
_, int_response_fig = oddball_net.simulate(
    sim_type='test_different', record='interneurons'
)
# increase figure width for better visualization
int_response_fig.set_figwidth(12)
int_response_fig.show()
