from figureGen.training_data import sim_params, weights, dataset
from script.oddball import oddball_simulator


# define the length of oddball sequence
sequence_length = 7
# define the position of deviant stimulus in the sequence
deviant_stim_pos = 4
# initialize network for an oddball experiment
oddball_net = oddball_simulator(
    len_seq=sequence_length, pos_dev=deviant_stim_pos,
    sim_params=sim_params, pretrained_weights=weights, dataset=dataset,
    dev_stim_type='testSet_differentClass'
)

# fig 5A: oddball sequence
oddball_net.simulate(record='all', show_sequence=True)
# fig 5B: population mean responses of pyramidal neurons in PE+, PE-, and Rep circuits
pyr_response_fig = oddball_net.plot_pop_responses(target='pyramidal')

# S4 Fig B
oddball_net.simulate(record='all', pfp_str=0.3, omit=True, show_sequence=True)
omit_pfp_pyr_response_fig = oddball_net.plot_pop_responses(target='pyramidal')

# S4 Fig C
oddball_net.simulate(record='all', pfp_str=0.3, omit=False, show_sequence=True)
pfp_pyr_response_fig = oddball_net.plot_pop_responses(target='pyramidal')

# fig 5C:
max_amplitudes, amp_fig = oddball_net.vary_dev_loc_and_get_peaks()
