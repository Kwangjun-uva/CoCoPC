import numpy as np
from script.pc_network import network
from script.tools import load_sim_data
from script.oscillations import (get_activity_log, plot_interneuron_activity, plot_pyr_activity, plot_cutoff_pyr,
                                 plot_phasePortraits, plot_autocorrelation, plot_phase_autocorr,
                                 plot_cutoff_interneurons, get_taus)


# specify the model dir
# model_dir = '/home/kwangjun/PycharmProjects/si_pc/cifar10/trial03/'
model_dir = '../results/trial03/'

# load simulation parameters, weights, and dataset
simParams, weights, dataset = load_sim_data(model_path=model_dir)

# pick random samples
nSample = 16
ran_idcs = np.random.choice(a=len(dataset['test_x']), size=nSample, replace=False)
dataset_select = dataset['test_x'][ran_idcs]

# set simulation time: default has sim time of 2 sec and isi time of 0.2 sec
simParams['sim_time'] *= 1
simParams['isi_time'] *= 0.1

# set sim steps
steps_isi = int(simParams['isi_time'] / simParams['dt'])
steps_sim = int(simParams['sim_time'] / simParams['dt'])

# initialize network
test_net = network(simParams=simParams, pretrained_weights=weights)
test_net.initialize_network(batch_size=dataset_select.shape[0])
test_net.initialize_error()

# simulate and save activity of all neurons at every time step
activity_dict = get_activity_log(
    ntwork=test_net,
    steps_isi=steps_isi, steps_sim=steps_sim,
    dataset_select=dataset_select
)

# # fig 4A
# fig = plot_pyr_activity(activity_dict=activity_dict, take_mean=False)
# fig.show()
#
# # fig 4A: cut off in the middle
# cutoff_pyr_fig = plot_cutoff_pyr(activity_dict=activity_dict, first_range=(0, 1000), second_range=(9000, 10000))
# cutoff_pyr_fig.show()

# fig 4B: phase plot in rows
phase_fig, phase_v = plot_phasePortraits(activity_dict=activity_dict, cut=2000)
phase_fig.show()

# fig 4B: compute autocorrelation and estimate frequency
autocorr_fig, freqs = plot_autocorrelation(activity_dict=activity_dict, sim_params=simParams, cut=2000)
autocorr_fig.show()

# fig 4B:
phase_autocorr_plot = plot_phase_autocorr(activity_dict, simParams, cut=2000)
phase_autocorr_plot.show()

# # fig 4C
# tau_multiples = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
# fpt, tau_fig = get_taus(
#     tau_multiples=tau_multiples,
#     sim_params=simParams, weights=weights, data_select=dataset_select,
#     steps_isi=steps_isi, steps_sim=steps_sim
# )
#
# # fig 4D
# fig = plot_interneuron_activity(activity_dict=activity_dict, take_mean=False)
# fig.show()
#
# # fig 4D: cut off in the middle
# cutoff_interneurons_fig = plot_cutoff_interneurons(activity_dict=activity_dict, first_range=(0, 1000),
#                                                    second_range=(9000, 10000))
# cutoff_interneurons_fig.show()
