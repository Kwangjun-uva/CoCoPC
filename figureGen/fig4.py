from figureGen.training_data import sim_params, weights, dataset
from script.oscillation import oscillation_simulator

# initialize network
net = oscillation_simulator(sim_params=sim_params, weights=weights, dataset=dataset)

# simulate
net.simulate(simT=7)

# set ranges to plot
range1 = (0, 1000)
range2 = (9000, 10000)
range_high_resolution = (180, 450)

# Fig 4A
pyr_cutoff_plot = net.plot_cutoffPlot_pyr(first_range=range1, second_range=range2)

# Fig 4B
pyr_high_resolution = net.plot_pyr_activity_across_images(t_range=range_high_resolution)

# Fig 4C
phase_autocorr_plot = net.plot_phase_autocorr()

# S1 Fig: cut off in the middle
cutoff_interneurons_fig = net.plot_cutoffPlot_interneurons(first_range=range1, second_range=range2)

# Fig 4D
taus = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
fpt, tau_plot = net.plot_freq_across_taus(simT=2, tau_multiples=taus)
