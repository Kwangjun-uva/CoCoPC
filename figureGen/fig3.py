import numpy as np

from figureGen.training_data import sim_params, weights, dataset
from script.noise import noise_simulator


# specify noise level range
noise_levels = np.linspace(start=0.0, stop=0.2, num=11, endpoint=0.2)

# initialize network
external_noise_sim = noise_simulator(sim_params=sim_params, weights=weights, dataset=dataset)
internal_noise_sim = noise_simulator(sim_params=sim_params, weights=weights, dataset=dataset)

# external noise simulation
external_noise_sim.run_experiment(noise_source='ext', noise_dist='normal', noise_levels=noise_levels)
# internal noise simulation
internal_noise_sim.run_experiment(noise_source='int', noise_dist='normal', noise_levels=noise_levels)

# show reconstructions
ext_noise_recon_fig = external_noise_sim.show_reconstruction_across_noise_level()
int_noise_recon_fig = internal_noise_sim.show_reconstruction_across_noise_level()

# Fig 3A&B: MSE and SSIM plots
ext_noise_metricFig = external_noise_sim.plot_mse_ssim()
int_noise_metricFig = internal_noise_sim.plot_mse_ssim()
