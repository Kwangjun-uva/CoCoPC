import numpy as np
from script.pc_network import test_noise
from script.noise_simulator import get_sample_imgs, get_recons, get_pop_response, plot_mse_ssim
from script.tools import load_sim_data, sample_imgs, create_dir

# specify model directory
# model_dir = '../results/trial03/'
model_dir = 'C:/Users/grand/OneDrive - UvA/project_02/cifar_results/data/trial03/'

# specify output directory
save_dir = model_dir + 'noise/'
# create directory
create_dir(save_dir)

# load trained params
simParams, weights, dataset = load_sim_data(model_path=model_dir)
# # for quick test
# simParams['isi_time'] = 0.05
# simParams['sim_time'] = 0.5

# specify noise level range
noise_lvls = np.linspace(start=0, stop=0.2, num=11, endpoint=0.2)

# pick random samples to simulate
# ran_idcs = np.random.choice(a=len(dataset['test_x']), size=100, replace=False)
ran_idcs = sample_imgs(dataset['test_y'], n_class=10, n_sample=2)
np.random.shuffle(ran_idcs)

# simulate: external noise
noiseExt_net, _, noiseExt_mse, noiseExt_ssim, noiseExt_imgs = test_noise(
    sim_param=simParams, pretrained_weights=weights,
    noise_levels=noise_lvls, noise_type=('ext', 'normal'),
    test_images=dataset['test_x'], test_sample_idcs=ran_idcs
)
# simulate: internal noise
noiseInt_net, _, noiseInt_mse, noiseInt_ssim, noiseInt_imgs = test_noise(
    sim_param=simParams, pretrained_weights=weights,
    noise_levels=noise_lvls, noise_type=('int', 'normal'),
    test_images=dataset['test_x'], test_sample_idcs=ran_idcs
)

# Fig 3A top panel: sample images
sample_fig, sample_img_dict = get_sample_imgs(img_dict=noiseExt_imgs, sigma_ext=noise_lvls[2])

# Fig 3A&B: reconstructions with varying sigma vals
noiseExt_recon_fig = get_recons(noise_lvls=noise_lvls, img_dict=noiseExt_imgs)
noiseExt_recon_fig.show()
noiseInt_recon_fig = get_recons(noise_lvls=noise_lvls, img_dict=noiseInt_imgs)
noiseInt_recon_fig.show()

# Fig 3A&B: MSE and SSIM plots
noiseExt_metricFig = plot_mse_ssim(noise_origin='ext', noise_lvls=noise_lvls, mse_dict=noiseExt_mse,
                                   ssim_dict=noiseExt_ssim)
noiseExt_metricFig.show()
noiseInt_metricFig = plot_mse_ssim(noise_origin='int', noise_lvls=noise_lvls, mse_dict=noiseInt_mse,
                                   ssim_dict=noiseInt_ssim)
noiseInt_metricFig.show()

# suppFig: population responses with noise
noiseExt_popFig, noiseExt_popFr = get_pop_response(
    noise_origin='ext', noise_val=0.1, sim_params=simParams,
    pretrained_weights=weights, dataset=dataset, ran_idcs=ran_idcs)
noiseExt_popFig.show()

noiseInt_popFig, noiseInt_popFr = get_pop_response(
    noise_origin='ext', noise_val=0.1, sim_params=simParams,
    pretrained_weights=weights, dataset=dataset, ran_idcs=ran_idcs)
noiseInt_popFig.show()