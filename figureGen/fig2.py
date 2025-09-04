import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from figureGen.training_data import sim_params, weights, dataset
from script.pc_network import network
from script.tools import pickle_load, plot_training_errors, generate_reconstruction_pad


def test_reconstruction(sim_param, weights, inputs):

    # Reshape input_vector = (n_sample, n_pixel) -> test_samples = (n_pixel, n_sample)
    sample_idcs = sim_param['recon_img_idcs']
    test_samples = inputs[sample_idcs].T
    # if a single image is presented, make it a (n_pixel, 1) vector for matrix operations
    if test_samples.shape == 1:
        test_samples = test_samples.reshape(test_samples.shape[0], 1)

    # simulate
    test_net = network(simParams=sim_param, pretrained_weights=weights)
    # initialize network
    test_net.initialize_network(batch_size=test_samples.shape[1])
    # reset error
    test_net.initialize_activity_log()

    # isi
    steps_isi = int(sim_param['isi_time'] / sim_param['dt'])
    for _ in trange(steps_isi):
        test_net.compute(inputs=np.zeros(test_samples.shape), record='error')

    # stimulus presentation
    steps_sim = int(sim_param['sim_time'] / sim_param['dt'])
    # rep saver
    rep_save = np.zeros((*test_net.network['layer_1']['rep_r'].shape, steps_sim))
    # sim
    for t_step in trange(steps_sim):
        test_net.compute(inputs=test_samples, record='error')
        # rep_save += model.network['layer_1']['rep_r']
        rep_save[:, :, t_step] = test_net.network['layer_1']['rep_r']

    # input_fr = pred_fr = (n_pixel, n_sample)
    input_fr = test_net.network['layer_0']['rep_e']
    pred_fr = test_net.weights['01'].T @ test_net.network[f'layer_1']['rep_r']

    # plot reconstruction
    nx = len(sample_idcs) // 5
    _, _, input_pad = generate_reconstruction_pad(img_mat=input_fr, nx=nx)
    _, _, pred_pad = generate_reconstruction_pad(img_mat=pred_fr, nx=nx)

    recon_fig, recon_axs = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
    input_imgs = recon_axs[0].imshow(input_pad, cmap='gray', vmin=0.0, vmax=1.0)
    recon_fig.colorbar(input_imgs, ax=recon_axs[0], shrink=0.4)
    recon_axs[0].set_title('input')
    pred_imgs = recon_axs[1].imshow(pred_pad, cmap='gray', vmin=0.0, vmax=1.0)
    recon_fig.colorbar(pred_imgs, ax=recon_axs[1], shrink=0.4)
    recon_axs[1].set_title('prediction')
    for ax in recon_axs.flatten():
        ax.axis('off')

    recon_fig.suptitle(f'Reconstructions')
    recon_fig.tight_layout()
    recon_fig.show()

    return recon_fig, test_net

# define model directory
model_dir = '../results/'
# load training errors
training_errors = pickle_load(save_path=model_dir, file_name='errors.pkl')

# fig 2B: training errors
error_fig = plot_training_errors(sim_params=sim_params, errs=training_errors)
error_fig.show()

# fig 2C: reconstruction
recon_fig, _ = test_reconstruction(sim_param=sim_params, weights=weights, inputs=dataset['test_x'])