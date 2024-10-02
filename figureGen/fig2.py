from script.pc_network import plot_training_errors, test_reconstruction
from script.tools import load_sim_data, pickle_load


# define model directory
# model_dir = '../results/trial03/'
model_dir = 'C:/Users/grand/OneDrive - UvA/project_02/cifar_results/data/trial05/'
# load simulation results
simParams, weights, dataset = load_sim_data(model_path=model_dir)
training_errors = pickle_load(save_path=model_dir, file_name='errors.pkl')

# fig 2B: training errors
error_fig = plot_training_errors(sim_params=simParams, errs=training_errors)
error_fig.show()

# # fig 2C: reconstruction
# recon_fig, _, _ = test_reconstruction(sim_param=simParams, weights=weights, layer=0, input_vector=dataset['test_x'], sample_idcs=simParams['recon_img_idcs'], record=None)
# recon_fig.show()