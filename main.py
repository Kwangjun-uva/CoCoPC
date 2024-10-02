from script.tools import create_dir
from script.pc_network import run_simulation, generate_input

model_dir = 'results/test_run/'
# create output directory
create_dir(model_dir)

# define simulation parameters
# sim_params = {
#     'model_dir': model_dir,
#     'dataset': 'gray_cifar-10',  # choose from ['mnist', 'fmnist', 'gray_cifar-10']
#     'n_class': 10,
#     'n_sample': 256,
#     'sim_time': 2.0,
#     'isi_time': 0.2,
#     'dt': 1e-3,
#     'max_fr': 1.0,
#     'batch_size': 64,
#     'tau_exc': 0.02,
#     'tau_inh': 0.02,
#     'bu_rate': 1.0,
#     'td_rate': 1.0,
#     'symmetric_weight': True,
#     'n_epoch': 100,  # number of training epochs
#     'plot_interval': 10,  # frequency of intermediate reports
#     'recon_sample_n': 16,  # number of samples in the reports
#     'lr': 1e-2,  # learning rate
#     'w_decay': 1e-4  # regularizer
# }

sim_params = {
    'model_dir': model_dir,
    'dataset': 'gray_cifar-10',  # choose from ['mnist', 'fmnist', 'gray_cifar-10']
    'n_class': 3,
    'n_sample': 16,
    'sim_time': 0.5,
    'isi_time': 0.05,
    'dt': 1e-3,
    'max_fr': 1.0,
    'batch_size': 16,
    'tau_exc': 0.02,
    'tau_inh': 0.02,
    'bu_rate': 1.0,
    'td_rate': 1.0,
    'symmetric_weight': True,
    'n_epoch': 10,  # number of training epochs
    'plot_interval': 5,  # frequency of intermediate reports
    'recon_sample_n': 16,  # number of samples in the reports
    'lr': 1e-2,  # learning rate
    'w_decay': 1e-4  # regularizer
}

# generate training and test datasets and plot sample images
dataset, input_fig = generate_input(simParams=sim_params, class_choice=None, shuffle=True)
input_fig.show()

# run simulation
pc_net = run_simulation(simParams=sim_params, dataset=dataset, save=True)
