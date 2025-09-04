from script.tools import create_dir
from script.data_loader import generate_input
from script.combinatorial_search import combinatorial_searcher
from script.pc_network import run_simulation


########################## combinatorial search ##########################

# initialize combinatorial search
combo_searcher = combinatorial_searcher()
# show input pattern
input_pattern_fig = combo_searcher.plot_input_pattern()
# run combinatorial search
combo_searcher.run()
# show an example response by pPE and nPE circuits
combo_searcher.plot_example_response()

##########################################################################


################################ data loading ############################

# create output directory
model_dir = 'results/'
create_dir(model_dir)

# select a dataset from ['mnist', 'fmnist', 'gray_cifar-10']
dataset_type = 'gray_cifar-10'
# number of image classes: all three datasets have max 10 classes
n_class = 10
# number of samples per class:
n_sample = 256
# total number of images
n_img = n_class * n_sample
# maximum firing rate
max_fr = 1.0
# input_x = (n_class * n_sample, n_pixel)
dataset, input_fig = generate_input(
    dataset_type=dataset_type,
    num_class=n_class, num_sample=n_sample,
    max_fr=max_fr, class_choice=None, shuffle=True
)

##########################################################################


################################ training ################################

sim_params = {
    'model_dir': model_dir,
    'dataset': dataset_type,
    'n_class': n_class,
    'n_sample': n_sample,
    'sim_time': 0.5, # time of stimulus presentation
    'isi_time': 0.01, # time of inter-stimulus interval
    'dt': 1e-3, # time resolution for simulation
    'max_fr': max_fr,
    'batch_size': 16, # batch size for batch training
    'tau_exc': 0.02, # membrane time constant for excitatory neurons
    'tau_inh': 0.02, # membrane time constant for inhibitory neurons
    'bu_rate': 1.0, # bottom-up inference rate
    'td_rate': 1.0, # top-down inference rate
    'n_epoch': 100, # number of training epochs
    'plot_interval': 10, # the frequency of intermediate reports showing reconstructions and errors (the lower the more frequent)
    'recon_sample_n': 16, # number of image samples to show for reconstruction (cannot exceed the batch size)
    'lr': 1e-2, # learning rate
    'w_decay': 1e-4 # weight decay
    }

pc_network = run_simulation(simParams=sim_params, dataset=dataset, save=True)

# ##########################################################################


