from script.tools import load_sim_data

# define model directory
model_dir = '../results/'

# load simulation results
sim_params, weights, dataset = load_sim_data(model_path=model_dir)