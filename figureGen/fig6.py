from figureGen.training_data import sim_params, weights, dataset
from script.opto import opto_simulator

# initialize network
opto_net = opto_simulator(sim_params=sim_params, pretrained_weights=weights, dataset=dataset, n_class=10, n_sample=3)

# Figure 6: recon_figs and response_figs are dictionaries
recon_figs, response_figs = opto_net.run_experiment(silencing_targets=['none', 'pv', 'sst', 'vip'], recon_fig_nrow=5)
