# time constant: cite?
# excitatory
tau_exc = 10e-3
# inhibitory
# tau_inh = 2e-3
tau_inh = 2e-3

# resolution
dt = 1e-3

# background input
bg_exc = 4
bg_inh = 1

params_dict = {
'exc': {'tau_r': 0.02, 'tau_d': 0.02, 'Ibg': 0.0},
'inh': {'tau_r': 0.02, 'tau_d': 0.02, 'Ibg': 0.0},
}