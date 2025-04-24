import numpy as np

# time constant: cite?
# excitatory
tau_exc = 2e-3#40e-3
# inhibitory
# tau_inh = 2e-3
tau_inh = 2e-3#20e-3

# scale factor for proportions
nPC = 1

# simulation time
dt = 1e-3
T = 8.0

# background input
bg_exc = 4e-3
bg_inh = 3e-3

Ibg_pc = bg_exc
Ibg_pv = bg_inh
Ibg_sst = bg_inh
Ibg_vip = bg_inh

# external input: both = 2-3 sec, visual = 6-7 sec, motor = 4-5 sec
# v_input = 2 between 2-3 and 6-7 sec and 0 elsewhere
v_input = 2
Iv = np.zeros(int(T / dt))
Iv[int(2 / dt):int(3 / dt)] = v_input
Iv[int(6 / dt):int(7 / dt)] = v_input

# m_input = 2 between 2-3 and 4-5 sec and 0 elsewhere
m_input = 2
Im = np.zeros(int(T / dt))
Im[int(2 / dt):int(3 / dt)] = m_input
Im[int(4 / dt):int(5 / dt)] = m_input

# synaptic weights
w_inh = 1
w_pv_pc = w_inh
w_sst_pc = w_inh
w_vip_sst = w_inh
w_sst_pv = w_inh * 0.1
w_sst_vip = w_inh
w_vip_pv = w_inh

w_exc = 1
w_pc_pv = w_exc
w_pc_sst = w_exc
w_pc_vip = w_exc

scale_apical = 1.0
