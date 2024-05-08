from test import combo
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse

T = 4.0  # s
dt = 1e-3

# external input
v_input = 2
Iv = np.zeros(int(T / dt))
Iv[int(1 / dt):int(1.5 / dt)] = v_input
Iv[int(3 / dt):int(3.5 / dt)] = v_input

m_input = 2
Im = np.zeros(int(T / dt))
Im[int(1 / dt):int(1.5 / dt)] = m_input
Im[int(2 / dt):int(2.5 / dt)] = m_input

windows = [(int(0.5 / dt), int(1.5 / dt)),
           (int(2 / dt), int(2.5 / dt)),
           (int(3 / dt), int(3.5 / dt))
           ]
thresholds = [(0.0, 0.1), 0.5]

frs, npe_idcs, npe_mat, inp_mat, Iv, Im, Ibg, tau = combo.main(windows=windows,
                                                               thresholds=thresholds,
                                                               Iv=Iv,
                                                               Im=Im)

# plot nPE combinations
nPE_fig = combo.plot_PE_combo(frs=frs, npe_idcs=npe_idcs[0], inp_mat=inp_mat, ncol=6, nrow=5)
plt.show()
# plot pPE combinations
pPE_fig = combo.plot_PE_combo(frs=frs, npe_idcs=npe_idcs[1], inp_mat=inp_mat, ncol=6, nrow=5)
plt.show()

steps = 10
min_converge_time = 1.5
T = steps * min_converge_time
dt = 1e-4
Ibgs = np.array([4e-3, 3e-3, 3e-3, 3e-3])
taus = np.array([60e-3, 2e-3, 2e-3, 2e-3])
# external input
m_input = 2
Im = np.zeros(int(T / dt))
for i in range(steps):
    Im[i * int(min_converge_time / dt):(i + 1) * int(min_converge_time / dt)] = i
Iv = np.ones(int(T / dt)) * m_input

nfrs = combo.simulation(T, dt,
                        Ibg=Ibgs,
                        Iv=Iv, Im=Im,
                        inp_mat=inp_mat[npe_idcs[0]], w_mat=npe_mat,
                        tau=taus)

fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(3 * 5, 3 * 6))
axs = axs.flatten()
for i, npe_idx in enumerate(inp_mat[npe_idcs[0]]):
    mean_nfrs = [np.mean(nfrs[i, 0, j * int(min_converge_time / dt):(j + 1) * int(min_converge_time / dt)]) for j in
                 range(10)]
    axs[i * 2].plot(mean_nfrs)
    axs[i * 2].axvline(x=2, c='r', ls='--')
    # axs[i * 2].plot(Im, nfrs[i, 0, 1:])
    axs[i * 2].set_ylim([0, 4])
    axs[i * 2].set_title(f'syn# = {npe_idcs[0][i]} \nmse ={np.round(mse(mean_nfrs, np.arange(10)), 3)}')
    axs[i * 2 + 1].imshow(npe_idx, cmap='gray')
    axs[i * 2 + 1].set_xticks(np.arange(2))
    axs[i * 2 + 1].set_yticks(np.arange(4))
    axs[i * 2 + 1].set_xticklabels(['V', 'M'], fontsize=10)
    axs[i * 2 + 1].set_yticklabels(['PC', 'PV', 'SST', 'VIP'], fontsize=10)
    axs[i * 2 + 1].xaxis.tick_top()
fig.tight_layout()
plt.show()

# external input
v_input = 2
Iv = np.zeros(int(T / dt))
for i in range(steps):
    Iv[i * int(min_converge_time / dt):(i + 1) * int(min_converge_time / dt)] = i
Im = np.ones(int(T / dt)) * v_input

pfrs = combo.simulation(T, dt,
                        Ibg=Ibgs,
                        Iv=Iv, Im=Im,
                        inp_mat=inp_mat[npe_idcs[1]], w_mat=npe_mat,
                        tau=taus)

fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(3 * 5, 3 * 6))
axs = axs.flatten()
for i, npe_idx in enumerate(inp_mat[npe_idcs[1]]):
    mean_pfrs = [np.mean(pfrs[i, 0, j * int(min_converge_time / dt):(j + 1) * int(min_converge_time / dt)])
                 for j in range(10)]
    axs[i * 2].plot(mean_pfrs)
    axs[i * 2].axvline(x=2, c='r', ls='--')
    axs[i * 2].set_ylim([0, 4])
    axs[i * 2].set_title(f'syn# = {npe_idcs[1][i]} \nmse ={np.round(mse(mean_pfrs, np.arange(10)), 3)}')
    axs[i * 2 + 1].imshow(npe_idx, cmap='gray')
    axs[i * 2 + 1].set_xticks(np.arange(2))
    axs[i * 2 + 1].set_yticks(np.arange(4))
    axs[i * 2 + 1].set_xticklabels(['V', 'M'], fontsize=10)
    axs[i * 2 + 1].set_yticklabels(['PC', 'PV', 'SST', 'VIP'], fontsize=10)
    axs[i * 2 + 1].xaxis.tick_top()
fig.tight_layout()
fig.show()
