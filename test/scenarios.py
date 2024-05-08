import matplotlib.pyplot as plt
from models.parameters import *


def ReLu(x, theta=1):
    new_x = x - theta

    return np.maximum(new_x, 0)


def jorge(x, d, theta=1):
    return (x-theta) / (1 - np.exp(-d * (x-theta)))


def sigmoid(x):
    return 100 / (1 + np.exp(-(x - 10)))


def drdt(r, Isyn):
    return -r + jorge(Isyn, d=10.0)
    # return -r + ReLu(Isyn)
    # return -r + sigmoid(Isyn)


scenario = 4

# scenarios
sc2_v = 0
sc2_m = 0
sc4 = 0
sc5 = 0
sc6 = 0

# initialize firing rates
rPC = np.zeros(int(T / dt) + 1)
rPV = np.zeros(int(T / dt) + 1)
rSST = np.zeros(int(T / dt) + 1)
rVIP = np.zeros(int(T / dt) + 1)

for t in range(int(T / dt)):

    if scenario == 0:
        '''
        PC cells receive background, visual, and inputs (+Ibg, +Iv)
        '''
        w_pv_pc = 0
        w_sst_pc = 0

    elif scenario == 1:
        '''
        PC cells receive background, visual, and motor inputs and PV inhibition (+Ibg, +Iv, -PV)
        PV cells receive background and visual inputs (+Ibg, +Iv)
        '''
        w_sst_pc = 0

    elif scenario == 2:
        '''
        PC cells receive background, visual, and motor inputs and PV and SST inhibitions (+Ibg, +Iv, +Im, -PV, -SST)
        PV cells receive background and visual inputs (+Ibg, +Iv)
        SST cells receive background and motor inputs (+Ibg, +Im)
        '''
        sc2_m = 1

    elif scenario == 3:
        '''
        PC cells receive background, visual, and motor inputs and PV and SST inhibitions (+Ibg, +Iv, +Im, -PV, -SST)
        PV cells receive background and visual inputs (+Ibg, +Iv)
        SST cells receive background and visual inputs (+Ibg, +Iv)
        '''
        sc2_v = 1

    elif scenario == 4:
        '''
        PC cells receive background, visual, and motor inputs and PV and SST inhibitions (+Ibg, +Iv, +Im, -PV, -SST)
        PV cells receive background and visual inputs (+Ibg, +Iv)
        SST cells receive background and motor inputs and VIP inhibition (+Ibg, +Im, +Iv, -VIP)
        VIP cells receive background and motor inputs (+Ibg, +Im)
        '''
        sc2_m = 1
        sc2_v = 1
        sc4 = 1

    elif scenario == 5:
        '''
        PC cells receive background, visual, and motor inputs and PV and SST inhibitions (+Ibg, +Iv, +Im, -PV, -SST)
        PV cells receive background and visual inputs and SST inhibition (+Ibg, +Iv, -SST)
        SST cells receive background and motor inputs and VIP inhibition (+Ibg, +Iv, +Im, -VIP)
        VIP cells receive background and motor inputs (+Ibg, +Im)
        '''
        sc2_m = 1
        sc2_v = 1
        sc4 = 1
        sc5 = 1

    elif scenario == 6:
        '''
        PC cells receive background, visual, and motor inputs and PV and SST inhibitions 
        (+Ibg, +Iv, +Im, -PV, -SST, -VIP)
        PV cells receive background and visual inputs and SST inhibition 
        (+Ibg, +Iv, -SST)
        SST cells receive background and motor inputs and VIP inhibition 
        (+Ibg, +Iv, +Im, -VIP)
        VIP cells receive background and motor inputs 
        (+Ibg, +Im, -SST)
        '''
        sc2_m = 1
        sc2_v = 1
        sc4 = 1
        sc5 = 1
        sc6 = 1

    Isyn_pv = Ibg_pv + Iv[t] + w_pc_pv * rPC[t] - sc5 * w_sst_pv * rSST[t] + sc6 * w_vip_pv * rVIP[t]
    Isyn_vip = Ibg_vip + sc4 * Im[t] + w_pc_vip * rPC[t] + sc6 * w_sst_vip * rSST[t]
    Isyn_sst = Ibg_sst + sc2_m * Im[t] + sc2_v * Iv[t] - sc4 * w_vip_sst * rVIP[t] + w_pc_sst * rPC[t]
    Isyn_pc = Ibg_pc + Iv[t] - w_pv_pc * rPV[t] + scale_apical * (Im[t] - w_sst_pc * rSST[t])

    rPV[t + 1] = rPV[t] + (dt / tau_inh) * drdt(r=rPV[t], Isyn=Isyn_pv)
    rVIP[t + 1] = rVIP[t] + (dt / tau_inh) * drdt(r=rVIP[t], Isyn=Isyn_vip)
    rSST[t + 1] = rSST[t] + (dt / tau_inh) * drdt(r=rSST[t], Isyn=Isyn_sst)
    rPC[t + 1] = rPC[t] + (dt / tau_exc) * drdt(r=rPC[t], Isyn=Isyn_pc)


def plot_figure(scenario_n, rPC, rPV, rSST, rVIP):
    fig_title = [f'scenario #{i}' for i in range(7)]
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    fig_ax1 = fig.add_subplot(gs[0, :])
    fig_ax1.plot(np.arange(len(Iv)) / 10, Iv + np.max(Im) * 1.5, c='g', label='visual')
    fig_ax1.plot(np.arange(len(Im)) / 10, Im, c='orange', label='motor')
    fig_ax1.legend()
    # fig_ax1.set_ylim([-3000, 15000])
    fig_ax1.get_yaxis().set_visible(False)
    fig_ax1.set_title('external inputs')
    fig_ax2 = fig.add_subplot(gs[1, 0])
    fig_ax2.plot(np.arange(len(rPC[1000:])) / 10, rPC[1000:])
    fig_ax2.set_ylabel('rPC (Hz)')
    fig_ax3 = fig.add_subplot(gs[1, 1])
    fig_ax3.plot(np.arange(len(rPV[1000:])) / 10, rPV[1000:])
    fig_ax3.set_ylabel('rPV (Hz)')
    fig_ax4 = fig.add_subplot(gs[2, 0])
    fig_ax4.plot(np.arange(len(rSST[1000:])) / 10, rSST[1000:])
    fig_ax4.set_ylabel('rSST (Hz)')
    fig_ax5 = fig.add_subplot(gs[2, 1])
    fig_ax5.plot(np.arange(len(rVIP[1000:])) / 10, rVIP[1000:])
    fig_ax5.set_ylabel('rVIP (Hz)')

    for ax_i, ax in enumerate(fig.get_axes()):
        ax.set_xlabel('time (msec)')
        # ax.set_xlim([0, 1000])
        # if ax_i > 0:
        #     ax.set_ylim([0, 10])

    fig.suptitle(fig_title[scenario_n])

    return fig

fig = plot_figure(4, rPC, rPV, rSST, rVIP)
plt.show()