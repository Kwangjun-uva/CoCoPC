from parameters import dt
import numpy as np
import matplotlib.pyplot as plt
from practice_scalar import dr

T = 10.0  # sec
T_steps = int(T / dt)

# test pPE and nPE

# initialize pPE circuit
r_ppe_e = np.zeros(T_steps)
r_ppe_pv = np.zeros(T_steps)
r_ppe_sst = np.zeros(T_steps)
r_ppe_vip = np.zeros(T_steps)

# initialize nPE circuit
r_npe_e = np.zeros(T_steps)
r_npe_pv = np.zeros(T_steps)
r_npe_sst = np.zeros(T_steps)
r_npe_vip = np.zeros(T_steps)

td_input = np.zeros(T_steps)
bu_input = np.zeros(T_steps)

# generate three conditions: match, mismatch, and playback

# timing for each condition
match_start = int(T_steps * 0.1)
match_end = int(T_steps * 0.3)

mismatch_start = int(T_steps * 0.4)
mismatch_end = int(T_steps * 0.6)

playback_start = int(T_steps * 0.7)
playback_end = int(T_steps * 0.9)

# input for each condition
input_current = 1.0

# match (pred = input)
td_input[match_start:match_end] = input_current
bu_input[match_start:match_end] = input_current
# mismatch (pred < input)
td_input[mismatch_start:mismatch_end] = input_current - input_current / 2.0
bu_input[mismatch_start:mismatch_end] = input_current
# playback (pred > input)
td_input[playback_start:playback_end] = input_current
bu_input[playback_start:playback_end] = input_current - input_current / 2.0

w_self = 1.0

act_func = 'relu'
bg_noise = 0.0
input_noise = 0.0

for i in range(T_steps - 1):
    # input to ppe
    input_ppe_e = bu_input[i] + r_ppe_e[i] - r_ppe_pv[i] - r_ppe_sst[i]
    input_ppe_pv = td_input[i] + r_ppe_e[i] #- r_ppe_sst[i] #- 0.5 * r_ppe_pv[i]
    input_ppe_sst = td_input[i] + r_ppe_e[i] - r_ppe_vip[i]
    input_ppe_vip = td_input[i] + r_ppe_e[i]

    # input to npe
    input_npe_e = bu_input[i] + td_input[i] + r_npe_e[i] - r_npe_pv[i] - r_npe_sst[i]
    input_npe_pv = bu_input[i] + r_npe_e[i] #- w_self * r_npe_pv[i]
    input_npe_sst = bu_input[i] + td_input[i] + r_npe_e[i] - r_npe_vip[i]
    input_npe_vip = td_input[i] + r_npe_e[i]

    # ppe
    r_ppe_e[i + 1] = dr(r=r_ppe_e[i], inputs=input_ppe_e, ei_type='exc', func=act_func, input_noise=input_noise, bg_noise=bg_noise)
    r_ppe_pv[i + 1] = dr(r=r_ppe_pv[i], inputs=input_ppe_pv, func=act_func, input_noise=input_noise, bg_noise=bg_noise)
    r_ppe_sst[i + 1] = dr(r=r_ppe_sst[i], inputs=input_ppe_sst, func=act_func, input_noise=input_noise, bg_noise=bg_noise)
    r_ppe_vip[i + 1] = dr(r=r_ppe_vip[i], inputs=input_ppe_vip, func=act_func, input_noise=input_noise, bg_noise=bg_noise)

    # npe
    r_npe_e[i + 1] = dr(r=r_npe_e[i], inputs=input_npe_e, ei_type='exc', func=act_func, input_noise=input_noise, bg_noise=bg_noise)
    r_npe_pv[i + 1] = dr(r=r_npe_pv[i], inputs=input_npe_pv, func=act_func, input_noise=input_noise, bg_noise=bg_noise)
    r_npe_sst[i + 1] = dr(r=r_npe_sst[i], inputs=input_npe_sst, func=act_func, input_noise=input_noise, bg_noise=bg_noise)
    r_npe_vip[i + 1] = dr(r=r_npe_vip[i], inputs=input_npe_vip, func=act_func, input_noise=input_noise, bg_noise=bg_noise)

rs_ppe = [r_ppe_e, r_ppe_pv, r_ppe_sst, r_ppe_vip]
rs_npe = [r_npe_e, r_npe_pv, r_npe_sst, r_npe_vip]
rs_label = ['pyramidal', 'PV', 'SST', 'VIP']

# plot input and response to three conditions
fig = plt.figure()
grid = plt.GridSpec(nrows=3, ncols=2)
# input
plt.subplot(grid[0, :])
plt.plot(bu_input, c='green', label='BU')
plt.plot(td_input, c='orange', label='TD')
plt.xticks(
    [(match_start + match_end) / 2, (mismatch_start + mismatch_end) / 2, (playback_start + playback_end) / 2],
    ['bu = td', 'bu > td', 'bu < td']
)
plt.ylabel('firing rate (a.u.)')
plt.legend()
plt.title('input pattern')
# pyr response
plt.subplot(grid[1, 0])
plt.plot(r_ppe_e, c='r', label='pPE')
plt.plot(r_npe_e, c='b', label='nPE')
plt.legend(loc='upper left')
plt.ylabel('firing rate (a.u.)')
plt.title('pyramidal')
# pv response
plt.subplot(grid[1, 1])
plt.plot(r_ppe_pv, c='b')
plt.plot(r_npe_pv, c='r')
plt.title('PV')
# sst response
plt.subplot(grid[2, 0])
plt.plot(r_ppe_sst, c='b')
plt.plot(r_npe_sst, c='r')
plt.ylabel('firing rate (a.u.)')
plt.xlabel('time (ms)')
plt.title('SST')
# vip response
plt.subplot(grid[2, 1])
plt.plot(r_ppe_vip, c='b')
plt.plot(r_npe_vip, c='r')
plt.xlabel('time (ms)')
plt.title('VIP')

plt.tight_layout()
plt.show()


# figure 1A: input patterns
# plt.close('all')
# input_fig, input_axs = plt.subplots(1, 1, figsize=(12,3))
# input_axs.plot(bu_input, c='black', lw=2.0, label='BU')
# input_axs.plot(td_input, c='black', ls='--', lw=2.0, label='TD')
# input_axs.set_xticks(
#     [(match_start + match_end) / 2, (mismatch_start + mismatch_end) / 2, (playback_start + playback_end) / 2]
# )
# input_axs.set_xticklabels(['BU = TD', 'BU > TD', 'BU < TD'])
# input_axs.set_ylabel('firing rate (a.u.)')
# # input_axs.set_title('input pattern')
# # input_fig.legend(False)
# input_axs.spines['top'].set_visible(False)
# input_axs.spines['right'].set_visible(False)
# input_fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/figure1A_input_pattern.png', dpi=300, bbox_inches='tight')
# input_fig.show()


# # figure 1A: pPE pyramidal responses
# plt.close('all')
# ppe_pyr_fig, ppe_pyr_axs = plt.subplots(1, 1, figsize=(10,5))
# ppe_pyr_axs.plot(r_ppe_e, c='red', lw=2.0)
# ppe_pyr_axs.set_xticks(
#     [(match_start + match_end) / 2, (mismatch_start + mismatch_end) / 2, (playback_start + playback_end) / 2]
# )
# ppe_pyr_axs.set_xticklabels(['BU = TD', 'BU > TD', 'BU < TD'], fontsize=20.0)
# ppe_pyr_axs.set_yticks(
#     [0, 0.5, 1]
# )
# ppe_pyr_axs.set_yticklabels([0, 0.5, 1], fontsize=20.0)
# ppe_pyr_axs.set_xticklabels(['BU = TD', 'BU > TD', 'BU < TD'], fontsize=20.0)
# ppe_pyr_axs.set_ylabel('firing rate (a.u.)', fontsize=20.0)
# ppe_pyr_axs.spines['top'].set_visible(False)
# ppe_pyr_axs.spines['right'].set_visible(False)
# ppe_pyr_axs.spines['left'].set_linewidth(2.0)
# ppe_pyr_axs.spines['bottom'].set_linewidth(2.0)
# ppe_pyr_axs.set_ylim([0,1])
# ppe_pyr_fig.tight_layout()
# ppe_pyr_fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/figure1A_pPEpyr.png', dpi=300, bbox_inches='tight')
# ppe_pyr_fig.show()

# # figure 1A: nPE pyramidal responses
# plt.close('all')
# npe_pyr_fig, npe_pyr_axs = plt.subplots(1, 1, figsize=(10,5))
# npe_pyr_axs.plot(r_npe_e, c='blue', lw=2.0)
# npe_pyr_axs.set_xticks(
#     [(match_start + match_end) / 2, (mismatch_start + mismatch_end) / 2, (playback_start + playback_end) / 2]
# )
# npe_pyr_axs.set_xticklabels(['BU = TD', 'BU > TD', 'BU < TD'], fontsize=20.0)
# npe_pyr_axs.set_yticks(
#     [0, 0.5, 1]
# )
# npe_pyr_axs.set_yticklabels([0, 0.5, 1], fontsize=20.0)
# npe_pyr_axs.set_xticklabels(['BU = TD', 'BU > TD', 'BU < TD'], fontsize=20.0)
# npe_pyr_axs.set_ylabel('firing rate (a.u.)', fontsize=20.0)
# npe_pyr_axs.spines['top'].set_visible(False)
# npe_pyr_axs.spines['right'].set_visible(False)
# npe_pyr_axs.spines['left'].set_linewidth(2.0)
# npe_pyr_axs.spines['bottom'].set_linewidth(2.0)
# npe_pyr_axs.set_ylim([0,1])
# npe_pyr_fig.tight_layout()
# npe_pyr_fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/figure1A_nPEpyr.png', dpi=300, bbox_inches='tight')
# npe_pyr_fig.show()

# figure 1A: input pattern and pyramidal responses together
# plt.close('all')
# fig1a_fig, fig1a_axs = plt.subplots(3, 1, figsize=(6,3), sharex=True, sharey=True)
#
# fig1a_axs[0].plot(bu_input, c='black', lw=2.0)
# fig1a_axs[0].plot(td_input, c='black', ls='--', lw=2.0)
#
# fig1a_axs[1].plot(r_ppe_e, c='red', lw=2.0)
#
# fig1a_axs[2].plot(r_npe_e, c='blue', lw=2.0)
#
# for ax_i, ax in enumerate(fig1a_axs.flat):
#     ax.set_xticks(
#         [(match_start + match_end) / 2, (mismatch_start + mismatch_end) / 2, (playback_start + playback_end) / 2]
#     )
#     ax.set_yticks(
#         [0, 0.5, 1]
#     )
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.set_ylim([0,1])
#     if ax_i == len(fig1a_axs) - 1:
#         ax.set_xticklabels(['BU = TD', 'BU > TD', 'BU < TD'])
# fig1a_axs[1].set_ylabel('firing rate (a.u.)')
#
# fig1a_fig.tight_layout()
# fig1a_fig.savefig('/home/kwangjun/PycharmProjects/si_pc/cifar10/figure1A_input_response_together.png', dpi=300, bbox_inches='tight')
# fig1a_fig.show()