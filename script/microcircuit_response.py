from script.params import dt
from script.tools import dr
import numpy as np
import matplotlib.pyplot as plt


def plot_inputPatterns(bu_input, td_input, activity_idcs):
    # figure 1A: input patterns
    plt.close('all')
    input_fig, input_axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all', figsize=(10, 6))
    # bu input
    input_axs[1].plot(bu_input, c='black', lw=4.0, label='BU')
    # td input
    input_axs[0].plot(td_input, c='black', ls='--', lw=4.0, label='TD')

    for ax in input_axs:
        ax.set_xticks(
            [np.sum(activity_idc) / 2 for activity_idc in activity_idcs],
            #['BU = TD', 'BU > TD', 'BU < TD'],
            [],
            fontsize=15
        )
        ax.set_yticks(
            np.arange(0, 1.5, 0.5),
            #[0, 0.5, 1],
            [],
            fontsize=15
        )

        ax.xaxis.set_tick_params(width=4, length=8)
        ax.yaxis.set_tick_params(width=4, length=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)

    # input_fig.text(x=0.04, y=0.35, s='Firing rate (a.u.)', fontsize=15, rotation='vertical')
    # input_fig.tight_layout()

    return input_fig


def barplot_response(activity, activity_idcs, sign):

    colormap = 'gray'

    if sign == 'positive':
        colormap = 'Reds'
    elif sign == 'negative':
        colormap = 'Blues'

    plot_color = plt.colormaps.get_cmap(colormap)(0.75)
    # fig 1A: pPE bar
    yy = [np.mean(activity[slice(*activity_i)]) for activity_i in activity_idcs]
    ye = [np.std(activity[slice(*activity_i)]) for activity_i in activity_idcs]
    fig, axs = plt.subplots(1, 1)
    axs.bar(x=[0, 1, 2], height=yy, color=plot_color)
    axs.errorbar(x=[0, 1, 2], y=yy, yerr=ye, fmt=' ', c='k')
    # axs.set_ylim([0, 1])
    axs.set_xticks([0, 1, 2], ['BU=TD', 'BU > TD', 'BU < TD'], fontsize=15)
    axs.set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=15)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_linewidth(4)
    axs.spines['left'].set_linewidth(4)
    axs.xaxis.set_tick_params(width=4, length=8)
    axs.yaxis.set_tick_params(width=4, length=8)
    axs.set_ylabel('Firing rate (a.u.)', fontsize=15)
    fig.suptitle(f'Response by {sign} prediction error microcircuit', c=plot_color)
    # fig.savefig(save_dir + 'ppe_bar.png', dpi=300, bbox_inches='tight')

    return fig


def test_npe_ppe(
        T=10.0,
        act_func='relu',
        input_current=1.0,
        bg_noise=0.0, input_noise=0.0
):
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

    # # input for each condition
    # input_current = 1.0

    # match (pred = input)
    td_input[match_start:match_end] = input_current
    bu_input[match_start:match_end] = input_current
    # mismatch (pred < input)
    td_input[mismatch_start:mismatch_end] = input_current - input_current / 2.0
    bu_input[mismatch_start:mismatch_end] = input_current
    # playback (pred > input)
    td_input[playback_start:playback_end] = input_current
    bu_input[playback_start:playback_end] = input_current - input_current / 2.0

    for i in range(T_steps - 1):
        # input to ppe
        input_ppe_e = bu_input[i] + r_ppe_e[i] - r_ppe_pv[i] - r_ppe_sst[i]
        input_ppe_pv = td_input[i] + r_ppe_e[i]  # - r_ppe_sst[i] #- 0.5 * r_ppe_pv[i]
        input_ppe_sst = td_input[i] + r_ppe_e[i] - r_ppe_vip[i]
        input_ppe_vip = td_input[i] + r_ppe_e[i]

        # input to npe
        input_npe_e = bu_input[i] + td_input[i] + r_npe_e[i] - r_npe_pv[i] - r_npe_sst[i]
        input_npe_pv = bu_input[i] + r_npe_e[i]  # - w_self * r_npe_pv[i]
        input_npe_sst = bu_input[i] + td_input[i] + r_npe_e[i] - r_npe_vip[i]
        input_npe_vip = td_input[i] + r_npe_e[i]

        # ppe
        r_ppe_e[i + 1] = dr(r=r_ppe_e[i], inputs=input_ppe_e, ei_type='exc', func=act_func, input_noise=input_noise,
                            bg_noise=bg_noise)
        r_ppe_pv[i + 1] = dr(r=r_ppe_pv[i], inputs=input_ppe_pv, func=act_func, input_noise=input_noise,
                             bg_noise=bg_noise)
        r_ppe_sst[i + 1] = dr(r=r_ppe_sst[i], inputs=input_ppe_sst, func=act_func, input_noise=input_noise,
                              bg_noise=bg_noise)
        r_ppe_vip[i + 1] = dr(r=r_ppe_vip[i], inputs=input_ppe_vip, func=act_func, input_noise=input_noise,
                              bg_noise=bg_noise)

        # npe
        r_npe_e[i + 1] = dr(r=r_npe_e[i], inputs=input_npe_e, ei_type='exc', func=act_func, input_noise=input_noise,
                            bg_noise=bg_noise)
        r_npe_pv[i + 1] = dr(r=r_npe_pv[i], inputs=input_npe_pv, func=act_func, input_noise=input_noise,
                             bg_noise=bg_noise)
        r_npe_sst[i + 1] = dr(r=r_npe_sst[i], inputs=input_npe_sst, func=act_func, input_noise=input_noise,
                              bg_noise=bg_noise)
        r_npe_vip[i + 1] = dr(r=r_npe_vip[i], inputs=input_npe_vip, func=act_func, input_noise=input_noise,
                              bg_noise=bg_noise)

    # rs_ppe = [r_ppe_e, r_ppe_pv, r_ppe_sst, r_ppe_vip]
    # rs_npe = [r_npe_e, r_npe_pv, r_npe_sst, r_npe_vip]
    # rs_label = ['pyramidal', 'PV', 'SST', 'VIP']

    # plot input and response to three conditions
    plt.close('all')
    fig = plt.figure()
    grid = plt.GridSpec(nrows=3, ncols=2)
    # input
    plt.subplot(grid[0, :])
    plt.plot(bu_input, c='k', lw=2, label='BU')
    plt.plot(td_input, c='k', ls='--', lw=2, label='TD')
    plt.xticks(
        [(match_start + match_end) / 2, (mismatch_start + mismatch_end) / 2, (playback_start + playback_end) / 2],
        ['bu = td', 'bu > td', 'bu < td']
    )
    plt.ylabel('firing rate (a.u.)')
    plt.legend()
    plt.title('Input pattern')
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

    fig.suptitle(
        r'$I_{input}$ = %.2f, $\epsilon_{bg}$ = %.2f, $\epsilon_{input}$ = %.2f' % (
            input_current, bg_noise, input_noise)
    )
    fig.tight_layout()
    fig.show()

    activity_idcs = [(match_start, match_end), (mismatch_start, mismatch_end), (playback_start, playback_end)]

    return bu_input, td_input, r_ppe_e, r_npe_e, activity_idcs
