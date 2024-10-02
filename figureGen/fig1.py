from script.microcircuit_response import barplot_response, plot_inputPatterns, test_npe_ppe

bu_input, td_input, r_ppe_e, r_npe_e, activity_idcs = test_npe_ppe(
    T=10.0, act_func='relu', bg_noise=0.1, input_noise=0.0
)

# fig 1A: pPE and nPE responses with barplots
ppe_bar = barplot_response(activity=r_ppe_e, sign='positive', activity_idcs=activity_idcs)
ppe_bar.show()
npe_bar = barplot_response(activity=r_npe_e, sign='negative', activity_idcs=activity_idcs)
npe_bar.show()

# fig 1A: input patterns
input_plot = plot_inputPatterns(bu_input, td_input, activity_idcs)
input_plot.show()
