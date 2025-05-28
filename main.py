from test.combo import combinatorial_search, plot_mean_firing_rates

# initialize combinatorial search
searcher = combinatorial_search()

# show input pattern
input_pattern_fig = searcher.plot_input_pattern()
input_pattern_fig.show()

# run combinatorial search
combi_dict = searcher.run()

# show an example response by pPE and nPE circuits
# [0,4] for the pyramidal cell in pPE and nPE circuit
combo_idx = list(combi_dict.keys())
ppe_stats, npe_stats, pe_response_fig = plot_mean_firing_rates(
    windows=searcher.windows, frs=combi_dict[combo_idx[0]][[0, 4]]
)
pe_response_fig.show()