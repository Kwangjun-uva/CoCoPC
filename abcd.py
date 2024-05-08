import numpy as np
from scipy.stats import pearsonr
from scipy import sparse


def test_corr(inp_pattern, idx_pattern='random', pc=0.14, inp_size=250, n_neurons=1000):

    # Let W1 be a random matrix with connection probability, pc
    W1 = sparse.random(n_neurons, n_neurons, density=pc).toarray()

    # generate uniform input
    x_ones = np.ones(inp_size)
    # generate random input
    x_random = np.random.random(inp_size)

    # let x1 and x2 be L4 response to two patterns of input
    x1 = np.zeros(n_neurons)
    x2 = np.zeros(n_neurons)

    if idx_pattern == 'random':
        # select random patches of L4 to receive input
        x1_idx = np.arange(n_neurons)
        x2_idx = np.arange(n_neurons)

        np.random.shuffle(x1_idx)
        np.random.shuffle(x2_idx)

        x1_idc = np.where(x1_idx < inp_size)[0]
        x2_idc = np.where(x2_idx < inp_size)[0]
    elif idx_pattern == 'equal':
        # select first and second fourth of L4 indices to receive input
        x1_idc = np.arange(inp_size)
        x2_idc = np.arange(inp_size, 2 * inp_size)
    else:
        raise ValueError('Invalid index pattern')

    # feed the input
    if inp_pattern == 'uniform':
        x1[x1_idc] = x_ones
        x2[x2_idc] = x_ones
    elif inp_pattern == 'random':
        x1[x1_idc] = x_random
        x2[x2_idc] = x_random
    else:
        raise ValueError('Invalid input pattern')

    return pearsonr(W1 @ x1, W1 @ x2)[0]


#
corr_uniform = test_corr(inp_pattern='uniform')
corr_random = test_corr(inp_pattern='random')
#
corr_uniform_eq = test_corr(inp_pattern='uniform', idx_pattern='equal')
corr_random_eq = test_corr(inp_pattern='random', idx_pattern='equal')

print('Pearson correlation between two patterns')
print(f'same input intensity, random indices = {corr_uniform}')
print(f'random input intensity, random indices = {corr_random}')

print(f'same input intensity, 1/4 indices = {corr_uniform_eq}')
print(f'random input intensity, 1/4 indices = {corr_random_eq}')