import numpy as np
import matplotlib.pyplot as plt
from script.pc_network import test_oddball, pickle_load, generate_input


def oddball_input_generator(standard_img, deviant_img, n_seq, dev_loc):

    # create a sequence with a length (n_seq) and a deviant stimulation location (dev_loc)
    oddball_seq = np.repeat(standard_img[None, ...], n_seq, axis=0)
    oddball_seq[dev_loc - 1] = deviant_img

    # show the sequence
    xy_dim = np.sqrt(oddball_seq[0].shape[0]).astype(int)
    oddball_seq_fig, oddball_seq_axs = plt.subplots(nrows=1, ncols=n_seq)
    for i, ax in enumerate(oddball_seq_axs.flat):
        ax.imshow(oddball_seq[i].reshape(xy_dim, xy_dim), cmap='gray')
        ax.axis('off')

        if i == dev_loc - 1:
            ax.set_title('deviant')
        else:
            ax.set_title('standard')

    oddball_seq_fig.suptitle('Oddball stimulus sequence')
    oddball_seq_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return oddball_seq, oddball_seq_fig

def get_dev_idx(deviant_idx, idcs_list):

    if deviant_idx == None:
        return np.random.choice(idcs_list, 1, replace=False)[0]
    else:
        return deviant_idx

def oddball_simulation(sim_type, sim_params, dataset, dev_idx = None, len_seq=5, loc_deviant = 3, record='all', savefig=False):
    ref_img = dataset['test_x'][0]
    ref_class = dataset['test_y'][0]

    if sim_type == 'training_same':
        idx_same_class = np.argwhere(dataset['train_y'] == ref_class).flatten()
        idx_random = get_dev_idx(deviant_idx=dev_idx, idcs_list=idx_same_class)
        odd_img = dataset['train_x'][idx_random]
    elif sim_type == 'training_different':
        idx_diff_class = np.argwhere(dataset['train_y'] != ref_class).flatten()
        idx_random = get_dev_idx(deviant_idx=dev_idx, idcs_list=idx_diff_class)
        odd_img = dataset['train_x'][idx_random]
    elif sim_type == 'test_same':
        idx_same_class = np.argwhere(dataset['test_y'] == ref_class).flatten()
        idx_random = get_dev_idx(deviant_idx=dev_idx, idcs_list=idx_same_class)
        odd_img = dataset['test_x'][idx_random]
    elif sim_type == 'test_different':
        idx_diff_class = np.argwhere(dataset['test_y'] != ref_class).flatten()
        idx_random = get_dev_idx(deviant_idx=dev_idx, idcs_list=idx_diff_class)
        odd_img = dataset['test_x'][idx_random]
    elif sim_type == 'fmnist':
        # fmnist input
        fmnist_x, fmnist_y, _, _, _ = generate_input(
            input_type='mnist',
            num_class=10, num_sample=3, max_fr=sim_params['max_fr'], class_choice=None, shuffle=True
        )
        train_img_dim = np.sqrt(ref_img.shape[0]).astype(int)
        odd_img_dim = np.sqrt(fmnist_x[1].shape[0]).astype(int)
        odd_img = np.zeros((train_img_dim, train_img_dim))
        odd_img[2:30, 2:30] = fmnist_x[1].reshape(odd_img_dim, odd_img_dim)
        odd_img = odd_img.flatten()
        # odd_img2 = np.zeros((train_img_dim, train_img_dim))
        # odd_img2[2:30, 2:30] = fmnist_x[-1].reshape(odd_img_dim, odd_img_dim)
        # odd_img2 = odd_img2.flatten()
    elif sim_type == 'novel_set':
        fmnist_x, fmnist_y, _, _, _ = generate_input(
            input_type='fmnist',
            num_class=10, num_sample=3, max_fr=sim_params['max_fr'], class_choice=None, shuffle=True
        )
        odd_img = fmnist_x[1]
        idx_random = np.random.choice(np.arange(len(fmnist_y)), 1, replace=False)[0]
        ref_img = fmnist_x[idx_random]

    # oddball figure
    oddball_x, odd_fig = oddball_input_generator(
        standard_img=ref_img, deviant_img=odd_img, n_seq=len_seq, dev_loc=loc_deviant
    )
    # oddball_x, odd_fig = oddball_input(odd_img2, odd_img, n_repeat=nrepeat)

    # oddball params
    # sim_time = int(sim_params['sim_time'] / sim_params['dt'])
    # isi_time = int(sim_params['isi_time'] / sim_params['dt'])
    # oddball simulation
    oddball_net, re_fig = test_oddball(
        sim_params=sim_params, weights=weights, oddball_x=oddball_x,
        record=record
    )

    odd_fig._suptitle._text += f': {sim_type}'
    re_fig.suptitle(f'{sim_type}')

    if savefig:
        odd_fig.savefig(model_dir + f'oddball/{sim_type}_inputs.png', dpi=300)
        re_fig.savefig(model_dir + f'oddball/{sim_type}.png', dpi=300)

    return oddball_x, oddball_net, odd_fig, re_fig


project_dir = '/home/kwangjun/PycharmProjects/si_pc/'
# model_dir = project_dir + 'gamma_test/'
# model_dir = project_dir + 'one_layer_model/'
# model_dir = project_dir + 'three_layer/'
model_dir = project_dir + 'cifar10/trial03/'
# model_dir = project_dir + 'fmnist/trial01/'

dataset = pickle_load(model_dir, 'dataset.pkl')
sim_params = pickle_load(model_dir, 'sim_params.pkl')
weights = pickle_load(model_dir, 'weights.pkl')

plt.close('all')
sim_types = ['training_same', 'training_different', 'test_same', 'test_different', 'fmnist']
# for odd_type in sim_types:
#     oddball_net, oddball_seq, oddball_response = oddball_simulation(
#         sim_type=odd_type,
#         sim_params=sim_params,
#         record='all',
#         savefig=True
#     )
#     oddball_seq.show()
#     oddball_response.show()

sim_params['isi_time'] = 0.005
oddball_inputs, oddball_net, oddball_seq, oddball_response = oddball_simulation(
    sim_type='test_different',
    sim_params=sim_params,
    dataset=dataset,
    dev_idx=100,
    len_seq=5,
    loc_deviant=2,
    record='all',
    savefig=False
)
oddball_seq.show()
oddball_response.show()
