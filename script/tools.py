import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from tqdm import trange

# from script.params import params_dict


dt = 0.001

def f(x, func_type='ReLu'):

    if func_type == 'jorge':

        d = 10.0
        theta = dt * 0.1

        return (x - theta) / (1 - np.exp(-d * (x - theta)))


    elif func_type == 'sigmoid':

        sat_val = 30
        offset = 5

        return sat_val / (1 + np.exp(-x + offset))


    elif func_type == 'ReLu':

        theta = 0
        new_x = x - theta

        return np.maximum(new_x, 0)

    else:
        raise ValueError('not supported')


def dr(r, inputs, func='relu', ei_type='inh', tau=0.02, input_noise=0.0, bg_noise=0.0):
    noisy_input = inputs + np.random.normal(loc=0, scale=input_noise, size=np.array(inputs).shape)

    fx = f(x=noisy_input, func_type=func)

    bg_r = np.random.normal(loc=0, scale=bg_noise, size=np.array(r).shape)
    # decay = -r / tau
    # rise = (fx + bg_r) / tau
    #
    # return r + dt * (decay + rise)

    return r + dt * (-r + fx + bg_r)


def sample_imgs(img_labels, n_class, n_sample, class_choice=None):
    if class_choice:
        y_choice = class_choice
    else:
        y_choice = np.random.choice(np.unique(img_labels), n_class, replace=False)

    img_idcs = []
    for cl in y_choice:
        cl_imgs = np.where(img_labels == cl)[0]
        img_idcs.append(np.random.choice(cl_imgs, n_sample, replace=False))
    img_idcs = np.ravel(img_idcs)

    return img_idcs


def pickle_save(save_path, file_name, var):
    with open(save_path + file_name, 'wb') as file:
        pickle.dump(var, file, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(save_path, file_name):
    with open(save_path + file_name, 'rb') as file:
        return pickle.load(file)


def generate_reconstruction_pad(img_mat, nx=16):
    # img_mat = (n_pixel, n_sample)
    x_pixel = np.sqrt(img_mat.shape[0]).astype(int)
    # nx = number of images per col, ny = number of images per row
    ny = img_mat.shape[1] // nx
    # pad width
    dim_x = nx * x_pixel  # + (nx - 1)
    # pad height
    dim_y = ny * x_pixel  # + (ny - 1)
    result_pad = np.zeros((dim_x, dim_y))

    for xi in range(nx):
        for yi in range(ny):
            x_idx = slice(xi * x_pixel, (xi + 1) * x_pixel)
            y_idx = slice(yi * x_pixel, (yi + 1) * x_pixel)
            result_pad[x_idx, y_idx] = img_mat[:, xi * ny + yi].reshape(
                x_pixel, x_pixel
            )

    x_ticks = [x_pixel - 1, x_pixel * ny - 1, x_pixel]
    y_ticks = [x_pixel - 1, x_pixel * nx - 1, x_pixel]

    return x_ticks, y_ticks, result_pad


def mean_errors(simulation_params, trial_errors, plot_by='epoch'):
    ppe = trial_errors['layer_0']['ppe_pyr']
    npe = trial_errors['layer_0']['npe_pyr']

    len_errors = len(ppe)

    sim_timesteps = int(simulation_params['sim_time'] / simulation_params['dt'])
    isi_timesteps = int(simulation_params['isi_time'] / simulation_params['dt'])
    time_per_batch = sim_timesteps + isi_timesteps

    n_epoch = simulation_params['n_epoch']

    n_batch = int(len_errors / time_per_batch)
    n_batch_per_epoch = int(n_batch / n_epoch)
    time_per_epoch = n_batch_per_epoch * time_per_batch

    errors = np.zeros((2, n_epoch))
    if plot_by == 'batch':
        for batch_i in range(n_batch):
            last_error_idx = (batch_i + 1) * time_per_batch
            errors[:, batch_i] = [ppe[last_error_idx - 1], npe[last_error_idx - 1]]

    elif plot_by == 'epoch':
        for epoch_i in range(n_epoch):
            last_error_idx = (epoch_i + 1) * time_per_epoch
            errors[:, epoch_i] = [ppe[last_error_idx - 1], npe[last_error_idx - 1]]

    line_colors = ['r', 'b']
    line_labels = ['pPe', 'nPE']

    fig, axs = plt.subplots(1, 1)
    for pe_i in range(len(line_colors)):
        axs.plot(errors[pe_i], c=line_colors[pe_i], label=line_labels[pe_i])
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.set_xlabel('training iteration')
        axs.set_ylabel('Mean prediction errors')
        axs.legend(fancybox=True, shadow=True, labelcolor='linecolor')

    fig.suptitle('Prediction errors across training iterations')

    return errors, fig


def plot_dataset_dist():
    import tensorflow as tf

    (mnist_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    (fmnist_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    (cifar10_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()

    cifar10_images = tf.image.rgb_to_grayscale(cifar10_images).numpy()[:, :, :, 0]

    fig_titles = ['MNIST', 'fashion-MNIST', 'CIFAR-10']
    fig_plots = [mnist_images, fmnist_images, cifar10_images]
    hist_colors = plt.colormaps.get_cmap('inferno')(np.linspace(0, 1, 20))

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex='all', figsize=(5, 5))

    for i, ax in enumerate(axs):
        ax.hist(fig_plots[i].flatten(), density=True, color=hist_colors[i * 5], alpha=0.75)
        ax.set_title(fig_titles[i], c=hist_colors[i * 5])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axs[1].set_ylabel('Probability density')
    axs[2].set_xlabel('Pixel intensity')

    fig.suptitle('Pixel intensity distributions')
    fig.tight_layout()

    return fig


def load_sim_data(model_path):
    """
    :param model_path: directory where trained params (sim_params, weights, dataset) are located
    :return: simulation parameters, pretrained weights, and dataset used for training and test
    """

    sim_params = pickle_load(save_path=model_path, file_name='sim_params.pkl')
    pretrained_weights = pickle_load(save_path=model_path, file_name='weights.pkl')
    dataset = pickle_load(save_path=model_path, file_name='dataset.pkl')

    return sim_params, pretrained_weights, dataset


def create_dir(dir_path):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {dir_path}: {e}")
        return

def save_model(save_path, data_to_save):
    """
    Saves a dictionary of key-value pairs to the specified directory.
    The keys become the filenames, and the values are the data.
    """
    # Create output directory if it doesn't exist
    create_dir(save_path)

    for filename, data in data_to_save.items():
        file_path = os.path.join(save_path, f"{filename}.pkl")
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Successfully saved {file_path}")
        except (IOError, pickle.PicklingError) as e:
            print(f"Error saving {file_path}: {e}")

### noise ###

def noise_generator(noise_type, noise_lvl, target_shape):

    noise_func = {
        'uniform': [np.random.uniform, [-noise_lvl, noise_lvl],
                    r'external noise ~ $\mathcal{U}$' + f' (0, {noise_lvl:.3f})'],
        'normal': [np.random.normal, [0, noise_lvl],
                   r'external noise ~ $\mathcal{N}$' + f' (0, {noise_lvl:.3f})'],
        'constant': [return_self, [noise_lvl],
                     f'external noise = (0, {noise_lvl:.2f})']
    }

    return noise_func[noise_type][0](*noise_func[noise_type][1], target_shape), noise_func[noise_type][2]


def return_self(x, size):
    return np.ones(size) * x


### oddball ####

def create_vlines(target_axes, total_sim_time, trial_sim_time, interval_time):
    for yi in np.arange(0, total_sim_time, trial_sim_time + interval_time):
        target_axes.axvline(x=yi, ls='--', c='black')
        target_axes.axvline(x=yi + interval_time, ls='--', c='black')


### plotting ###

def remove_top_right_spines(target_axes, target_spines=None):
    if target_spines is None:
        target_spines = ['top', 'right']
    for spine in target_spines:
        target_axes.spines[spine].set_visible(False)

### training ###

def plot_training_errors(sim_params, errs):

    # fig 2B - errors
    tsim = int(sim_params['sim_time'] / sim_params['dt'])
    tisi = int(sim_params['isi_time'] / sim_params['dt'])
    plt.close('all')
    aa = np.zeros(sim_params['n_epoch'])
    bb = np.zeros(sim_params['n_epoch'])
    len_epoch = int(tsim + tisi) * int(sim_params['n_class'] * sim_params['n_sample'] / sim_params['batch_size'])
    for i in range(sim_params['n_epoch']):
        curr_epoch_ppe = errs['layer_0']['ppe_pyr'][i * len_epoch: (i + 1) * len_epoch]
        curr_epoch_npe = errs['layer_0']['npe_pyr'][i * len_epoch: (i + 1) * len_epoch]
        aa[i] = curr_epoch_ppe[-1]
        bb[i] = curr_epoch_npe[-1]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(aa, c='#CA181D', lw=3.0, label='PE+')
    ax.plot(bb, c='#2070B4', lw=3.0, label='PE-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_xlabel('training iteration', fontsize=20)
    ax.set_ylabel('firing rate (a.u.)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.legend(labelcolor='linecolor', fontsize=20, edgecolor='white')  # , fancybox=True, shadow=True)
    fig.tight_layout()
    fig.show()

    return fig

