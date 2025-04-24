import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt

dt = 0.001


def jorge(x, d=10.0, theta=dt * 0.1):
    return (x - theta) / (1 - np.exp(-d * (x - theta)))


def sigmoid(x, sat_val=30, offset=5):
    return sat_val / (1 + np.exp(-x + offset))


def ReLu(x, theta=0.0):
    new_x = x - theta

    return np.maximum(new_x, 0)


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

    if plot_by == 'batch':
        errors = np.zeros((2, n_batch))
        for batch_i in range(n_batch):
            last_error_idx = (batch_i + 1) * time_per_batch
            errors[:, batch_i] = [ppe[last_error_idx - 1], npe[last_error_idx - 1]]

    elif plot_by == 'epoch':
        errors = np.zeros((2, n_epoch))
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

def plot_datset_dist():

    import tensorflow as tf

    (mnist_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    (fmnist_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    (cifar10_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()

    cifar10_images = tf.image.rgb_to_grayscale(cifar10_images).numpy()[:, :, :, 0]

    fig_titles = ['MNIST', 'fashion-MNIST', 'CIFAR-10']
    fig_plots = [mnist_images, fmnist_images, cifar10_images]
    hist_colors = plt.cm.inferno(np.linspace(0, 1, 20))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(5, 5))

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