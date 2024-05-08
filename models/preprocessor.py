import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def data_prep(img_set, n_class, n_sample):
    # select an image set
    if img_set == 'mnist':
        imgset = tf.keras.datasets.mnist
    elif img_set == 'fmnist':
        imgset = tf.keras.datasets.fashion_mnist
    elif img_set == 'cifar10':
        imgset = tf.keras.datasets.cifar10
    else:
        raise ValueError('Not a valid image set.')

    # load dataset and labelset
    (x_train, y_train), (x_test, y_test) = imgset.load_data()

    # pick samples from each class: train and test sets have different sizes
    train_idcs = pick_samples(y_train, n_class, n_sample)
    test_idcs = pick_samples(y_test, n_class, n_sample)

    # if cifar10, convert to grayscale
    if img_set == 'cifar10':
        x_train = tf.squeeze(tf.image.rgb_to_grayscale(x_train), axis=-1)
        x_test = tf.squeeze(tf.image.rgb_to_grayscale(x_test), axis=-1)

    _, y, z = np.shape(x_train)
    total_sample = n_class * n_sample

    return x_train[train_idcs].reshape(total_sample, y*z), \
           y_train[train_idcs], x_test[test_idcs].reshape(total_sample, y*z), y_test[test_idcs]

def pick_samples(label, n_class, n_sample):

    idcs = []
    # for each class, pick n_sample samples
    for cl in range(n_class):
        cl_train = np.where(label == cl)[0]
        idcs.append(np.random.choice(cl_train, n_sample, replace=False))

    return np.ravel(idcs)

def normalize_data(data, norm_ax=1, fr_max=60):

    # normalize to unit variance
    new_data = normalize(data, axis=norm_ax, norm='l2')
    # max fr
    fr_adjust = fr_max / np.max(new_data)
    # normalize per img (= over neurons) and scale
    return new_data * fr_adjust

# def poisson_input_generator(fr, simDur, dt):
#     nbins = np.floor(simDur / dt)
#     spikemat = np.random.random(nbins) < fr * dt

if __name__ == "__main__":

    nClass = 10
    nSample = 64
    # load data and label
    train_data, train_label, test_data, test_label = data_prep('mnist', nClass, nSample)
    # normalize and scale
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    # specify simulation time and resolution
    sim_time = 1000e-3
    dt = 1e-3
    tau_rise = 5e-3
    tau_decay = 50e-3
    glu_reset = -1

    # record spikes
    spike = np.zeros((train_data.shape[1], int(sim_time/dt)))
    spike_trace = np.zeros((train_data.shape[1], int(sim_time/dt)))
    glu_trace = np.zeros((train_data.shape[1], int(sim_time/dt)))

    for t in range(int(sim_time / dt) - 1):
        spike[:, t] = np.random.random(train_data.shape[1]) < train_data[0] * dt

        dglu_dt = glu_reset * spike[:, t] - (glu_trace[:, t] / tau_rise) * (1 - spike[:, t])
        dca_dt = -glu_trace[:, t] / tau_rise - spike_trace[:, t] / tau_decay
        glu_trace[:, t+1] = glu_trace[:, t] + dglu_dt * dt
        spike_trace[:, t+1] = spike_trace[:, t] + dca_dt * dt

    # check
    non_zero_idcs = np.where(spike.sum(axis=1) != 0)[0]
    non_zero_idx = np.random.choice(non_zero_idcs, 1)[0]

    fig, axs = plt.subplots(nrows=3, sharex=True)
    axs[0].plot(spike[non_zero_idx])
    axs[0].set_title('spike')
    axs[0].set_yticks([])
    axs[1].plot(glu_trace[non_zero_idx])
    axs[1].set_title('presynaptic glutamate deposit')
    axs[1].set_yticks([])
    axs[2].plot(spike_trace[non_zero_idx])
    axs[2].set_title('postsynpatic calcium influx')
    axs[2].set_yticks([])
    fig.tight_layout()
    fig.show()