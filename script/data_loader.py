import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from script.tools import sample_imgs, create_dir


def get_training_test_set(ds_key, num_class, num_sample, class_choice=None, max_fr=30, shuffle=True):

    training_set = {
        'mnist': tf.keras.datasets.mnist,
        'fmnist': tf.keras.datasets.fashion_mnist,
        'gray_cifar-10': tf.keras.datasets.cifar10
    }

    if ds_key not in list(training_set.keys()):
        raise ValueError('Please chooose from: mnist, fmnist, gray_cifar-10.')

    (x_train, y_train), (x_test, y_test) = training_set[ds_key].load_data()

    # convert cifar-10 to grayscale
    if ds_key == 'gray_cifar-10':
        x_train = tf.image.rgb_to_grayscale(x_train).numpy()[:, :, :, 0]
        x_test = tf.image.rgb_to_grayscale(x_test).numpy()[:, :, :, 0]
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]

    train_idx = sample_imgs(y_train, num_class, num_sample, class_choice=class_choice)
    test_idx = sample_imgs(y_test, num_class, num_sample, class_choice=class_choice)

    # shuffle
    if shuffle:
        np.random.shuffle(train_idx)

    # scale to [0,1]
    data_train = x_train[train_idx] / 255.0 * max_fr
    data_test = x_test[test_idx] / 255.0 * max_fr

    n_imgs, dim_x, dim_y = data_train.shape

    # reshape
    data_train = data_train.reshape(n_imgs, dim_x * dim_y)
    label_train = y_train[train_idx]
    data_test = data_test.reshape(n_imgs, dim_x * dim_y)
    label_test = y_test[test_idx]

    return data_train, label_train, data_test, label_test


def generate_input(dataset_type, num_class, num_sample, max_fr, class_choice=None, shuffle=False):

    x_input, y_input, x_test, y_test = get_training_test_set(
        ds_key=dataset_type,
        num_class=num_class, num_sample=num_sample,
        max_fr=max_fr,
        class_choice=class_choice, shuffle=shuffle
    )

    img_dim = np.sqrt(x_input.shape[1]).astype(int)

    # show training image samples
    if num_class * num_sample > 1:
        if num_class == 1:
            fig = plt.figure()
            plt.imshow(x_input[0].reshape(img_dim, img_dim), cmap='gray')
            plt.title(dataset_type)
        else:
            # find one sample per class
            sample_idcs = [np.argwhere(y_input == y_unique)[0][0] for y_unique in np.unique(y_input)]

            fig, axs = plt.subplots(1, num_class, figsize=(2 * num_class, 2))
            for ax_i, ax in enumerate(axs):
                img_idx = sample_idcs[ax_i]
                img = ax.imshow(x_input[img_idx].reshape(img_dim, img_dim), cmap='gray')
                ax.axis("off")
                fig.colorbar(img, ax=ax, shrink=0.6)
                ax.set_title(f'{dataset_type}: class {ax_i + 1}')
    else:
        fig, axs = plt.subplots(1, 1)
        img = axs.imshow(x_input[0].reshape(img_dim, img_dim), cmap='gray')
        axs.axis("off")
        fig.colorbar(img, shrink=0.6)
        axs.set_title(dataset_type)

    fig.tight_layout()
    fig.show()

    # construct a dataset dictionary
    dataset = {
        'train_x': x_input,
        'train_y': y_input,
        'test_x': x_test,
        'test_y': y_test
    }

    return dataset, fig

if __name__ == '__main__':

    # create output directory
    model_dir = './testing/'
    create_dir(model_dir)

    # data parameters
    n_class = 10
    n_sample = 32
    n_img = n_class * n_sample
    # input_x = (n_class * n_sample, n_pixel)
    dataset, input_fig = generate_input(
        dataset_type='mnist', num_class=n_class, num_sample=n_sample, max_fr=1.0, class_choice=None, shuffle=True
    )


