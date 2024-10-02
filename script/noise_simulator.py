import numpy as np
import matplotlib.pyplot as plt
from script.pc_network import noise_generator, test_reconstruction


def plot_mse_ssim(noise_origin, noise_lvls, mse_dict, ssim_dict, colormap='PuOr'):

    plt.close('all')

    # pick color codes for MSE and SSIM
    mse_color = '#298c8c' #plt.colormaps.get_cmap(colormap)(0.1)
    ssim_color = '#f1a226' #plt.colormaps.get_cmap(colormap)(0.9)

    noise_fig, noise_ax1 = plt.subplots(figsize=(8, 4))

    # MSE
    noise_ax1.plot(noise_lvls, mse_dict['noise_recon'], color=mse_color, lw=2)
    # SSIM
    noise_ax2 = noise_ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    noise_ax2.plot(noise_lvls, ssim_dict['noise_recon'], color=ssim_color, lw=2)

    # for external noise, plot original vs noisy image comparison
    if noise_origin == 'ext':
        noise_ax1.plot(noise_lvls, mse_dict['noise'], color=mse_color, lw=2, ls='--')
        noise_ax2.plot(noise_lvls, ssim_dict['noise'], color=ssim_color, lw=2, ls='--')
    # for internal noise, there is no 'noisy' image
    else:
        pass

    # axis range
    noise_ax1.set_ylim([0, 0.04])
    noise_ax2.set_ylim([0, 1.0])
    # axis labels
    noise_ax1.set_xlabel(r'Noise level ($\sigma_{%s}$)' % noise_origin, fontsize=15)
    noise_ax1.set_ylabel('MSE', color=mse_color, fontsize=15)
    noise_ax2.set_ylabel('SSIM', color=ssim_color, fontsize=15)  # we already handled the x-label with ax1
    # axis ticks
    noise_ax1.set_xticks(noise_lvls)
    # axis tick colors
    noise_ax1.tick_params(axis='y', labelcolor=mse_color)
    noise_ax2.tick_params(axis='y', labelcolor=ssim_color)
    # axis tick size
    noise_ax1.tick_params(axis='both', which='major', labelsize=15)
    noise_ax2.tick_params(axis='both', which='major', labelsize=15)
    # spines
    for axis in ['top', 'bottom', 'left', 'right']:
        noise_ax1.spines[axis].set_linewidth(1.5)

    noise_fig.tight_layout()  # otherwise the right y-label is slightly clipped

    return noise_fig


def get_sample_imgs(img_dict, sigma_ext):
    # pick a sample image
    original = img_dict['original_img']
    img_shape = original.shape

    # generate noise
    n, _ = noise_generator(noise_type='normal', noise_lvl=sigma_ext, target_shape=img_shape)

    # sample images in dictionary
    sample_img_dict = {
        'original': original,
        'noise': n,
        'noisy': original + n
    }

    # plot: [original, noise, noisy]
    plot_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(3, 9))
    for ax_i, (img_key, img) in enumerate(sample_img_dict.items()):
        axs[ax_i].imshow(img, **plot_args)
        axs[ax_i].axis('off')
        axs[ax_i].set_title(img_key)
    fig.tight_layout()

    return fig, sample_img_dict


def get_recons(noise_lvls, img_dict):
    plot_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    recon_fig, recon_axs = plt.subplots(1, img_dict['sample_imgs'].shape[0])
    for i, img in enumerate(img_dict['sample_imgs']):
        recon_axs[i].imshow(img, **plot_args)
        recon_axs[i].axis('off')
        recon_axs[i].set_title(f'{noise_lvls[i]:.2f}')
    recon_fig.suptitle(r'Reconstructions with varying $\sigma_{%s}$' % "ext")
    recon_fig.tight_layout()

    return recon_fig


def get_pop_response(noise_origin, noise_val, sim_params, pretrained_weights, dataset, ran_idcs, noise_type='normal'):
    if noise_origin == 'ext':
        noise_input, _ = noise_generator(
            noise_type=noise_type, noise_lvl=noise_val, target_shape=dataset['test_x'].shape
        )
        noise_input = np.clip(noise_input + dataset['test_x'], 0.0, 1.0)
        noise_jit_type = 'constant'
        noise_jit_lvl = 0.0

    else:
        noise_input = dataset['test_x']
        noise_jit_type = noise_type
        noise_jit_lvl = noise_val

    _, popResponse_net, popResponse_rep = test_reconstruction(
        sim_param=sim_params, weights=pretrained_weights,
        layer=0, input_vector=noise_input, sample_idcs=ran_idcs,
        jit_type=noise_jit_type, jit_lvl=noise_jit_lvl
    )

    pop_activities = {
        'rep': popResponse_rep.mean(axis=1).mean(axis=0),
        'ppe': popResponse_net.errors['layer_0']['ppe_pyr'],
        'npe': popResponse_net.errors['layer_0']['npe_pyr']
    }

    fig_title = r'with $\sigma_{%s}$ = %.2f' % (noise_origin, noise_val)
    ax_labels = ['Representation', r'PE $+$', r'PE $-$']
    ax_colors = ['purple', 'r', 'b']

    popResponse_fig, popResponse_axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 6))
    for ax_i, (pop_key, pop_fr) in enumerate(pop_activities.items()):
        popResponse_axs[ax_i].plot(pop_fr, c=ax_colors[ax_i])
        popResponse_axs[ax_i].set_title(ax_labels[ax_i], c=ax_colors[ax_i])

        if ax_i == 1:
            popResponse_axs[ax_i].set_ylabel('Firing rate (a.u.)')
        elif ax_i == 2:
            popResponse_axs[ax_i].set_xlabel('Time (ms)')
        else:
            pass

    popResponse_fig.suptitle(f'Population activity ' + fig_title)
    popResponse_fig.tight_layout()

    return popResponse_fig, pop_activities
