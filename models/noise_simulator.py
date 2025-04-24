import pickle5 as pickle
import numpy as np
from script.pc_network import test_noise

model_dir = '/home/kwangjun/PycharmProjects/si_pc/cifar10/trial03/'
# model_dir = '/home/kwangjun/PycharmProjects/si_pc/fmnist/trial01/'
# model_dir = '/home/kwangjun/PycharmProjects/si_pc/mnist/trial01/'

with open(model_dir + 'sim_params.pkl', 'rb') as f:
    sim_params = pickle.load(f)
with open(model_dir + 'weights.pkl', 'rb') as f:
    pretrained_weights = pickle.load(f)
with open(model_dir + 'dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

ran_idcs = np.random.choice(len(dataset['test_x']), 20, replace=False)
noise_recon_fig, noise_net, metric_fig = test_noise(
    save_dir=model_dir + 'noise/', sim_param=sim_params, pretrained_weights=pretrained_weights,
    noise_levels=np.linspace(0,0.2,10), noise_type=('int', 'normal'),
    test_images=dataset['test_x'], test_sample_idcs=ran_idcs,
    savefig=True
)
metric_fig.show()

# # to simulate population responses with noise
# import matplotlib.pyplot as plt
# from script.pc_network import noise_generator, test_reconstruction
#
# noise_source = 'ext'
# noise_type = 'normal'

# noise_lvls = np.arange(0, 0.31, 0.06)
# for noise_lvl in noise_lvls:
#
#     if noise_source == 'ext':
#         noise_input, _ = noise_generator(
#             noise_type=noise_type, noise_lvl=noise_lvl, target_shape=dataset['test_x'].shape
#         )
#         noise_input = np.clip(noise_input + dataset['test_x'], 0.0, 1.0)
#         noise_jit_type = 'constant'
#         noise_jit_lvl = 0.0
#
#     else:
#         noise_input = dataset['test_x']
#         noise_jit_type = noise_type
#         noise_jit_lvl = noise_lvl
#
#     fig_title = f'with {noise_source}ernal noise = {noise_lvl}'
#
#     recon_fig, test_net, rep = test_reconstruction(
#         sim_param=sim_params, weights=pretrained_weights,
#         layer=0, input_vector=noise_input, sample_idcs=ran_idcs,
#         jit_type=noise_jit_type, jit_lvl=noise_jit_lvl
#     )
#     recon_fig.suptitle('reconstruction ' + fig_title)
#     recon_fig.show()
#
#     plt.figure(figsize=(10, 6))
#     plt.subplot(311)
#     plt.plot(rep.mean(axis=1).mean(axis=0))
#     plt.subplot(312)
#     plt.plot(test_net.errors['layer_0']['ppe_pyr'])
#     plt.subplot(313)
#     plt.plot(test_net.errors['layer_0']['npe_pyr'])
#     plt.suptitle(f'Population activity ' + fig_title)
#     plt.show()

# from script.pc_network import noise_generator
# import matplotlib.pyplot as plt
# from script.pc_network import noise_generator, test_reconstruction
#
# noise_type = 'constant'
# noise_lvls = [0.0]
# for noise_lvl in noise_lvls:
#     # noise_input, _ = noise_generator(
#     #     noise_type=noise_type, noise_lvl=noise_lvl, target_shape=dataset['test_x'].shape
#     # )
#     # noise_input = np.clip(noise_input + dataset['test_x'], 0.0, 1.0)
#
#     recon_fig, test_net, rep = test_reconstruction(
#         sim_param=sim_params, weights=pretrained_weights,
#         layer=0, input_vector=dataset['test_x'], sample_idcs=ran_idcs,
#         jit_type=noise_type, jit_lvl=noise_lvl
#     )
#     recon_fig.suptitle(f'reconstruction with noise = {noise_lvl}')
#     recon_fig.show()
#
#     plt.figure(figsize=(10, 6))
#     plt.subplot(311)
#     plt.plot(rep.mean(axis=1).mean(axis=0))
#     plt.subplot(312)
#     plt.plot(test_net.errors['layer_0']['ppe_pyr'])
#     plt.subplot(313)
#     plt.plot(test_net.errors['layer_0']['npe_pyr'])
#     plt.suptitle(f'noise = {noise_lvl}')
#     plt.show()
