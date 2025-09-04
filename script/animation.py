import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from tqdm import trange, tqdm

from script.tools import pickle_load, remove_top_right_spines
from script.pc_network import network

class animation_creator(object):

    def __init__(self, simParams, pretrained_weights):

        # create network
        self.net = network(simParams=simParams, pretrained_weights=pretrained_weights)

        # set simulation time
        self.t_infer = int(simParams['sim_time'] / simParams['dt'])

        self.n_pixel, n_rep_neurons = simParams['net_size']
        # create arrays to save errors
        self.ppe_save = np.zeros((self.n_pixel, self.t_infer))
        self.npe_save = np.zeros((self.n_pixel, self.t_infer))
        # create an array to save reconstructions across time
        self.recons = np.zeros((self.n_pixel, self.t_infer))

    def simulate(self, dataset, sample_idx=None):

        # select a sample
        if sample_idx is None:
            sample_idx = np.random.choice(np.arange(len(dataset['train_x'])))
        # input
        self.x = dataset['train_x'][sample_idx].reshape(-1, 1)

        # initialize network
        self.net.initialize_network(batch_size=self.x.shape[1])
        # initialize neural activity
        self.net.initialize_activity_log()

        # simulate
        for t in trange(self.t_infer):
            self.net.compute(self.x, record='pyr')
            self.ppe_save[:, t] = self.net.network['layer_0']['ppe_pyr'].reshape(-1)
            self.npe_save[:, t] = self.net.network['layer_0']['npe_pyr'].reshape(-1)
            self.recons[:, t] = (self.net.weights['01'].T @ self.net.network['layer_1']['rep_r']).reshape(-1)

    def generate_animation(self, filename, frame_step=5, fps=60):
        """
        Generate MP4 animation of simulation.
        frame_step: use every Nth frame (e.g., 5 means 2000 -> 400 frames)
        fps: frames per second in output video
        """

        # precompute reshaped images for efficiency
        img_shape = (int(np.sqrt(self.n_pixel)), int(np.sqrt(self.n_pixel)))
        recons_reshaped = self.recons.T.reshape(self.t_infer, *img_shape)
        ppe_reshaped = self.ppe_save.T.reshape(self.t_infer, *img_shape)
        npe_reshaped = self.npe_save.T.reshape(self.t_infer, *img_shape)

        # precompute x-axis
        x = np.arange(self.t_infer)

        # figure + layout
        plt.close('all')
        fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(nrows=3, ncols=4, figure=fig)

        subplot_titles = ['Rep', 'PE+', 'PE-']

        f_ax_rep_response = fig.add_subplot(spec[0, :3])
        f_ax_ppe_response = fig.add_subplot(spec[1, :3])
        f_ax_ppe_response.set_ylabel('population response', rotation=90)
        f_ax_npe_response = fig.add_subplot(spec[2, :3])
        f_ax_npe_response.set_xlabel('time (ms)')

        f_ax_rep_img = fig.add_subplot(spec[0, 3])
        f_ax_ppe_img = fig.add_subplot(spec[1, 3])
        f_ax_npe_img = fig.add_subplot(spec[2, 3])

        # configure line subplots
        for i, ax in enumerate([f_ax_rep_response, f_ax_ppe_response, f_ax_npe_response]):
            ax.set_title(subplot_titles[i])
            remove_top_right_spines(ax)
            ax.set_xlim(0, self.t_infer)
            ax.set_ylim(0.0, 1.0)
            if i < 2:
                ax.set_xticks([])

        # configure images
        for ax in [f_ax_rep_img, f_ax_ppe_img, f_ax_npe_img]:
            ax.axis('off')

        # persistent artists
        line1, = f_ax_rep_response.plot([], [], c='purple')
        line2, = f_ax_ppe_response.plot([], [], c='red')
        line3, = f_ax_npe_response.plot([], [], c='blue')

        im1 = f_ax_rep_img.imshow(np.zeros(img_shape), cmap='gray', vmin=0, vmax=1, animated=True)
        im2 = f_ax_ppe_img.imshow(np.zeros(img_shape), cmap='Reds', vmin=0, vmax=1, animated=True)
        im3 = f_ax_npe_img.imshow(np.zeros(img_shape), cmap='Blues', vmin=0, vmax=1, animated=True)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            im1.set_data(np.zeros(img_shape))
            im2.set_data(np.zeros(img_shape))
            im3.set_data(np.zeros(img_shape))
            return line1, line2, line3, im1, im2, im3

        def update(frame):
            line1.set_data(x[:frame], self.net.activity_log['layer_1']['rep_r'][:frame])
            line2.set_data(x[:frame], self.net.activity_log['layer_0']['ppe_pyr'][:frame])
            line3.set_data(x[:frame], self.net.activity_log['layer_0']['npe_pyr'][:frame])

            im1.set_data(recons_reshaped[frame])
            im2.set_data(ppe_reshaped[frame])
            im3.set_data(npe_reshaped[frame])
            return line1, line2, line3, im1, im2, im3

        # pick frames to use
        frames = list(range(0, self.t_infer, frame_step))

        # wrap frames with tqdm to show progress
        ani = animation.FuncAnimation(fig, update, frames=tqdm(frames, desc="Rendering"),
                                      init_func=init, blit=True, interval=10)

        writer = animation.FFMpegWriter(fps=fps)
        ani.save(filename, writer=writer)
