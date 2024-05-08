# import tensorflow as tf
import numpy as np
import tqdm
from parameters import *

class si_circuit(object):

    def __init__(self, input_mat, w_mat):
        # firing rates: pyramidal, pv, sst, pv
        self.frs = np.zeros((len(input_mat), 4))
        # incoming input to each neuron
        self.input_mat = input_mat
        # weights between neurons
        self.w_mat = w_mat
        # # simulation time
        # self.sim_time = sim_time

        # background input to each neuron
        self.Ibg = np.array([4e-3, 3e-3, 3e-3, 3e-3])
        # time constant for each neuron
        self.tau = np.array([60e-3, 2e-3, 2e-3, 2e-3])

    # spiking approximator #1
    def ReLu(self, x, theta=1):
        new_x = np.subtract(x, theta)

        return np.maximum(new_x, 0.0)

    # spiking approximator #2
    def jorge(self, x, d=10.0, theta=1):
        return (x - theta) / (1 - np.exp(-d * (x - theta)))

    # spiking approximator #3
    def sigmoid(x):
        return 100 / (1 + np.exp(-(x - 10)))

    #
    def drdt(self, r, Isyn, spiking_func, *args):
        return -r + spiking_func(Isyn, *args)

    def __call__(self, Ibg, Iv, Im):

        # scale_apical = np.array([[1,1], [1, 1], [1,0.5], [1,1]])

        for i, inp_i in enumerate(tqdm.tqdm(self.input_mat, position=0, desc='input type', leave=False, colour='green')):

            Isyn = inp_i @ np.array([Iv, Im]) + self.w_mat @ self.frs[i, :] + self.Ibg
            self.frs += (dt / self.tau) * self.drdt(r=self.frs[i, :], Isyn=Isyn)

            print(f'{i + 1}/{len(self.input_mat)} done!')


# for t in tqdm.tqdm(range(int(T / dt)), position=1, desc='sim time', leave=False, colour='red'):
