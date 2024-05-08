# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from adex_const import *
#
# class neuron(object):
#
#     def __init__(self, type, number, batch_size=1, dt=1e-4, t_record=100):
#
#         # neuron type
#         self.neuron_tpye = type
#         # membrane potential
#         self.v = tf.ones(shape=(number, batch_size)) * EL
#         # adaptation parameter in AdEx neuron
#         self.c = tf.zeros(shape=(number, batch_size))
#         # synaptic current
#         self.x = tf.zeros(shape=(number, batch_size))
#         self.xtr = tf.zeros(shape=(number, batch_size))
#         # refractory
#         self.ref = tf.zeros(shape=(number, batch_size))
#         # firing count
#         self.fs = tf.zeros(shape=(number, batch_size))
#         # total incoming current
#         self.isyn = tf.zeros(shape=(number, batch_size))
#         # mean current
#         self.mean_current = tf.zeros(shape=(number, batch_size))
#
#         self._timestep = 0
#         self.dt = dt
#
#         self.t_record = t_record
#
#     def __call__(self, isynapse):
#         # isyn
#         self.isyn = isynapse
#
#         # update membrane potential
#         ref_constraint = self.threshold(self.ref, 0)
#         self.v += self.dv(thres=ref_constraint)
#         self.c += self.dc(thres=ref_constraint)
#
#         # discount refractory
#         self.ref = tf.maximum(self.ref - 1, 0)
#
#         # update synpatic current
#         self.x += self.dx()
#         self.xtr += self.dxtr()
#
#         # reset variables
#         # fired or not (spike train)
#         fired = self.threshold(self.v, VT)
#         # if fired, x = -x_reset; else, x = new_x
#         self.x = fired * -x_reset + (1 - fired) * self.x
#         # if fired, v = EL; else, v = max(new_v, EL)
#         self.v = tf.maximum(fired * Vr + (1 - fired) * self.v, Vr)
#         # if fired, c = new_c + b; else, c = new_c
#         self.c += b * fired
#         # save firing count
#         self.fs = fired
#
#         # apply refractory
#         self.ref += fired * (t_ref / self.dt)
#
#         # accumulate output current
#         if self._timestep >= self.t_record:
#             self.mean_current += self.xtr
#
#         # add time step
#         self._timestep += 1
#
#
#     @staticmethod
#     def threshold(vals, thres):
#         return tf.cast(tf.greater(vals, thres), dtype=tf.float32)
#
#     def dv(self, thres):
#         return (1 - thres) * (self.dt / Cm) * (
#                 gL * (EL - self.v) + gL * DeltaT * tf.exp((self.v - VT) / DeltaT) + self.isyn - self.c)
#
#     def dc(self, thres):
#         return (1 - thres) * (self.dt / tauw) * (a * (self.v - EL) - self.c)
#
#     def dx(self):
#         return (-self.x / tau_rise) * self.dt
#
#     def dxtr(self):
#         return (-self.x / tau_rise - self.xtr / tau_s) * self.dt
#
#
# # create a toy sample: ex. a sqaure
# toy_sample = np.ones((4,4)) * 2000e-12
# toy_sample[1:3, 1:3] = 0
# # visualize
# plt.imshow(toy_sample)
# plt.axis('off')
# plt.show()
#
# # create neurons: p0 = input, error circuit neurons divide into two for + & -, p1 = prediction
# input_size = int(len(toy_sample.flatten()))
# p1_size = 100
#
# neuron_types = ['pc', 'pv', 'sst', 'vip']
# # layer_size = [input_size]
#
# pred_dict = {'p0': neuron(type='pc', number=input_size),
#              'p1': neuron(type='pc', number=p1_size)}
#
# err_dict = {}
# for n_type in neuron_types:
#     err_dict[n_type] = neuron(type=n_type, number=input_size * 2)
#
# # connect
# conn_mat = tf.constant([[0, -1, -1, 0],
#                         [1, 0, 0, 0],
#                         [1, 0, 0, -1],
#                         [1, 0, 0, 0]])
#
# btw_weights = tf.ones((input_size, p1_size)) * 1e-4
# win_weights = tf.ones((p1_size, p1_size)) * 1e-4
#
# # inject inputs
#
# pred_dict['p0'](toy_sample.flatten())
# pred_dict['p1'](np.random.uniform(low=0, high=1000e-12, size=(p1_size, 1)))
#
# npe_inp_mat = [[1,1],
#                [1,0],
#                [1,1],
#                [0,1]]
#
# ppe_inp_mat = [[1,0],
#                [0,1],
#                [0,0],
#                [0,1]]
#
# isyn_dict = {}
# for i, n_type in enumerate(neuron_types):
#
#     # pred -> err
#     bu_input = tf.tile(pred_dict['p0'].xtr, [2])
#     td_input = tf.tile(btw_weights @ pred_dict['p1'].xtr, [2])
#     bu_mask = tf.repeat(tf.constant([npe_inp_mat[i, 0], ppe_inp_mat[i, 0]]), [input_size])
#     td_mask = tf.repeat(tf.constant([npe_inp_mat[i, 1], ppe_inp_mat[i, 1]]), [input_size])
#     between_input = bu_input * bu_mask + td_input * td_mask
#
#     # within err circuit
#     tf.concat([err_dict[key].xtr for key, grp in err_dict.items()], axis=1)
#
