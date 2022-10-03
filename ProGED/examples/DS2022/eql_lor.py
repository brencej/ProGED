import gzip
import pickle
import os

import numpy as np

# EQL data_util test:
from utils import to_float32, number_of_positional_arguments
filename = 'data/F1data_train_val'
# filename = 'data/F1data_test'
# filename = 'data/F1dataN_train_val'
# filename = 'data/F2data_train_val'
# filename = 'data/Lor146_data_train_val'
filename = 'data/VDPGrad_data_train_val'
filename = 'data/VDP_data_train_val_dx'
filename = 'data/VDP_data_train_val_dy'
# filename = 'data/LorGrad_data_train_val'
# filename = 'data/Lor2_data_train_val'
dataE = to_float32(pickle.load(gzip.open(filename, "rb"), encoding='latin1'))
print(type(dataE), type(dataE[0]), type(dataE[1]))
# print(dataE[0].shape, dataE[1].shape)
# print(type(dataE))
# print(len(data))
# 1/0



# Create lorenz

from ProGED.examples.DS2022.generate_data_ODE_systems import generate_ODE_data, lorenz, lorenz_stable, VDP

system = 'VDP'
inits = [-0.2, -0.8]
# data = generate_ODE_data(lorenz_stable, [1,1,1])[:, 1:]
data = generate_ODE_data(lorenz_stable, [1,1,1])
data = generate_ODE_data(system='VDP', inits=[-0.2, -0.8])
data = np.hstack((data, data[:,[0]]))
# data = np.array([[1.2, 2.3, 3.5], [1.0, 2.8, 3.8], [1.3, 2.9, 3.9]])



def Lor(t, x, y, z):
    """Requires 1 hidden layer."""
    y0 = 0.10*(y-x)
    return y0,

# x = data[:]
xs = data[:, 1]
ysi = data[:, 2]
dt = data[1][0]-data[0][0]
dx = np.gradient(xs, dt).reshape(-1, 1)
dy = np.gradient(ysi, dt).reshape(-1, 1)
# for i in train_val_set:
#     print(i.shape)

xs = data
input_dim = xs.shape[1]
# xs = np.random.uniform(lower, upper, (num_examples, input_dim)).astype(np.float32)
xs_as_list = np.split(xs, input_dim, axis=1)
# ys = Lor(*xs_as_list)
# ys = np.concatenate(ys, axis=1)
ysdx = np.hstack((xs, dx))
np.random.shuffle(ysdx)
ys = ysdx[:, :-1]
dx = ysdx[:, [-1]]

# train_val_set = (ys[:round(ys.shape[0]/2)], dx[:round(ys.shape[0]/2)])
# test_set = (ys[round(ys.shape[0]/2):], dx[round(ys.shape[0]/2):])
# test_set = (ys[:ys.nrows/2], dx[:ys.nrows/2])
# with gradients:
train_val_set_dx = (xs, dx)
train_val_set_dy = (xs, dy)

# print(train_val_set)
# 1/0

train_val_data_file_dx = 'data/VDP_data_train_val_dx'
train_val_data_file_dy = 'data/VDP_data_train_val_dy'
# test_data_file = 'data/VDPGrad_data_test'

pickle.dump(train_val_set_dx, gzip.open(train_val_data_file_dx, "wb"))
pickle.dump(train_val_set_dy, gzip.open(train_val_data_file_dy, "wb"))
# pickle.dump(train_val_set, gzip.open(test_data_file, "wb"))
1/0

# print(data[1])
# print(data[0])
# print(data[1][0])
# print(data[0][0])
print('dt', dt)
#




# print(type(data))

