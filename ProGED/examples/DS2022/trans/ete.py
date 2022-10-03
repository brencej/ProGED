from  generate_data_ODE_systems import generate_ODE_data
import numpy as np
import pandas as pd

# general settings for simulation
generation_settings = {"initial_time": 0,
                       "simulation_step": 0.01,
                       "simulation_time": 50,
                       "method": 'LSODA',
                       "rtol": 1e-12,
                       "atol": 1e-12}

system = 'VDP'
inits = [-0.2, -0.8]
data = generate_ODE_data(system, inits, **generation_settings)

print('data:')
print(data)
# x = data[:]
ts = data[:, 0]
xs = data[:, 1]
ysi = data[:, 2]
dt = data[1][0]-data[0][0]
dx = np.gradient(xs, dt).reshape(-1, 1)
dy = np.gradient(ysi, dt).reshape(-1, 1)
# for i in train_val_set:
#     print(i.shape)

# ds_time = np.hstack((ts, , ))
ds_notime = np.hstack((data[:, [1, 2]], dy))
ds_write_dx = np.hstack((data[:, [1, 2]], dx))

# ds_notime = data[:, [1, 2]]
print('ds_notime:')
print(ds_notime)
print('ds_write_dx:')
print(ds_write_dx)



filename = "ete.csv"
filename_dx = "ete_dx.csv"
pd.DataFrame(ds_notime).to_csv(path_or_buf=filename, header=False, index=False)
pd.DataFrame(ds_write_dx).to_csv(path_or_buf=filename_dx, header=False, index=False)
np.savetxt("ete_st.csv", ds_notime, delimiter=",")

# ds = pd.read_csv(filename, header=False, index=False)
ds = pd.read_csv(filename)
ds_dx = pd.read_csv(filename_dx)
print('ds:')
print(ds)
print('ds_dx:')
print(ds_dx)


## Results:
# ete(dy):
# -0.494825968*x0**2*x1 + 0.000355264000000004*x0**2 - 1.6156e-5*x0*x1**2 - 0.008780198*x0*x1 - 0.989284422*x0 + 0.5*x1 - 0.018
# after trsh
# -0.49482*x**2*y - 0.98928*x + 0.5*y - 0.018
#
# time needed: 21.46s

## Results:
# ete(dx):
# 0.000105*x0*x1 + 1.029*x1
# after trsh
# 1.029*y
#
# time needed: 10.78s

