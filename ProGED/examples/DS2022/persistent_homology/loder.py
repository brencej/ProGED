import pickle as p
name = "ph_lorenz_systems11_16__15_1703.p"
name = "ph_lorenz_systems11_17__09_2852.p"
data = p.load(open(name, "rb"))
model = data[0][0]
# model.
preamble = data[1]

print(2)
