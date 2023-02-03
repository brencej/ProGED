
import torch
import pandas as pd
import numpy as np
import sympy as sp
import os, sys
import symbolicregression
import requests


DEFAULT_SETTINGS = {}

model_path = "model.pt"
model_path = "/home/bosg/EDtools/ete-sr-trans/symbolicregression/model.pt"

# try:
if True:
    if not os.path.isfile(model_path):
        raise FileNotFoundError
        # url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
        # r = requests.get(url, allow_redirects=True)
        # open(model_path, 'wb').write(r.content)
    if not torch.cuda.is_available():
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
        model = model.cuda()
    # print(f"{model.device}")
    print(model.device)
    print("Model successfully loaded!")


# except Exception as e:
#     print("ERROR: model not loaded! path was: {}".format(model_path))
#     print(e)

def ete(x, y, settings=DEFAULT_SETTINGS):
    """
    Settings are:
        - max_input_points
        - n_trees_to_refine
        - rescale

    Data:
        -x: shape (n, 2)
        -y: shape (n, ), i.e. (n, 0) or (1, n)?
    """


    est = symbolicregression.model.SymbolicTransformerRegressor(
                            model=model,
                            max_input_points=300,
                            n_trees_to_refine=100,
                            rescale=True
                            )

    est.fit(x, y)
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
    for op, replace_op in replace_ops.items():
        model_str = model_str.replace(op, replace_op)
    return sp.parse_expr(model_str)


if __name__ == "__main__":
    ##Example of data

    x = np.random.randn(100, 2)
    y = np.cos(2*np.pi*x[:,0])+x[:,1]**2
    # (x,y)0
    print(x.shape, y.shape)

    # # ales challenge
    # csv = pd.read_csv('alesed.csv')
    # cs = np.array(csv)
    # x = cs[:, [0]]
    # y = cs[:, 1]
    # print(x.shape, y.shape)
    # # 1/0
    #
    res = ete(x, y)

    print(res)



