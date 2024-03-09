import hTorch.htorch.layers as layers
import hTorch.htorch.quaternion as quaternion
import torch
import numpy as np
import os
import pandas as pd

def tensorToCsv(tensor, path):
    df = pd.DataFrame(tensor.numpy())
    df.to_csv(path, index=False, header=False)


def extractQuaternionWeights(model):
    d = model.state_dict()
    num_layers = int(len(d.keys())/4)

    weigths = {}
    weigths["r_weight"] = []
    weigths["i_weight"] = []
    weigths["j_weight"] = []
    weigths["k_weight"] = []
    weigths["bias"] = []

    for i in range(num_layers):
        weigths["r_weight"].append(d[f"{i*2}.r_weight"])
        weigths["i_weight"].append(d[f"{i*2}.i_weight"])
        weigths["j_weight"].append(d[f"{i*2}.j_weight"])
        weigths["k_weight"].append(d[f"{i*2}.k_weight"])
        weigths["bias"].append(d[f"{i*2}.bias"])


    w = []
    for i in range(num_layers):
        w.append(torch.cat([torch.cat([weigths["r_weight"][i], -weigths["i_weight"][i], -weigths["j_weight"][i],  -weigths["k_weight"][i]], dim=0),
                            torch.cat([weigths["i_weight"][i],  weigths["r_weight"][i], -weigths["k_weight"][i],   weigths["j_weight"][i]], dim=0),
                            torch.cat([weigths["j_weight"][i],  weigths["k_weight"][i],  weigths["r_weight"][i],  -weigths["i_weight"][i]], dim=0),
                            torch.cat([weigths["k_weight"][i], -weigths["j_weight"][i],  weigths["i_weight"][i],   weigths["r_weight"][i]], dim=0)], dim = 1))
    for i in range(num_layers):
        w[i] = torch.t(w[i])
        weigths["bias"][i] = torch.t(weigths["bias"][i])
    
    return w, weigths["bias"]

def modelQuaternionToCsv(model, path):
    weigths, biases = extractQuaternionWeights(model)
    d = model.state_dict()
    num_layers = int(len(d.keys())/4)
    
    names = []
    for i in range(num_layers):
        names.append("W" +  str(i+1))
        names.append("B" +  str(i+1))

    paths = []
    for name in names:
        paths.append(os.path.join(path, name + ".csv"))
    
    parameters = []
    for i in range(num_layers):
            parameters.append(weigths[i])
            parameters.append(biases[i])

    for path, par in zip(paths, parameters):
        tensorToCsv(par, path)
    return paths


x = torch.tensor([1., 2., 3., 4.])

model = torch.nn.Sequential(
    layers.QLinear(1, 2),
    torch.nn.ReLU(),
    layers.QLinear(2, 2),
    torch.nn.ReLU(),
    layers.QLinear(2, 1)
)

modelQuaternionToCsv(model, "./Test")

w, b = extractQuaternionWeights(model)
out = x
for i in range(len(w)):
    out = np.dot(w[i], out) + np.array(b[i].flatten())
    if i < len(w) - 1:
        out = np.where(out < 0, 0.01 * out, out)


model.eval()
out_torch = model(x)

print(f"Custom Net output: {out}")
print("--------------------")
print(f"Pytorch Net output: {out_torch.detach().numpy()}")
