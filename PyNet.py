import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import os


def tensorToCsv(tensor, path):
    df = pd.DataFrame(tensor.numpy())
    df.to_csv(path, index=False, header=False)

def modelToCsv(model, path):
    d = model.state_dict()
    num_layers = int(len(d.keys())/2)

    names = []
    for i in range(num_layers):
        names.append("W" +  str(i+1))
        names.append("B" +  str(i+1))

    paths = []
    for name in names:
        paths.append(os.path.join(path, name + ".csv"))

    parameters = []
    for i in range(num_layers):
            parameters.append(d[f"{i*2}.weight"])
            parameters.append(d[f"{i*2}.bias"])

    for path, par in zip(paths, parameters):
        tensorToCsv(par, path)
    return paths


model = torch.nn.Sequential(
    nn.Linear(9, 7),
    nn.ReLU(),
    nn.Linear(7, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
paths = modelToCsv(model, "./Test")

