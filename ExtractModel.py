import csv
import torch
from prettytable import PrettyTable
import numpy as np
import os
import sys
import pandas as pd


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


base_path = "./Models/px4/1/"

# Load Model from local
dtype = torch.float64
model = torch.load(os.path.join(base_path, "controller.pt"))
count_parameters(model)
model_state_dict = model.state_dict()


paths = [os.path.join(base_path, "W1_05_01_LTI_zero_eq_torch.csv"),
         os.path.join(base_path, "W2_05_01_LTI_zero_eq_torch.csv"),
         os.path.join(base_path, "b1_05_01_LTI_zero_eq_torch.csv"),
         os.path.join(base_path, "b2_05_01_LTI_zero_eq_torch.csv")]
parameters = [model_state_dict['0.weight'],
              model_state_dict['2.weight'],
              model_state_dict['0.bias'],
              model_state_dict['2.bias']]

# Write parameters to csv
for path, par in zip(paths, parameters):
    df = pd.DataFrame(par.numpy())
    df.to_csv(path, index=False, header=False)


# Test correctness model
x = np.random.rand(9)

W1 = pd.read_csv(paths[0], header=None).to_numpy()  # 7x9
W2 = pd.read_csv(paths[1], header=None).to_numpy()  # 4x7
B1 = pd.read_csv(paths[2], header=None).to_numpy()  # 7x1
B2 = pd.read_csv(paths[3], header=None).to_numpy()  # 4x1
hid1 = np.dot(W1, x) + B1.flatten()
hid1 = np.where(hid1 < 0, 0.01 * hid1, hid1)
out = np.dot(W2, hid1) + B2.flatten()

model.eval()
out_torch = model(torch.tensor(x))

print(f"Custom Net output: {out}")
print("--------------------")
print(f"Pytorch Net output: {out_torch.detach().numpy()}")
