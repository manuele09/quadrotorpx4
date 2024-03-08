import hTorch.htorch.layers as layers
import hTorch.htorch.quaternion as quaternion
import torch
import numpy as np


x = torch.tensor([1., 2., 3., 4.])

model = torch.nn.Sequential(
    layers.QLinear(1, 2),
    torch.nn.ReLU(),
    layers.QLinear(2, 1),
)



d = model.state_dict()
num_layers = int(len(d.keys())/4)
# print(d)

r_weight = []
i_weight = []
j_weight = []
k_weight = []
bias = []

for i in range(num_layers):
    r_weight.append(d[f"{i*2}.r_weight"])
    i_weight.append(d[f"{i*2}.i_weight"])
    j_weight.append(d[f"{i*2}.j_weight"])
    k_weight.append(d[f"{i*2}.k_weight"])
    bias.append(d[f"{i*2}.bias"])

# print(r_weight)
# print(i_weight)
# print(j_weight)
# print(k_weight)
# print(bias)

weight = []
for i in range(num_layers):
    weight.append(torch.cat([torch.cat([r_weight[i], -i_weight[i], -j_weight[i],  -k_weight[i]], dim=0),
                            torch.cat([i_weight[i],  r_weight[i], -k_weight[i],   j_weight[i]], dim=0),
                            torch.cat([j_weight[i],  k_weight[i],  r_weight[i],  -i_weight[i]], dim=0),
                            torch.cat([k_weight[i], -j_weight[i],  i_weight[i],   r_weight[i]], dim=0)], dim = 1))
    

hid1 = np.dot(torch.t(weight[0]), x) + np.array(bias[0].flatten())
hid1 = np.where(hid1 < 0, 0.01 * hid1, hid1)
out = np.dot(torch.t(weight[1]), hid1) + np.array(bias[1].flatten())

model.eval()
out_torch = model(x)

print(f"Custom Net output: {out}")
print("--------------------")
print(f"Pytorch Net output: {out_torch.detach().numpy()}")

# print(weight[0].t)  
# return Q(F.linear(x, weight.t(), self.bias))