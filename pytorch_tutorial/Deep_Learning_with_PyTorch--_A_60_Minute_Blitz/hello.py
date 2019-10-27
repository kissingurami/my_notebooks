import torch
import numpy as np


W = torch.Tensor([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6]
])

x = torch.Tensor([1, 2, 3, 4])

print(W.matmul(x))


import torch.nn as nn
fc = nn.Linear(in_features=4, out_features=3)



