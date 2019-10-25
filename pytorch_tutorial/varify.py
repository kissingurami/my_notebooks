from __future__ import print_function
import torch

torch.cuda.set_device(3)
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())

## Get Id of default device
print(torch.cuda.current_device())


