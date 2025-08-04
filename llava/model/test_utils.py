import torch

a = torch.Tensor([1,2,3])
b = [None]
c = torch.cat((b,a), dim=0)

print(c)