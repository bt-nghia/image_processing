import torch
from torch import nn
from torch import optim


def loss(y, y_hat):
    return torch.abs(y - y_hat)


a = torch.tensor([2.], requires_grad=True)
b = torch.tensor([0.], requires_grad=False)
op = optim.Adam(params=[a])

for i in range(10000):
    op.zero_grad()
    l = loss(a * a - 2 * a + 1, b)
    l.backward()
    op.step()

print(a)
