import torch
from torch import optim

a = torch.tensor([[0.2, 0.3], [0.3, 0.2]], requires_grad=False)
w = torch.rand((2, 2), requires_grad=True)

b = torch.tensor([[0.1, 0.2], [0.1, 0.2]], requires_grad=False)
op = optim.Adam([w])

for i in range(20):
    op.zero_grad()
    loss = torch.sum(torch.matmul(a, w) - b)
    loss.backward()
    op.step()

print(w)
print(torch.matmul(a, w))
print(a)
print(b)
