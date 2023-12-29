import torch


inputs = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], requires_grad=True)

t1 = torch.tensor(
    [
        [1.1, 1.2, 1.3, 1.4],
        [1.5, 1.6, 1.7, 1.8],
        [1.9, 2.0, 2.1, 2.2],
        [2.3, 2.4, 2.5, 2.6],
    ],
    requires_grad=True,
)

t2 = torch.tensor([[1.1, 1.2], [1.3, 1.4]], requires_grad=True)
t3 = torch.tensor([[1.5, 1.6], [1.7, 1.8]], requires_grad=True)

t4 = torch.matmul(inputs, t1)

t6 = t4[..., 0:2]
t7 = t4[..., 2:]

t8 = torch.matmul(t2, t6)
t9 = torch.matmul(t3, t7)

t10 = torch.matmul(t9, t8)

res = t10.mean()

print("MEAN", res)

res.backward()

print("t1 -", t1.grad)
print("t2 - ", t2.grad)
print("t3 -", t3.grad)
