import torch
dim = 3
A = torch.rand(dim, dim, requires_grad=False)
b = torch.rand(dim, 1,  requires_grad=False)
x = torch.autograd.Variable(torch.rand(dim, 1), requires_grad=True)
stop_loss = 1e-2
step_size = stop_loss / 3.0
print(A)
print(b)
print('Loss before: %s' % (torch.norm(torch.matmul(A, x) - b)))
for i in range(1000*1000):
    s = torch.matmul(A, x) - b
    L = torch.norm(s, p=3)
    L.backward()
    x.data -= step_size * x.grad.data 
    x.grad.data.zero_()
    if i % 10000 == 0: print('Loss is %s at iteration %i' % (L, i))
    if abs(L) < stop_loss:
        print('Это потребовало %s итерации чтобы достичь %s ошибки.' % (i, step_size))
        break
print('Loss after: %s' % (torch.norm(torch.matmul(A, x) - b)))
print(x)