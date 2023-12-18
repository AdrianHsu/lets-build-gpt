import os
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B,T,C = 4,8,2
x = torch.randn(B,T,C)
print(x.shape)

# version 1. compute attention NOT by matmul, but by per-row mean
# bag of words
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # shape (t, C)
        xbow[b, t] = torch.mean(xprev, 0)
print(x[0])
print(xbow[0])

# some prerequisite knowledge to understand version 2.
# use Matrix multiplication to compute the same thing. Make it simplier
torch.manual_seed(42)
a = torch.ones(3, 3)
b = torch.randint(low = 0, high = 10, size = (3, 2)).float()
c = a @ b
print(c)

# triangular matrix

# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
a = torch.tril(torch.ones(3, 3))
print(a)

# make it weighted sum, and sum up to 1. to mimic mean function 
# tensor([[1.0000, 0.0000, 0.0000],
#         [0.5000, 0.5000, 0.0000],
#         [0.3333, 0.3333, 0.3333]])
a = a / torch.sum(a, 1, keepdim = True)
print(a)
c = a @ b
print(c)

# -----------------
# version 2
# now we really use it in the previous scenario: B,T,C
weights = torch.tril(torch.ones(T, T))
weights = weights / torch.sum(weights, 1, keepdim = True)
weights
# does not match? no. pytorch will ignore the batch dim, and apply matmul on (T, C) each time
xbow2 = weights @ x # (T, T) @ (B, T, C) -> (B,T,C) 

print(xbow2[0]) # same results as xbow[0] but more efficient via matmul
print(torch.allclose(xbow2, xbow))

# -----------------
# version 3: use SoftMax to achieve the same thing.
# we will use this to implement the self-attention block
tril = torch.tril(torch.ones(T, T))
weights = torch.zeros((T, T))
print(tril)
# all the weights which has 0, will become -Inf
weights = weights.masked_fill(tril == 0, float('-inf'))
print(weights)
weights = F.softmax(weights, dim = -1) # -Inf becomes 0, the rest will sum up to 1.0
print(weights)
xbow3 = weights @ x

print(xbow3[0]) # same results as xbow2[0], xbow[0] but more efficient via matmul
print(torch.allclose(xbow3, xbow))

# -----------------
# version 4: self-attention!

# query: what am i looking for?
# key: what do i contain?
torch.manual_seed(1337)
B,T,C=4,8,32 # bump up C from 2 to 32 for better illustration
x = torch.randn(B,T,C)

# let's see a single head performs self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B,T,16)
q = query(x) #(B,T,16)
k_t = k.transpose(-2, -1) # (B, 16, T)
# compute attention scores ("affinities")
# head_size **-0.5 is a scaling factor to normalize variance
weights = q @ k_t * (head_size **-0.5) # (B,T,16) @ (B,16,T) = (B,T,T), same as the shape we need in version 3
print(weights[0])

# now do we what we have done in version 3
weights = weights.masked_fill(tril == 0, float('-inf'))
print(weights[0])
weights = F.softmax(weights, dim = -1) # -Inf becomes 0, the rest will sum up to 1.0
print(weights[0])

v = value(x) # (B,T,16)
v = weights @ v 
print(v.shape) # (B,T,16)