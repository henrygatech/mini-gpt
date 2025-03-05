import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# %%

torch.manual_seed(1337)

B, T, C = 4,8,2

x = torch.randn(B, T, C)
x.shape



# %%

xbow = torch.zeros((B, T, C))

for b in range(B):
    for t in range(T):
        print('b', b, 't', t)
        print(x[b, :t+1].shape)
        #x[b,:t+1]
        xbow[b, t] = torch.mean(x[b, :t+1], dim = 0)

# %%

weight = torch.tril(torch.ones(T,T))

weight = weight/torch.sum(weight, dim = 1, keepdim=True)

xbow2 = torch.matmul(weight, x)

# %%


torch.manual_seed(42)


a = torch.tril(torch.ones(3,3))
a = a/torch.sum(a, dim=1, keepdim=True)

print(a)

b = torch.randint(0,10,(3,2),dtype=torch.float)

torch.matmul(a,b)

# %%


tril = torch.tril(torch.ones(T,T))

wei = torch.zeros((T, T))

wei = wei.masked_fill(tril==0, float('-inf'))

wei = F.softmax(wei,dim = -1, dtype= torch.float)

xbow3 = torch.matmul(wei, x) # (T, T) * (B, T, C) -> (B, T, C) 


# %%


tril = torch.tril(torch.ones(T,T))

wei = torch.zeros((T, T))

wei = wei.masked_fill(tril==0, float('-inf'))

wei = F.softmax(wei,dim = -1, dtype= torch.float)

xbow3 = torch.matmul(wei, x) # (T, T) * (B, T, C) -> (B, T, C) 


# %%


torch.manual_seed(1337)

B, T, C = 4, 8, 32

x = torch.randn(B, T, C)

# single heaad attn

head_size = 16

key = nn.Linear(C, head_size, bias=False)

query = nn.Linear(C, head_size, bias=False)

value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, head_size)
q = query(x)
v = value(x)
wei = torch.matmul(k, q.transpose(1,2)) # (B, T, T)


tril = torch.tril(torch.ones(T,T))

wei = wei.masked_fill(tril==0, float('-inf'))


wei = F.softmax(wei,dim = -1, dtype= torch.float)

xbow3 = torch.matmul(wei, v) 



# %%

#
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)

wei = torch.matmul(k, q.transpose(-2,-1))# / head_size**0.5




# %%
#torch.softmax(torch.tensor([0,1 , -0.2, 0.3]), dim=-1)

torch.softmax(torch.tensor([0,1 , -0.2, 0.3])*10, dim=-1)
# %%
