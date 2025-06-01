#%%


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline 


#%%

words = open('names.txt', 'r').read().splitlines()

words[:8]


#%%

chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i,s in enumerate(chars)}

stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}


# %%


def build_dataset(words):
    block_size = 3
    X, Y = [], []

    for w in words:
        #print(w)
        context = [0] * block_size

        for ch in w + '.':
            ix = stoi[ch]
            #print('ix', ix)
            X.append(context)
            Y.append(ix)
            #print(''.join(itos[i] for i in context), '---->', itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

# %%

import random

random.seed(42)
random.shuffle(words)

n1 = int(0.8*len(words))

n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])



# %%

Xtr.shape, Xtr.dtype, Ytr.shape, Ytr.dtype

# %%

Xdev.shape, Xdev.dtype, Ydev.shape, Ydev.dtype
Xte.shape, Xte.dtype, Yte.shape, Yte.dtype

# %%

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]


#%%

for p in parameters:
    p.requires_grad = True

#%%

lre = torch.linspace(-3, 0, 1000)

lrs = 10**lre

# %%

lrei = []

lossi = []

for i in range(10000):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forward pass
    emd = C[Xtr[ix]]
    h = torch.tanh(emd.view(-1, 6)@W1+b1)
    logits = h@W2 + b2

    loss = F.cross_entropy(logits, Ytr[ix])

    #counts = logits.exp()
    #prob = counts/counts.sum(1, keepdim=True)
    #loss = -prob[torch.arange(32), Y].log().mean()


    # backward pass
    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data += -0.1 * p.grad
print(loss.item())

# %%

plt.plot(lrei, lossi)

#%%

torch.randint(0, X.shape[0], (32, ))


# %%
#torch.cat([emd[:,i,:] for i in range(block_size)], dim = 1) less efficient because it is creating new memory



# %%



# %%
