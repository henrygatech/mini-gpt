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

vocab_size = len(itos)

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

Xdev.shape, Xdev.dtype, Ydev.shape, Ydev.dtype
Xte.shape, Xte.dtype, Yte.shape, Yte.dtype

# %%

#MLP revisited 

n_embd = 10 # dim of embd
n_hidden = 200
block_size = 3

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g) 
# X [n, 3, 10]
W1 = torch.randn((n_embd*block_size, n_hidden), generator=g) *  (5/3)*((n_embd * block_size)**0.5) # 0.2
b1 = torch.randn(n_hidden, generator=g) *0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) *0.1
b2 = torch.randn(vocab_size, generator=g) * 0

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.zeros((1, n_hidden))

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
sum(p.nelement() for p in parameters)

#%%

for p in parameters:
    p.requires_grad = True

#%%

lre = torch.linspace(-3, 0, 1000)

lrs = 10**lre

#%%
lrei = []
stepi = []
lossi = []

# %%

#training

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X, Y

    # forward pass
    emd = C[Xb]
    emdcat = emd.view(-1, n_embd*block_size)
    hpreact = emdcat @ W1 + b1
    bnmeani = hpreact.mean(0, keepdim = True)
    bnstdi = hpreact.std(0, keepdim = True)
    hpreact = bngain * (hpreact - bnmeani)/(bnstdi + 1e-5) + bnbias
    
    # with running mean that used for inference time

    with torch.no_grad():
        bnmeani = 0.999 * bnmean_running + 0.001 * bnmeani #hpreact.mean(dim=0, keepdims = True)
        bnstdi =  0.999 * bnstd_running + 0.001 * bnstdi #hpreact.std(dim=0, keepdims = True)

    h = torch.tanh(hpreact)
    logits = h@W2 + b2 # output layer

    loss = F.cross_entropy(logits, Yb) #loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update 
    lr = 0.1 if i < max_steps/2 else 0.01 # step lr decay

    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 ==0:
        print(i, ':', loss.item())
        stepi.append(i)
        lossi.append(loss.log10().item())

    break
print(loss.item())


#%%

hpreact.shape

#%%
# good vis on model health example
plt.figure(figsize = (20, 10))
plt.imshow(h.abs() > 0.99, cmap = 'gray')
# %%
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte)
    }[split]

    emb = C[x] # (N, block_size, b_emd)
    embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_emd)
    h = torch.tanh(embcat@W1 + b1) # (N, n_hidden)

    logits = h@W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

# %%

g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size # init with all ...

    while True:
        # forward pass neural net
        emb = C[torch.tensor([context])] # (1, block_size, n_emb)
        
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)

        logits = h @ W2 + b2

        probs = F.softmax(logits, dim=1)

        # sample from dist

        ix = torch.multinomial(probs, num_samples=1, generator=g).item()

        # shift the ontext window and track the samples
        
        context = context[1:] +[ix]

        out.append(ix)

        # if we sample special '.' break
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))


#%%

plt.plot(stepi, lossi)

# %%

emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30)@W1+b1)
logits = h@W2 + b2
loss = F.cross_entropy(logits, Ytr)
loss

# %%

emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30)@W1+b1)
logits = h@W2 + b2
loss = F.cross_entropy(logits, Ydev)
loss

#%%

plt.figure(figsize=(8, 8))
plt.scatter(C[:,0].data, C[:, 1].data, s = 200)

for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha = "center", va = "center", color = 'white')

plt.grid('minor')

# %%
#torch.cat([emd[:,i,:] for i in range(block_size)], dim = 1) less efficient because it is creating new memory



# %%

(torch.randn(1000)*0.2).std()


# %%


(5/3) / (30**0.5)
# %%
