#%%
words = open('names.txt', 'r').read().splitlines()




# %%
#list1 = [1, 2, 3]
#list2 = ['a', 'b', 'c']

#zipped = zip(list1, list2)
#(1, 'a')
#(2, 'b')
#(3, 'c')

b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]): # zip will iterate two
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        
sorted(b.items(), key= lambda kv: kv[1], reverse=True)


# %%

import torch

char = sorted(list(set(''.join(words))))

for s, i in enumerate(char):
    print(s, i)

stoi = {s:i+1 for i, s in enumerate(char)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}



# %%

N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): # zip will iterate two
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# %%

import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(N)
# %%

p = N[0].float()

p = p/p.sum()

# %%

g = torch.Generator().manual_seed(2147483647)

p = torch.rand(3, generator=g)

p = p/p.sum()


#%%
P = (N+1).float()
P = P/torch.sum(N,-1,keepdim=True)

# %%


g = torch.Generator().manual_seed(2147483647)



for i in range(50):
    idx = 0

    out = []

    while True:
        #p = N[idx].float()
        #p = p/p.sum()
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        if idx == 0:
            break

    print(''.join(out))

# %%

loglikihood = 0
n = 0

#for w in words:
for w in ['henryq']:
    chs = ['.'] + list(w) + ['.']

    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        loglikihood += logprob
        n += 1
        print(ch1, ch2, prob.item(), logprob.item())

print('nnl=', -loglikihood/n)
# %%

# create the training set, (x,y)
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(itos[ix1], itos[ix2])
        #N[ix1, ix2] += 1
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)





# %%

import torch
import torch.nn.functional as F


g = torch.Generator().manual_seed(2147483647)

W = torch.randn((27, 27), generator=g, requires_grad=True)
num = xs.nelement()


for i in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = - probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()

    print(loss.item())

    W.grad = None
    loss.backward()

    W.data += -50 * W.grad


# %%
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0

    while True:
        # ------
        # BEFORE:
        #p = P[ix]
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts/counts.sum(1, keepdim=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix ==0 :
            break
    print(''.join(out))


# %%
