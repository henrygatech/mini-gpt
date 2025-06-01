import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
%matplotlib inline

#%%

# download the names.txt file from github
!wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt

#%%

# read in all the words
words = open('names.txt', 'r').read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])


#%%

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)


#%%

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

#%%

# ok biolerplate done, now we get to the action:

# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

# %%

n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of BN
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True


#%%

batch_size = 32
n = batch_size # a shorter variable also, for convenience
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

# %%

# forward pass, "chunkated" into smaller steps that are possible to backward on at a time

# linear layer 1
emb = C[Xb]
emb_cat = emb.view(emb.shape[0], -1)
hprebn = emb_cat@W1+b1

#batchnorm layer 

bnmean_i = 1/n*hprebn.sum(dim=0, keepdim= True)
bndiff = hprebn - bnmean_i
bndiff2 = bndiff**2
bnvar = 1/(n-1) * bndiff2.sum(dim=0, keepdim = True) # bessel's correction
bnvar_inv = (bnvar+ 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain*bnraw+bnbias

# non linearity

h = torch.tanh(hpreact) # hidden layer

# linear layer 2 

logits = h@W2 + b2

# cross entropy loss (same as F.cross_entropy(logits, Yb))

logit_max = logits.max(dim = 1, keepdim = True).values
norm_logits = logits - logit_max # substract max for numerical stability, softmax remain invariant 
counts = norm_logits.exp()
counts_sum = counts.sum(dim=0, keepdim=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv










# %%
