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
counts_sum = counts.sum(dim=1, keepdim=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv

logprobs = probs.log()

loss = -logprobs[range(n), Yb].mean()

# PyTorch backward pass
for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
          norm_logits, logit_max, logits, h, hpreact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmean_i,
         emb_cat, emb]:
  t.retain_grad()
loss.backward()
loss


# %%


# Exercise 1: backprop through the whole thing manually, 
# backpropagating through exactly all of the variables 
# as they are defined in the forward pass above, one by one

# -----------------
# YOUR CODE HERE :)
# -----------------

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n

cmp('logprobs', dlogprobs, logprobs)

dprobs = dlogprobs * 1/probs
cmp('probs', dprobs, probs)

dcounts_sum_inv = (counts * dprobs).sum(dim=1, keepdim=True)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

dcounts_sum = dcounts_sum_inv * (-counts_sum**-2)

cmp('counts_sum', dcounts_sum, counts_sum)

#dcounts = dprobs * counts_sum_inv + dcounts_sum 
dcounts = dprobs * counts_sum_inv + torch.ones_like(counts) * dcounts_sum 
cmp('counts', dcounts, counts)

dnorm_logits = dcounts * norm_logits.exp()
cmp('norm_logits', dnorm_logits, norm_logits)

#dlogit_maxes = -dnorm_logits[range(n), logits.max(dim = 1, keepdim = True).indices]

dlogit_maxes = -dnorm_logits.sum(dim=1, keepdim=True)
cmp('logit_maxes', dlogit_maxes, logit_max)

dlogits_2 = torch.zeros_like(logits)
dlogits_2[range(n), logits.max(dim = 1).indices] = dlogit_maxes.squeeze()

dlogits = dnorm_logits.clone() + dlogits_2
cmp('logits', dlogits, logits)
dh = dlogits @ W2.T
cmp('h', dh, h)

dW2 = h.T@dlogits
cmp('W2', dW2, W2)

db2 = dlogits.sum(dim=0)

cmp('b2', db2, b2)

dhpreact =  (1.0 - h**2) * dh
cmp('hpreact', dhpreact, hpreact)

dbngain = (dhpreact*bnraw).sum(dim = 0)

cmp('bngain', dbngain, bngain)

dbnbias = dhpreact.sum(dim = 0)
cmp('bnbias', dbnbias, bnbias)

dbnraw = dhpreact * bngain

cmp('bnraw', dbnraw, bnraw)

dbnvar_inv = (dbnraw * bndiff).sum(dim=0)

cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

dbnvar = dbnvar_inv * (-.5)*(bnvar+ 1e-5)**-1.5
cmp('bnvar', dbnvar, bnvar)

dbndiff2 = (dbnvar * 1/(n-1)).expand(n, -1)
cmp('bndiff2', dbndiff2, bndiff2)

dbndiff = bnvar_inv * dbnraw + dbndiff2 * 2 * bndiff

cmp('bndiff', dbndiff, bndiff)

dbnmean_i = -dbndiff.sum(dim = 0)

cmp('bnmean_i', dbnmean_i, bnmean_i)

dhprebn = dbndiff.clone() + 1/n*dbnmean_i.expand(n, -1)

cmp('hprebn', dhprebn, hprebn)

demb_cat = dhprebn@W1.T

cmp('emb_cat', demb_cat, emb_cat)

dW1 = emb_cat.T@dhprebn

cmp('W1', dW1, W1)

db1 = dhprebn.sum(dim = 0)
cmp('b1', db1, b1)

demb = demb_cat.view(n, -1,C.shape[1] )
cmp('emb', demb, emb)

dC = torch.zeros_like(C)

for i in range(Xb.shape[0]):
  for j in range(Xb.shape[1]):
    ix = Xb[i, j] # Xb i, j is the letter index in vocab
    dC[ix] += demb[i, j] # ix is the letter index in C/ vocab, demb[i, j] is the actual gradient of the embedding

cmp('C', dC, C)


# %%


# Exercise 2: backprop through cross_entropy but all in one go
# to complete this challenge look at the mathematical expression of the loss,
# take the derivative, simplify the expression, and just write it out

# forward pass

# before:
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # subtract max for numerical stability
# counts = norm_logits.exp()
# counts_sum = counts.sum(1, keepdims=True)
# counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
# probs = counts * counts_sum_inv
# logprobs = probs.log()
# loss = -logprobs[range(n), Yb].mean()

# now:
loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), 'diff:', (loss_fast - loss).item())


# %%

# backward pass

# -----------------
# YOUR CODE HERE :)
dlogits = None # TODO. my solution is 3 lines
sm = F.softmax(logits)
sm[range(n), Yb] -= 1
dlogits = 1.0/n *sm


# -----------------

cmp('logits', dlogits, logits) # I can only get approximate to be true, my maxdiff is 6e-9



# %%
# Exercise 3: backprop through batchnorm but all in one go
# to complete this challenge look at the mathematical expression of the output of batchnorm,
# take the derivative w.r.t. its input, simplify the expression, and just write it out
# BatchNorm paper: https://arxiv.org/abs/1502.03167

# forward pass

# before:
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias

# now:
hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
print('max diff:', (hpreact_fast - hpreact).abs().max())


#%%


# backward pass

# before we had:
# dbnraw = bngain * dhpreact
# dbndiff = bnvar_inv * dbnraw
# dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
# dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
# dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
# dbndiff += (2*bndiff) * dbndiff2
# dhprebn = dbndiff.clone()
# dbnmeani = (-dbndiff).sum(0)
# dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)

# calculate dhprebn given dhpreact (i.e. backprop through the batchnorm)
# (you'll also need to use some of the variables from the forward pass up above)

# -----------------
# YOUR CODE HERE :)
dhprebn = None # TODO. my solution is 1 (long) line

dhprebn =  (1.0/n*bnvar_inv* bngain) * ( (dhpreact*n) -dhpreact.sum(dim=0, keepdim = True)  -(n/(n-1))*(dhpreact*bnraw.clone()).sum(dim=0, keepdim = True)*bnraw)

# -----------------

cmp('hprebn', dhprebn, hprebn) # I can only get approximate to be true, my maxdiff is 9e-10


#%%

# Exercise 4: putting it all together!
# Train the MLP neural net with your own backward pass

# init
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

# same optimization as last time
max_steps = 200000
batch_size = 32
n = batch_size # convenience
lossi = []

# use this context manager for efficiency once your backward pass is written (TODO)
#with torch.no_grad():

# kick off optimization
for i in range(max_steps):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

  # forward pass
  emb = C[Xb] # embed the characters into vectors
  embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
  # Linear layer
  hprebn = embcat @ W1 + b1 # hidden layer pre-activation
  # BatchNorm layer
  # -------------------------------------------------------------
  bnmean = hprebn.mean(0, keepdim=True)
  bnvar = hprebn.var(0, keepdim=True, unbiased=True)
  bnvar_inv = (bnvar + 1e-5)**-0.5
  bnraw = (hprebn - bnmean) * bnvar_inv
  hpreact = bngain * bnraw + bnbias
  # -------------------------------------------------------------
  # Non-linearity
  h = torch.tanh(hpreact) # hidden layer
  logits = h @ W2 + b2 # output layer
  loss = F.cross_entropy(logits, Yb) # loss function

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward() # use this for correctness comparisons, delete it later!

  # manual backprop! #swole_doge_meme
  # -----------------
  # YOUR CODE HERE :)
  #dC, dW1, db1, dW2, db2, dbngain, dbnbias = None, None, None, None, None, None, None
  grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
  # -----------------

  # update
  lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
  for p, grad in zip(parameters, grads):
    p.data += -lr * p.grad # old way of cheems doge (using PyTorch grad from .backward())
    #p.data += -lr * grad # new way of swole doge TODO: enable

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())

  #if i >= 100: # TODO: delete early breaking when you're ready to train the full net
  #  break

# %%

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # init
# n_embd = 10 # the dimensionality of the character embedding vectors
# n_hidden = 200 # the number of neurons in the hidden layer of the MLP

# g = torch.Generator().manual_seed(2147483647) # for reproducibility
# C  = torch.randn((vocab_size, n_embd),            generator=g).to(device)
# # Layer 1
# W1 = (torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)).to(device)
# b1 = (torch.randn(n_hidden,                        generator=g) * 0.1).to(device)
# # Layer 2
# W2 = (torch.randn((n_hidden, vocab_size),          generator=g) * 0.1).to(device)
# b2 = (torch.randn(vocab_size,                      generator=g) * 0.1).to(device)
# # BatchNorm parameters
# bngain = (torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0).to(device)
# bnbias = (torch.randn((1, n_hidden), generator=g) * 0.1).to(device)

# parameters = [C, W1, b1, W2, b2, bngain, bnbias]
# print(sum(p.nelement() for p in parameters)) # number of parameters in total
# for p in parameters:
#   p.requires_grad = True

# # assume you load or define Xtr, Ytr before this
# Xtr = Xtr.to(device)
# Ytr = Ytr.to(device)

# # same optimization as last time
# max_steps = 200000
# batch_size = 32
# n = batch_size # convenience
# lossi = []

# #with torch.no_grad():

# # kick off optimization
# for i in range(max_steps):

#   # minibatch construct
#   ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
#   Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

#   # forward pass
#   emb = C[Xb] # embed the characters into vectors
#   embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
#   # Linear layer
#   hprebn = embcat @ W1 + b1 # hidden layer pre-activation
#   # BatchNorm layer
#   # -------------------------------------------------------------
#   bnmean = hprebn.mean(0, keepdim=True)
#   bnvar = hprebn.var(0, keepdim=True, unbiased=True)
#   bnvar_inv = (bnvar + 1e-5)**-0.5
#   bnraw = (hprebn - bnmean) * bnvar_inv
#   hpreact = bngain * bnraw + bnbias
#   # -------------------------------------------------------------
#   # Non-linearity
#   h = torch.tanh(hpreact) # hidden layer
#   logits = h @ W2 + b2 # output layer
#   loss = F.cross_entropy(logits, Yb) # loss function

#   # backward pass
#   for p in parameters:
#     p.grad = None
#   loss.backward() # use this for correctness comparisons, delete it later!

#   # manual backprop! #swole_doge_meme
#   # -----------------
#   # YOUR CODE HERE :)
#   #dC, dW1, db1, dW2, db2, dbngain, dbnbias = None, None, None, None, None, None, None
#   grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
#   # -----------------

#   # update
#   lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
#   for p, grad in zip(parameters, grads):
#     p.data += -lr * p.grad # old way of cheems doge (using PyTorch grad from .backward())
#     #p.data += -lr * grad # new way of swole doge TODO: enable

#   # track stats
#   if i % 10000 == 0: # print every once in a while
#     print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
#   lossi.append(loss.log10().item())

  #if i >= 100: # TODO: delete early breaking when you're ready to train the full net
  #  break

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g).to(device)
# Layer 1
W1 = (torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)).to(device)
b1 = (torch.randn(n_hidden,                        generator=g) * 0.1).to(device)
# Layer 2
W2 = (torch.randn((n_hidden, vocab_size),          generator=g) * 0.1).to(device)
b2 = (torch.randn(vocab_size,                      generator=g) * 0.1).to(device)
# BatchNorm parameters
bngain = (torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0).to(device)
bnbias = (torch.randn((1, n_hidden), generator=g) * 0.1).to(device)

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

# assume you load or define Xtr, Ytr before this
Xtr = Xtr.to(device)
Ytr = Ytr.to(device)

# same optimization as last time
max_steps = 200000
batch_size = 32
n = batch_size # convenience
lossi = []

#with torch.no_grad():

# kick off optimization
for i in range(max_steps):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

  # forward pass
  emb = C[Xb] # embed the characters into vectors
  embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
  # Linear layer
  hprebn = embcat @ W1 + b1 # hidden layer pre-activation
  # BatchNorm layer
  # -------------------------------------------------------------
  bnmean = hprebn.mean(0, keepdim=True)
  bnvar = hprebn.var(0, keepdim=True, unbiased=True)
  bnvar_inv = (bnvar + 1e-5)**-0.5
  bnraw = (hprebn - bnmean) * bnvar_inv
  hpreact = bngain * bnraw + bnbias
  # -------------------------------------------------------------
  # Non-linearity
  h = torch.tanh(hpreact) # hidden layer
  logits = h @ W2 + b2 # output layer
  loss = F.cross_entropy(logits, Yb) # loss function

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward() # use this for correctness comparisons, delete it later!

  # manual backprop! #swole_doge_meme
  # -----------------
  # YOUR CODE HERE :)
  #dC, dW1, db1, dW2, db2, dbngain, dbnbias = None, None, None, None, None, None, None


dbnbias = dhpreact.sum(dim = 0)



  grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
  # -----------------

  # update
  lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
  for p, grad in zip(parameters, grads):
    p.data += -lr * p.grad # old way of cheems doge (using PyTorch grad from .backward())
    #p.data += -lr * grad # new way of swole doge TODO: enable

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())

  #if i >= 100: # TODO: delete early breaking when you're ready to train the full net
  #  break



#%%

# calibrate the batch norm at the end of training

with torch.no_grad():
  # pass the training set through
  emb = C[Xtr]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1 + b1
  # measure the mean/std over the entire training set
  bnmean = hpreact.mean(0, keepdim=True)
  bnvar = hpreact.var(0, keepdim=True, unbiased=True)


 #%%

@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x, y = {
    'train': (Xtr, Ytr),
    'val':   (Xdev, Ydev),
    'test':  (Xte, Yte),
  }[split]
  x, y = x.to(device), y.to(device)  # move data to GPU

  emb = C[x] # (N, block_size, n_embd)
  embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  hpreact = embcat @ W1 + b1
  # recompute batch norm stats on eval set
  bnmean = hpreact.mean(0, keepdim=True)
  bnvar = hpreact.var(0, keepdim=True, unbiased=True)
  hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
  h = torch.tanh(hpreact) # (N, n_hidden)
  logits = h @ W2 + b2 # (N, vocab_size)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')


#%%
import torch
import torch.nn.functional as F

# force all model parameters to CPU
C      = C.cpu()
W1     = W1.cpu()
b1     = b1.cpu()
W2     = W2.cpu()
b2     = b2.cpu()
bngain = bngain.cpu()
bnbias = bnbias.cpu()

@torch.no_grad()
def sample_model_cpu():
    g = torch.Generator().manual_seed(2147483647 + 10)

    for _ in range(20):
        out = []
        context = [0] * block_size  # initialize with all ...

        while True:
            ctx_tensor = torch.tensor([context], dtype=torch.long, device='cpu')  # CPU input

            # forward pass
            emb = C[ctx_tensor]  # (1, block_size, n_embd)
            embcat = emb.view(1, -1)
            hpreact = embcat @ W1 + b1

            # recompute BN stats
            bnmean = hpreact.mean(0, keepdim=True)
            bnvar = hpreact.var(0, keepdim=True, unbiased=True)
            hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias

            h = torch.tanh(hpreact)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)

            ix = torch.multinomial(probs, num_samples=1, generator=g).item()

            # safety check
            if ix < 0 or ix >= vocab_size:
                raise ValueError(f"Sampled invalid index {ix} (vocab_size={vocab_size})")

            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print(''.join(itos[i] for i in out))

sample_model_cpu()




# %%
