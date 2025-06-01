# %%

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_layers = 6
n_head = 6
dropout = 0.2

torch.manual_seed(1337)


with open('input.txt') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}


with open('input.txt') as f:
    text = f.read()

chars = sorted(list(set(text)))


stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda i: ''.join([itos[id] for id in i])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    
    return x,y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False) # (C, head_size)
        self.query = nn.Linear(n_embd, head_size, bias=False) # (C, head_size)
        self.value = nn.Linear(n_embd, head_size, bias=False) 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  #  (B, T, C)
        q = self.query(x)  #  (B, T, C)

        # compute attention scores ('affinities')

        wei = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size**0.5) # (B, T, T)
        masked_wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(masked_wei, dim = -1) # (B, T, T)
        wei = self.dropout(wei)

        # perform weighted avg of values by mat mul
        v = self.value(x)
        out = torch.matmul(wei, v)

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for i in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape:  (B, T, C)
        out  = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):
    '''a simple linear layer followed by non linearity'''

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) # (B, T, C) -> (B, T, C)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size) # i.e. 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x)) # apply one head of self attn (B, T, C)
        x = x + self.ffwd(self.ln2(x)) # apply simple MLP : (B, T, C)
        return x


class bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # positional embedding
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self-attention
        #self.ffwd = FeedForward(n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head = n_head) for _ in range(n_layers)]
            
        )
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):

        B, T = idx.shape
        # idx and targets are both of shape (batch_size, block_size) (B, T)
        tok_emb = self.token_embedding_table(idx)
        # postional embedding
        pos_emb = self.positional_embedding_table(torch.arange(0, T, device=device)) # (T, C)
        # add token embedding and position embedding
        x = tok_emb + pos_emb # (B, T, C)
        #x = self.sa_heads(x) # apply one head of self attn (B, T, C)
        #x = self.ffwd(x) # apply simple MLP : (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x) # (B, T, C)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T,)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is of shape (batch_size, block_size)
        for _ in range(max_new_tokens):
            
            # crop context because of now we have pos embd
            idx_cond = idx[:, -block_size:]

            logits, _ = self.forward(idx_cond)
            # sample the next token
            # focus on the last token because it is a bigram model
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            idx = torch.cat([idx, idx_next], dim = 1)
        return idx
    
@torch.no_grad()
def estimate_loss():
    out = {}
    # set model to  eval mode
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_eval, y_eval = get_batch(split)
            logits, loss = model(x_eval, y_eval)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out  
    

model = bigram().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print('losses', losses)
        print(f'Iter {i} train Loss ', losses)
        #print(f'Iter {i} Loss {losses}')
    #   pass

    xb, yb = get_batch('train')

    #eval the loss
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%
context = torch.zeros((1,1), dtype=torch.long, device = device)

print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))


# %%

# Save the model's state dictionary
torch.save(model.state_dict(), 'bigram_model.pth')

# %%
