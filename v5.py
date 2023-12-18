import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# v5: wrap them into Blocks. add Residual connections. multiplier 4 in ffwd, layer norm
# Build a character-level Biagram language model (our baseline for LLM)
# hyperparameters
batch_size = 32 # B, how many independent sequences will we process in parallel?
block_size = 8 # T, what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3 # self attention can't tolerate too high LR. reduce it to 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32

# ------------

torch.manual_seed(1337)

# Load the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text)) # ~1M characters in total

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) # C = 65

# print(''.join(chars))
# print(vocab_size) # 65 unique characters

# Encode the data
# create a mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)} # ch to int
itos = {i: ch for i, ch in enumerate(chars)} # int to ch

# encoder: take a string, output a list of integers
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(encode("hii there"), decode(encode("hii there")))


# Prep the data, split into train and validation sets
data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape, data.dtype)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# spot-check the data
# example: enumerate all possible hidden chunks in the training set.
# block_size = 8
# x = train_data[:block_size]
# y = train_data[1:block_size + 1]
# for t in range(block_size):
#     context = x[:t + 1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")

# Build data loader
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # batch_size is hardcoded to 4
    # random int between 0 to (1115394 - 8), of size (4, 1)
    ix = torch.randint(low = 0, high = len(data) - block_size, size=(batch_size,))
    # print(ix) # tensor([ 76049, 234249, 934904, 560986])
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # merely shift by one from x
    return x, y

# unit test the get_batch() function

# xb, yb = get_batch('train') # batch_size is hardcoded to 4 from global variable
# print('inputs:')
# print(xb.shape) # [4, 8] = [B, T]
# print(xb)
# print('targets:')
# print(yb.shape) # [4, 8] = [B, T]
# print(yb)

# print('----spill them out----')

# for b in range(batch_size): # batch dimension. 4 sequences
#     for t in range(block_size): # time dimension, add 1 char at a time, until 8 chars
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         # ie., 8 predictions in total for a sequence. and we have 4 sequences in 1 batch, so 4 * 8
#         print(f"when input is {context.tolist()} the target: {target}")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # tril variable. it is not a param and will not be updated via backprop.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# just a MLP
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # the inner layer in ffwd has dimensionality multiplier 4 
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # coming back down to n_embed, projection layer. 
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        # slight deviation from the original paper.
        # apply layer norm "before" self.sa transformation. ie., pre-norm formulation
        # apply layer norm "before" self.ffwd transformation. ie., pre-norm formulation
        self.ln1 = nn.LayerNorm(n_embed) # my implementation batch_norm.LayerNorm has some bugs
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # do a `x = x + ` : residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# build the model
# Bigram model: even though we feed in a sequence of 8 chars, we only use the current char, to predict the next char
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # holds token (ie., identity) information. What is this char? is it A? B? C?
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # (65, 65) embedding table
        # plug in a Linear layer between head and input, size (n_embed, vocab_size)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # holds positional info. Where was it? Was it a position 0, 1, 2, T - 1?
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head = 4),
            Block(n_embed, n_head = 4),
            Block(n_embed, n_head = 4),
            nn.LayerNorm(n_embed)
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape # (batch_size, context_length)
        # idx and targets are both (B,T) tensor of integers
        # B: batch, T: Time (context length), C: Channels (vocab_size)
        tok_emb = self.token_embedding_table(idx) # (B,T,C) = (4,8,65)
        pos_input = torch.arange(T, device = device) # [0, 1, 2, ..., T - 1] 
        pos_emb = self.position_embedding_table(pos_input) # maps pos_input to (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, C)

        if targets is None: # inference time
            loss = None
        else: # training time
            # pytorch cross entropy expects (M,N) as inputs
            # also, we are only using bigram, so we reshape logits to flatten 4x8x65 to a 32x65

            # ie., now we see it as 32 independent predictions, each of which is a 65-way classification problem
            # in xb = [24, 43, 58,  5, 57,  1, 46, 43],
            #    yb = [43, 58,  5, 57,  1, 46, 43, 39],
            #   when input is [24] the target: 43
            #   when input is [43] the target: 58 (even though the context given is [24, 43] we only use [43])
            #   when input is [43] the target: 39 (same idea)
            # same for the rest 3 sequences. in total, 32 predictions

            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        # in xb = [24, 43, 58,  5, 57,  1, 46, 43],
        # eg., [24, 43, 58] and we are predicting the next target [5]
        # we first take only (B, C) so the [58], and we apply softmax
        # and we get a 65 dim array, and we sample from it to get the next chart, eg., [8] (ie., wrong, not 5)
        # and we concat it to be [24, 43, 58, 8]
        # next run we are predicting [57]. Given [24, 43, 58, 8]
        # generate until max_new_tokens = 100 times and stop to avoid infinite loop

        for _ in range(max_new_tokens):
            # note: because we have positional embedding which has size at most T,
            # we have to ensure idx not exceed T (up to block_size)
            # crop idx to the last block_size tokens
            idx_crop = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_crop)

            # we are using a bigram model. it only take one char. So focus only on the last time step
            logits = logits[:, -1, :] # (B,T,C) becomes (B, C). We take the current char
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C), the prob of the next char among those 65 possible chars
            # convert the (B, C) to a single prediction. Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# See what this totally random model will generate
# given batch_size = 1, time = 1, holding a 0. 
# we kick off the generation by this [0], and we generate 100 chars
# .reshape(-1) get rid of batch_size dimension. can also use [0] instead of .reshape(-1)

model = BigramLanguageModel(vocab_size)
m = model.to(device)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100).reshape(-1).tolist()))

# Actually train the model
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))