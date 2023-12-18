import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# Build a character-level Biagram language model (our baseline for LLM)
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# ------------

torch.manual_seed(1337)

# Load the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text)) # ~1M characters in total

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

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

# build the model
# Bigram model: even though we feed in a sequence of 8 chars, we only use the current char, to predict the next char
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # e.g, [4] maps to the 4th row, which has 65 floats
        # e.g, [58] maps to the 58th row, which has 65 floats
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # (65, 65) embedding table

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # B: batch, T: Time (context length), C: Channels (vocab_size)
        logits = self.token_embedding_table(idx) # (B,T,C) = (4,8,65)
        
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
            # get the predictions
            logits, loss = self(idx)

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