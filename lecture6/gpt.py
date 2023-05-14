import torch
import torch.nn as nn
from torch.nn import functional as F
from time import time

block_size = 32 # context length
dropout = 0.2
n_embed = 64
n_head = 4
n_layer = 4
batch_size = 32
learning_rate = 1e-3
device = 'cpu'

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = ''.join(sorted(list(set(text))))
vocab_size = len(vocab)
itos = list(vocab)
stoi = {itos[i] : i for i in range(vocab_size)}

n = int(0.9*len(text))
test_text = text[:n]
val_text = text[n:]
train_data = torch.tensor([stoi[c] for c in test_text])
val_data = torch.tensor([stoi[c] for c in val_text])

class Head(nn.Module):

    def __init__(self, head_size) -> None:
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        q: torch.Tensor = self.query(x)
        k: torch.Tensor = self.key(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, -1)
        wei = self.dropout(wei)

        v: torch.Tensor = self.value(x)
        return wei @ v
    
class MultiHeadAttention(nn.Module):

    def __init__(self, head_count, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_count)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        # x = (4, 8, 64)
        out = torch.cat([head(x) for head in self.heads], dim=-1) # out = (4, 8, 64), each head(x) outputs (4, 8, 16)
        out = self.proj(out) # (4, 8, 64)
        return self.dropout(out)
    
class FeedForward(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        ])
    
    def forward(self, x: torch.Tensor):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_head, head_size) -> None:
        super().__init__()
        self.multihead = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)


    def forward(self, x: torch.Tensor):
        x = x + self.multihead(self.ln1(x))
        return x + self.ffw(self.ln2(x))
    
class BigramLanguageModel2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_head, n_embed // n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.ffw = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx: torch.Tensor, target=None):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.ln(x)
        logits: torch.Tensor = self.ffw(x) # (B, T, vocab_size)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, -1) # (B, T, vocab_size) -> (B*T, vocab_size)
            # target: (B, T) -> (B*T,)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_tokens=100):
        for _ in range(max_tokens):
            # idx: (B, T)
            logits, _  = self(idx[:, -block_size:]) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, 1, vocab_size)
            prob = F.softmax(logits, 1) # (B, vocab_size)
            idx_next = torch.multinomial(prob, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        for batch in idx:
            print(''.join(itos[n] for n in batch))

@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iter: int):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()
    return out

def training(model: nn.Module, steps: int, eval_iter: int):
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for i in range(steps):

        if i % eval_iter == 0 or i == steps - 1:
            losses = estimate_loss(model, eval_iter)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        # forward
        _, loss = model(xb, yb)

        # backward
        optim.zero_grad(set_to_none=True)
        loss.backward()

        # update
        optim.step()

# m = BigramLanguageModel2().to(device)
# training(m, 10000, 200)
# saving
# filename = './model.pt'
# torch.save(m, './model.pt')

# loading
m: BigramLanguageModel2 =torch.load('./model.pt')
m.eval()
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
m.generate(idx, 10000)