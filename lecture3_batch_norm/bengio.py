import pickle
import random
import string
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

batch_size= 32
block_size = 3
embed_size = 10
hidden_layer_neurons = 200

# length of names = 32033
def get_names() -> List[str]:
    names = []
    with open('../names.txt') as f:
        # names = list(map(lambda x: x[:-1], (next(f) for _ in range(50))))
        names = f.read().splitlines()
    return names

# building stoi
def get_stoi() -> Dict[str, int]:
    stoi = {c: i+1 for i, c in enumerate(string.ascii_lowercase)}
    stoi['.'] = 0
    return stoi


# bulding itos
def get_itos() -> List[str]:
    itos = ['.'] + list(string.ascii_lowercase)
    return itos

stoi = get_stoi()
itos = get_itos()
# preparing dataset
def get_xs_ys_from_name(name: str, block_size: int) -> Tuple[List[List[int]], List[int]]:

    xs, ys = [], []
    x = [0] * block_size
    for i in range(len(name)):
        c1 = name[i]
        c2 = name[i + 1] if i < len(name) - 1 else '.'
        first = stoi[c1]
        second = stoi[c2]

        x.append(first)
        x = x[1:]

        xs.append(x.copy())
        ys.append(second)

    return xs, ys

# xs, ys = get_xs_ys_from_name('emma', 10)
def print_in_outs(xs: List[List[str]], ys: List[str]):
    name_blocks = [ ''.join(map(lambda i: itos[i], x)) for x in xs]
    outs = map(lambda i: itos[i], ys)
    for x, y in zip(name_blocks, outs):
        print(f'{x} -> {y}')

# converting name blocks to weight tensor
# 1 input: (assuming block size = 3)
# (block_size, embed_size) -> (1, block_size * embed_size)

# batch_size inputs:
# (batch_size, block_size, embed_size) -> (batch_size, block_size * embed_size)
def build_dataset(names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for name in names:
        x, y = get_xs_ys_from_name(name, block_size)
        xs.extend(x)
        ys.extend(y)
    return torch.tensor(xs), torch.tensor(ys)

names = get_names()
print("names retrieved")
random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))
Xtr, Ytr = build_dataset(names[:n1])
Xval, Yval = build_dataset(names[n1:n2])
Xtest, Ytest = build_dataset(names[n2:])


g = torch.Generator().manual_seed(2147483647)
def training() -> torch.Tensor:
    # setup parameters
    C = torch.randn((27, embed_size),                                 generator=g)
    W1 = torch.randn((block_size * embed_size, hidden_layer_neurons), generator=g) * (5/3)/(block_size * embed_size)**0.5
    b1 = torch.randn((hidden_layer_neurons),                          generator=g) * 0.01
    W2 = torch.randn((hidden_layer_neurons, 27),                      generator=g) * 0.01
    b2 = torch.randn((27),                                            generator=g) * 0
    bngain = torch.ones((1, hidden_layer_neurons))
    bnbias = torch.zeros((1, hidden_layer_neurons))

    bnmean_running = torch.ones((1, hidden_layer_neurons))
    bnstd_running = torch.zeros((1, hidden_layer_neurons))

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True
    
    num_of_ws = sum(p.nelement() for p in parameters) 
    print(f'{num_of_ws=}')

    for i in range(200000):
        # pick random batch_size inputs from Xtr
        ix = torch.randint(0, len(Xtr), (batch_size,), generator=g)
        embed = C[Xtr[ix]]

        # forward
        hpreact = (embed.view(-1, block_size * embed_size) @ W1 + b1)

        # batch norm
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = (hpreact - bnmeani) / bnstdi
        hpreact = bngain * hpreact + bnbias
        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

        h = hpreact.tanh()
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])

        # backward
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

    parameters += [bnmean_running, bnstd_running]
    return parameters

@torch.no_grad()
def inference(dataset: torch.Tensor, labels: torch.Tensor, bnmean: torch.Tensor, bnstd: torch.Tensor, bngain: torch.Tensor, bnbias: torch.Tensor):
    emb = C[dataset] # (32, 3, 2)
    hpreact = emb.view(-1, block_size * embed_size) @ W1 + b1

    # batch norm
    hpreact = (hpreact - bnmean) / bnstd
    hpreact = bngain * hpreact + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2 # (32, 27)
    loss = F.cross_entropy(logits, labels)
    return loss

C, W1, b1, W2, b2, bngain, bnbias, bnmean_running, bnbias_running = training()
training_loss = inference(Xtr, Ytr, bnmean_running, bnbias_running, bngain, bnbias)
validation_loss = inference(Xval, Yval, bnmean_running, bnbias_running, bngain, bnbias)
print(f'{training_loss=}')
print(f'{validation_loss=}')


@torch.no_grad()
def generate_names():
    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor(context)] # (32, 3, 2)
            h = torch.tanh(emb.view(-1, block_size * embed_size) @ W1 + b1) # (32, 100)
            logits = h @ W2 + b2 # (32, 27)
            prob = F.softmax(logits, dim=1)
            idx = torch.multinomial(prob, num_samples=1, generator=g).item()
            context.append(idx)
            context = context[1:]
            out.append(itos[idx])
            if idx == 0:
                break
        print(''.join(out))


