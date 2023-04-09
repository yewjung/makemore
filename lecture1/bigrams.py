import torch.nn.functional as F
import string
import torch
import matplotlib.pyplot as plt


stoi = {c: i+1 for i, c in enumerate(string.ascii_lowercase)}
stoi['.'] = 0
itos = ['.'] + list(string.ascii_lowercase)

def make_bigram():
    with open('names.txt') as f:
        names = f.read().splitlines()

    N = torch.zeros((27, 27), dtype=torch.int32)
    for name in names:
        tokens = ['.'] + list(name) + ['.']
        for first, second in zip(tokens, tokens[1:]):
            i = stoi[first]
            j = stoi[second]
            N[i, j] += 1

    N /= N.sum(1, keepdim=True)
    torch.multinomial(N, 1, replacement=True,generator=g)
    return N

def plot(N):
    plt.imshow(N)


    plt.figure(figsize=(16,16))
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')

def training(names):
    xs = []
    ys = []
    for name in names:
        tokens = ['.'] + list(name) + ['.']
        for first, second in zip(tokens, tokens[1:]):
            xs.append(stoi[first])
            ys.append(stoi[second])
            
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    xenc = F.one_hot(xs, num_classes=27).float()
    # Weight initialization
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)
    # training
    for i in range(300):

        # forward
        out = xenc @ W
        count = out.exp()
        probs = count / count.sum(1, keepdims=True)
        loss = -probs[torch.arange(xs.nelement()), ys].log().mean() + 0.01*(W ** 2).mean()

        # backward
        W.grad = None
        loss.backward()

        # update
        W.data += -50 * W.grad
        if i % 100 == 0:
            print(loss.item())

def inference(W):
    g = torch.Generator().manual_seed(2147483647)

    for i in range(10):
    
        out = []
        ix = 0
        while True:
            
            # ----------
            # BEFORE:
            #p = P[ix]
            # ----------
            # NOW:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W # predict log-counts
            counts = logits.exp() # counts, equivalent to N
            p = counts / counts.sum(1, keepdims=True) # probabilities for next character
            # ----------
            
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))





