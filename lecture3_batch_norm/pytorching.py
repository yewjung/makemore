import random
import string
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures



g = torch.Generator().manual_seed(2147483647)


class Linear:

    def __init__(self, fan_in, fan_out, bias=True) -> None:
        self.weights = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
        pass

    def __call__(self, x) -> torch.Tensor:
        self.out = x @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        if self.bias is not None:
            return [self.weights, self.bias]
        
        return [self.weights]

class Tanh:

    def __call__(self, x) -> torch.Tensor:
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class BatchNormal1D:

    def __init__(self, dim, eps=1e-5, momentum=0.1) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
n_embed = 10
n_hidden = 100
block_size = 3

def get_names() -> List[str]:
    names = []
    with open('../names.txt') as f:
        # names = list(map(lambda x: x[:-1], (next(f) for _ in range(50))))
        names = f.read().splitlines()
    return names

vocab_size = 27

C = torch.randn((vocab_size, n_embed), generator=g)
layers: List[torch.Tensor] = [
    Linear(n_embed * block_size, n_hidden, bias=False), BatchNormal1D(n_hidden), Tanh(),
    Linear(            n_hidden, n_hidden, bias=False), BatchNormal1D(n_hidden), Tanh(),
    Linear(            n_hidden, n_hidden, bias=False), BatchNormal1D(n_hidden), Tanh(),
    Linear(            n_hidden, n_hidden, bias=False), BatchNormal1D(n_hidden), Tanh(),
    Linear(            n_hidden, n_hidden, bias=False), BatchNormal1D(n_hidden), Tanh(),
    Linear(          n_hidden, vocab_size, bias=False), BatchNormal1D(vocab_size)
]

with torch.no_grad():
    layers[-1].gamma *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weights *= 1.0

parameters: List[torch.Tensor] = [C] + [p for layer in layers for p in layer.parameters()]
no_of_params = sum(p.nelement() for p in parameters)
print(f"{no_of_params=}")
for p in parameters:
    p.requires_grad = True
    
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

# training
max_steps = 200000
batch_size = 32
lossi = []
ud = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g) # indices.shape = [32]
    Xb, Yb = Xtr[ix], Ytr[ix] # Xb.shape = [32, 3]
    emb = C[Xb] # C.shape = [27, 10], emb.shape = [32, 3, 10]
    x = emb.view(emb.shape[0], -1)
    # forward
    for layer in layers:
        x = layer(x)
    
    loss = F.cross_entropy(x, Yb)
    # backward
    for layer in layers:
        layer.out.retain_grad() 
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    
    # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

    if i >= 1000:
        break # AFTER_DEBUG: would take out obviously to run full optimization

plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends)
plt.title('activation distribution')

# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends)
plt.title('gradient distribution')

# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution')

plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
plt.legend(legends)

plt.show()

