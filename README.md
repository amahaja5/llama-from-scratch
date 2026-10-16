# Running the corrected tutorial

This is a small **Llama-inspired character model**, not a reproduction of pretrained Llama. The original results were invalidated by sequence-wide normalization leaking future-token information. Old notebook outputs and loss claims have been cleared; no new quality benchmark is claimed.

Use Python 3.11:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python train.py --steps 100 --checkpoint checkpoints/smoke.pt
jupyter lab llama.ipynb
```

The notebook preserves the intentionally broken softmax and unmasked-attention baselines. Its final model uses per-token RMSNorm, independent pre-norm residual branches, even-sized attention heads of width `d_model / n_heads`, buffered rotary sine/cosine tables, a SwiGLU feed-forward layer and final normalization. The reference dense rotation matrix is retained only for teaching checks. `llama.py` exposes the corrected model for tests and `train.py` provides seeded training, validation, and checkpoints including optimizer, configuration, vocabulary and RNG states. Checkpoints use a new architecture and are not compatible with the original notebook model.

```sh
python train.py --steps 1000 --checkpoint checkpoints/model.pt
python train.py --steps 100 --resume checkpoints/model.pt --checkpoint checkpoints/resumed.pt
```

Add `--cosine` for a cosine learning-rate schedule. Checkpoints are saved every 100 steps and at completion. On resume, `--steps` means additional steps. The saved cosine schedule retains its original total budget; extending a completed cosine run is rejected. Seeds reproduce runs on the same supported environment; results may differ across devices or library versions. The default device is CPU; use `--device mps` or `--device cuda` when available. The notebook remains a CPU walkthrough. No full training rerun has been published with this correction.

## Corrections to the original experiments

- RMSNorm now reduces only over the embedding axis, includes epsilon, and has one scale per feature. It no longer leaks future-token statistics.
- Functional attention dropout is disabled in evaluation; generation uses inference mode and restores the caller's training mode.
- Diagnostic attention weights apply the same causal mask as the attention operation.
- The cosine optimizer now owns the model actually being trained. Training rejects mismatched optimizers.
- Gradient inspection reads `.grad`, including missing gradients explicitly.
- Focused tests cover causality, normalization, rotary embeddings, gradients, optimizer ownership, sampling, and checkpoint continuation.


---

# Llama from scratch

I want to provide some tips from my experience implementing a paper. I'm going to cover my tips so far from implementing a dramatically scaled-down version of [Llama](https://arxiv.org/pdf/2302.13971.pdf) for training [TinyShakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). This post is heavily inspired by Karpathy's [Makemore series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), which I highly recommend.

I'm only going to *loosely* follow the layout of their paper; while the formatting and order of sections makes sense for publication, we're going to be implementing the paper. I'll also be skipping over some of the more obvious steps, like setting up a virtual environment and installing dependencies.

The saved results from the original implementation have been removed. Re-run the corrected notebook to obtain new losses and samples.

# Takeaways

## Always work iteratively: start small, stay certain, and build up.
My approach for implementing papers is:
1. Make all of the helper functions required to test your model quantitatively (data splits, training, plotting the loss).
1. Before you even look at the paper, pick a small, simple, and fast model that you've done in the past. Then make a helper function to evaluate the model qualitatively.
1. Start by picking apart different components of the paper, and then implementing them one-by-one, training and evaluating as you go.

## Make sure your layers do what you think.
1. Use `.shape` religiously. `assert` and `plt.imshow` are also your friends.
1. Work out the results without matrix multiplication first, and then use the `torch` functions to make it efficient after.
1. Have a test to see that your layer is right. For example, the RoPE embeddings have a specific property that you can test for. For the Transformer, you can test that the attention is working by looking at the attention map.
1. Test your layers on various batch, sequence, and embedding sizes. Even if it works for one size, it might not work for others, which will cause problems at inference time.

## About Llama

Llama is a transformer-based model for language modeling. Meta AI [open-sourced](https://github.com/facebookresearch/llama) Llama this summer, and it's gained a lot of attention (pun intended). When you're reading the introduction, they clearly indicate their goal: make a model that's cheaper for running inference, rather than optimizing training costs.

At this point, we'll just load our libraries and get started.

```python
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from llama import sample, validate_optimizer
torch.manual_seed(1337)
np.random.seed(1337)
```

## Setting up our dataset
While in Llama they train on 1.4T tokens, our dataset TinyShakespeare, a sample of Shakespeare's works, is about 1M characters.

```python
lines = open('./input.txt', 'r').read()

vocab = sorted(list(set(lines)))
itos = {i:ch for i, ch in enumerate(vocab)}
stoi = {ch:i for i, ch in enumerate(vocab)}

print(lines[:30])
```

They use the [SentencePiece](https://github.com/google/sentencepiece) byte-pair encoding tokenizer, but we're going to just use a simple character-level tokenizer.

```python
# simple tokenization by characters
def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return ''.join([itos[i] for i in l])

print('vocab size:', len(vocab))
decode(encode("hello"))
```

Since our dataset is small enough, we don't need to worry about how we store it in memory etc.

First tip: I'm creating a `config` object that stores some basic model params. It makes our code way more readable and removes constants and magic numbers from the code. I'm not going to use types, as  I want to keep things flexible for now and be able to add more parameters later on.

```python
MASTER_CONFIG = {
    "vocab_size": len(vocab),
}
```

```python
dataset = torch.tensor(encode(lines), dtype=torch.int8)
dataset.shape
```

Let's create a method to generate our training data and labels for batches. We'll use the same method for validation and test data. Note that I like to test my functions in the same block that I define them, just to make sure they work as expected before moving on.

```python
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    batch_data = train
    if split == 'val':
        batch_data = val

    if split == 'test':
        batch_data = test

    # pick random starting points
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y

MASTER_CONFIG.update({
    'batch_size': 8,
    'context_window': 16
})

xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

[(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]
```

What's interesting about implementing papers is that there are two aspects to the model *working*: compilation (do your tensors all match up from layer to layer), and training (does the loss go down). Figuring out how to ensure that each of your compoenents is working is key to developing your model in a predictable, engineering-minded way.

That's why we're also going to define the method for how we're going to evaluate the model. We want to do this before we even define the model, because we want to be able to use it to evaluate the model as we're training it.

```python
@torch.no_grad()  # don't compute gradients for this function
def evaluate_loss(model, config=MASTER_CONFIG):
    out = {}
    was_training = model.training
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train(was_training)
    return out
```

## Setting up a working base model

Here's a basic feed-forward neural network with embeddings. It's the base model we're going to start with, and then swap out parts of it as we go along until we eventually end up with the model as described in Llama.

```python
class SimpleBrokenModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        a = self.linear(x)
        logits = F.softmax(a, dim=-1)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

MASTER_CONFIG.update({
    'd_model': 128,
})
model = SimpleBrokenModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
```

It's at this point that we have to start worrying about the shape of our tensors and making indices match. Check out this line of our model definition:

```python
loss = F.cross_entropy(logits.view(-1, config['vocab_size']), targets.view(-1))
```

We have to reshape the `logits` and `targets` tensors so that their dimensions match when we compare. We do this with the `view` method. The `-1` argument means "infer this dimension from the others". So, in this case, we're saying "reshape `logits` and `targets` to have the same number of rows, and however many columns are needed to make that happen". This is a common pattern when you're working with batches of data.

Alright, let's train our `SimpleBrokenModel` to make sure gradients flow. After we confirm that, we can swap out parts of it to match Llama, train again, and track our progress. It's at this point that I start keeping a *log* of my training runs, so that I can easily just go back to a previous run in the event that I mess something up.

```python
MASTER_CONFIG.update({
    'epochs': 1000,
    'log_interval': 10,
    'batch_size': 32,
})
model = SimpleBrokenModel(MASTER_CONFIG)

optimizer = torch.optim.Adam(
    model.parameters(),
)

def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    validate_optimizer(model, optimizer)
    model.train()
    losses = []
    start_time = time.time()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()

        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])
        logits, loss = model(xs, targets=ys)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model, config)
            losses += [x]
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
            start_time = time.time()

            if scheduler:
                print("lr: ", scheduler.get_last_lr())

    print("validation loss: ", losses[-1]['val'])
    return pd.DataFrame(losses).plot()

train(model, optimizer)
```

The intentionally broken baseline applies softmax before cross-entropy. Inspect its learning curve, then compare it with the corrected raw-logit baseline. How can we interpret cross-entropy intuitively?

Cross-entropy in this context is referring to how likely we are to pick the wrong word. So here,

$$
H(T, q) = - \sum_{i = 1}^{N} \frac{1}{N} \log q(x_i)
$$

where $q(x_i)$ is the probability of picking the right word, as estimated by the model. If $q(x_i)$ is close to 1, then $\log q$ is close to 0; similarly, if $q$ is small, then $\log q$ is a large negative number, so $-\log q$ will be a large positive number. Now to build the intuition: to start, $-\log q = 4.17$, so $q = 0.015$, or around $\frac{1}{64.715}$. Recall that the vocabulary size $|V| = 65$, so what we're basically saying here is that the model is as good at choosing the next letter as randomly picking from our vocabulary. After training, $-\log q = 3.93$, so we're now basically choosing between 50 letters. This is a very small improvement, so something is probably wrong.

To get an intuition for how the loss relates to the model's performance, think about the model choosing among $\tilde V$ tokens; when $\tilde V$ is small, the model is more likely to guess right. In addition, we know $\max \tilde V = V$, which can help us understand if our model is learning at all.

$$\tilde V = \exp(L)$$

Let's try to debug what's going on. Notice that in our model we're using a softmax layer on our logits, which is a function that takes a vector of numbers and squashes them into a probability distribution. But for using the built in `F.cross_entropy` function, we need to pass in the [unnormalized logits directly](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html). So let's remove that from our model and try again.

```python
class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        logits = self.linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

model = SimpleModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)
```

For a loss L, exp(L) gives perplexity. Compare fresh measurements with the 65-character vocabulary, then inspect generated text.

```python
def generate(model, config=MASTER_CONFIG, max_new_tokens=30):
    idx = sample(model, torch.zeros(5, 1, dtype=torch.long), max_new_tokens)
    return [decode(row) for row in idx.tolist()]

generate(model)
```

It's not half bad, but also not half good. But now we have a working model that is training to a validation loss. So here we'll iterate on our model to make it closer to Llama.

## Llama specifics
Llama describes three architectural modifications to the original Transformer:
1. RMSNorm for pre-normalization
1. Rotary embeddings
1. SwiGLU activation function

We're going to add each one, one at a time to our base model, and iterate.

The original Transformer uses LayerNorm. Llama uses RMSNorm: normalize each token by the root mean square of its features, without subtracting their mean, then apply a learned feature scale. Pre-normalization applies this inside each residual branch, preserving the original residual stream.

```python
class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8):
        super().__init__()
        # Accept the tutorial's old (context_window, d_model) spelling.
        width = layer_shape if isinstance(layer_shape, int) else layer_shape[-1]
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.scale

config = {
    'batch_size': 5,
    'context_window': 11,
    'd_model': 13,
}
batch = torch.randn((config['batch_size'], config['context_window'], config['d_model']))
m = RMSNorm((config['context_window'], config['d_model']))
g = m(batch)
print(g.shape)
```

Check the per-token RMS formula, finite outputs for zero inputs, and independence from future-token values.

```python
# Token-local normalization: future tokens cannot change an earlier token.
expected = batch * torch.rsqrt(batch.square().mean(-1, keepdim=True) + m.eps)
assert torch.allclose(g, expected)
changed = batch.clone()
changed[:, 1:] *= 10
assert torch.allclose(m(batch)[:, 0], m(changed)[:, 0])
assert torch.isfinite(m(torch.zeros_like(batch))).all()
```

Alright, so that's RMSNorm, and it seems like it's working. Again, let's test it out.

```python
class SimpleModel_RMS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        x = self.rms(x) # rms pre-normalization
        logits = self.linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

model = SimpleModel_RMS(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)
```

Compare the new validation curve against the baseline.

### Rotary Embeddings
[RoPE](https://arxiv.org/pdf/2104.09864.pdf) is a kind of positional encoding for transformers. In Attention is All You Need, the authors propose two kinds of positional encodings, learned and fixed. In RoPE, the authors propose embedding the position of a token in a sequence by rotating the embedding, with a different rotation at each position.

```python
def get_rotary_matrix(context_window, embedding_dim):
    """Dense reference implementation retained for the tutorial's algebra checks."""
    if embedding_dim % 2:
        raise ValueError('RoPE requires an even dimension')
    angles = torch.outer(torch.arange(context_window).float(),
                         10000. ** (-torch.arange(0, embedding_dim, 2).float() / embedding_dim))
    R = torch.zeros(context_window, embedding_dim, embedding_dim)
    i = torch.arange(0, embedding_dim, 2)
    R[:, i, i] = angles.cos()
    R[:, i, i + 1] = -angles.sin()
    R[:, i + 1, i] = angles.sin()
    R[:, i + 1, i + 1] = angles.cos()
    return R
```

```python
K = 3
config = {
    'batch_size': 10,
    'd_model': 32,
    'n_heads': 8,
    'context_window': K**2,
}
batch = torch.randn(1, config['context_window'], config['d_model'])
R = get_rotary_matrix(config['context_window'], config['d_model'])
fig, ax = plt.subplots(K, K, figsize=(K * 3, K * 4))

for i in range(K):
    for j in range(K):
        ax[i, j].imshow(R[i * K + j, :, :].detach().numpy())
        ax[i, j].set_title(f'rotation at {i * K + j}')
```

Let's make sure these work. They should exhibit the quality that
$$
q_m^T k_n = (R^d_{\Theta, m}W_q x_m)^T (R^d_{\Theta, n} W_k x_n) = x^T W_q R^d_{\Theta, n-m} W_k x_n.
$$

```python
config = {
    'd_model': 128,
    'context_window': 16,
}

R = get_rotary_matrix(config['context_window'], config['d_model'])
x = torch.randn(config['d_model'])
y = torch.randn(config['d_model'])

m = 3
n = 13

x_m = R[m,:,:] @ x
x_n = R[n,:,:] @ y

assert torch.isclose(x_m @ x_n, x @ R[n-m,:,:] @ y)
```

So the RoPE rotations work as expected.

```python
config = {
    'batch_size': 10,
    'd_model': 512,
    'n_heads': 8,
    'context_window': 16,
}

class RoPEAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)

        self.register_buffer("R", get_rotary_matrix(config['context_window'], config['d_model']))

    def get_rotary_matrix(context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = - np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R

    def forward(self, x, return_attn_weights=False):
        b,m,d = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)

        activations = F.scaled_dot_product_attention(
            q_rotated,k_rotated,v,dropout_p=.1 if self.training else 0.0
        )

        if return_attn_weights:
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations

layer = RoPEAttentionHead(config)
batch = torch.randn((config['batch_size'], config['context_window'], config['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)
```

> Tip here: know the difference between tensor dimensions at train time vs tensor dimensions at inference time.

Although at train time, you can expect your tensor dimensions to match your model parameters closely, eg `batch.shape = (config['batch_size'], config['context_window'], config['d_model'])`, at inference time, you may have to deal with a single example, eg `batch.shape = (1, 1, config['d_model'])`. For this reason, you need to make sure that when you're indexing in the `forward` pass, you're indexing using shapes derived from the input, not necessarily the model parameters.

Let's make sure it does what we think it does. For this layer, we're going to want to test three things:
1. that it rotates embeddings the way we think it does
2. that the attention mask used for causal attention is working properly.

```python
x = torch.randn((config['batch_size'], config['context_window'], config['d_model']))

q = layer.w_q(x)
k = layer.w_k(x)
v = layer.w_v(x)

q_rotated = torch.zeros_like(x)
k_rotated = torch.zeros_like(x)
v_rotated = torch.zeros_like(x)

for position in range(config['context_window']):
    q_rotated[:,position,:] = torch.matmul(q[:,position,:], layer.R[position,:,:])
    k_rotated[:,position,:] = torch.matmul(k[:,position,:], layer.R[position,:,:])
    v_rotated[:,position,:] = torch.matmul(v[:,position,:], layer.R[position,:,:])

q_vectorized = (torch.bmm(q.transpose(0,1), layer.R)).transpose(0,1)
k_rotated = (torch.bmm(k.transpose(0,1), layer.R)).transpose(0,1)
v_out = (torch.bmm(v.transpose(0,1), layer.R)).transpose(0,1)

assert torch.allclose(q.transpose(0,1)[0], q[:,0,:])
assert torch.allclose(q.transpose(0,1)[0] @ layer.R[0], q[:,0,:] @ layer.R[0])
assert torch.allclose(q_rotated, q_vectorized)
```

```python
config = {
    'batch_size': 1,
    'd_model': 2,
    'n_heads': 2,
    'context_window': 3,
}

layer = RoPEAttentionHead(config)
batch = torch.ones((config['batch_size'], config['context_window'], config['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)

m = 0
x_q = batch[0, m]
q = layer.R[m,:,:] @ layer.w_q(x_q)

assert torch.allclose(layer.w_q(x_q), layer.w_q.weight @ x_q)
assert torch.allclose(q, layer.R[m, :, :] @ layer.w_q.weight @ x_q)

n = 2
x_k = batch[0, n]
k = layer.R[n,:,:] @ layer.w_k(x_k)

assert torch.allclose(layer.w_k(x_k), layer.w_k.weight @ x_k)
assert torch.allclose(k, layer.R[n, :, :] @ layer.w_k.weight @ x_k)

assert q.T @ k == q @ k # transpose is redundant
assert torch.allclose(q @ k, x_k.T @ layer.w_k.weight.T @ layer.R[n, :, :].T @ layer.R[m, :, :] @ layer.w_q.weight @ x_q)
assert torch.allclose(q @ k, x_k.T @ layer.w_k.weight.T @ layer.R[n-m, :, :].T @ layer.w_q.weight @ x_q)
```

Let's setup a multi-headed attention layer for this singular attention head and see what happens when we train.

```python
# definitely there's an optimization we could make where we cache the rotation matrices, but skip.
class RoPEMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([
            RoPEAttentionHead(config) for _ in range(config['n_heads'])
        ])
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x

MASTER_CONFIG.update({
    'n_heads': 8,
})
layer = RoPEMultiheadAttention(MASTER_CONFIG)
batch = torch.ones((MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']))
output = layer(batch)
output.shape
```

```python
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMultiheadAttention(config)

        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
        )

        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)

        # one block of attention
        x = x + self.rope_attention(self.rms(x))

        x = x + self.linear(self.rms(x))

        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)
```

The unmasked model can read future tokens. A low validation loss here is not a valid measure of autoregressive performance. Compare it with generated samples.

```python
generate(model, config=MASTER_CONFIG)
```

Inspect the unmasked attention to see why teacher-forced evaluation can disagree with generation.

```python
MASTER_CONFIG.update({
    'n_heads': 8,
})
layer = RoPEAttentionHead(MASTER_CONFIG)
batch = torch.ones((MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)

plt.imshow(attn_weights[0].detach().numpy(), interpolation='nearest')
plt.colorbar()
```

So here we see that all the attention is lit up, meaning that characters in any position are attending to characters in any other position. What's bad about this? We are trying to predict the next token *solely* on the tokens that came before it, but here we're seeing that the model is attending to tokens that come after it. In other words, the model is cheating, or leaking information from the future. This is a problem, and it's why we need to use a causal mask.

```python
config = {
    'batch_size': 10,
    'd_model': 512,
    'n_heads': 8,
    'context_window': 16,
}

class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['d_model']
        heads = config['n_heads']
        if width % heads or (width // heads) % 2:
            raise ValueError('d_model must divide evenly into even-sized heads')
        self.head_dim = width // heads
        self.dropout_p = config.get('dropout', .1)
        self.w_q = nn.Linear(width, self.head_dim, bias=False)
        self.w_k = nn.Linear(width, self.head_dim, bias=False)
        self.w_v = nn.Linear(width, self.head_dim, bias=False)
        angles = torch.outer(torch.arange(config['context_window']).float(),
                             10000. ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('rope_cos', angles.cos())
        self.register_buffer('rope_sin', angles.sin())

    def rotate(self, x):
        length = x.shape[1]
        cos = self.rope_cos[:length].to(x.dtype)
        sin = self.rope_sin[:length].to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x, return_attn_weights=False):
        length = x.shape[1]
        if length > self.rope_cos.shape[0]:
            raise ValueError('sequence exceeds context_window')
        q, k = self.rotate(self.w_q(x)), self.rotate(self.w_k(x))
        v = self.w_v(x)
        activations = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        if return_attn_weights:
            scores = q @ k.transpose(-2, -1) / self.head_dim ** .5
            future = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
            weights = scores.masked_fill(future, float('-inf')).softmax(dim=-1)
            return activations, weights  # probabilities before attention dropout
        return activations

layer = RoPEMaskedAttentionHead(config)
batch = torch.randn((config['batch_size'], config['context_window'], config['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)
```

```python
layer = RoPEMaskedAttentionHead(MASTER_CONFIG)
batch = torch.ones((MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']))
output, attn_weights = layer(batch, return_attn_weights=True)

plt.imshow(attn_weights[0].detach().numpy())
plt.colorbar()
```

The diagnostic weights now use negative infinity above the diagonal, matching causal attention before dropout. Future-token weights must be exactly zero.

```python
# definitely there's an optimization we could make where we cache the rotation matrices, but skip.
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])])
        self.linear = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.dropout = nn.Dropout(config.get('dropout', .1))

    def forward(self, x):
        return self.dropout(self.linear(torch.cat([h(x) for h in self.heads], dim=-1)))

MASTER_CONFIG.update({
    'n_heads': 8,
})
layer = RoPEMaskedMultiheadAttention(MASTER_CONFIG)
batch = torch.ones((MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']))
output = layer(batch)
output.shape
```

```python
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMaskedMultiheadAttention(config)

        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
        )

        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)

        # one block of attention
        x = x + self.rope_attention(self.rms(x))

        x = x + self.linear(self.rms(x))

        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)
```

Train the corrected causal model longer and compare training and validation loss.

```python
MASTER_CONFIG.update({
    "epochs": 5000,
    "log_interval": 10,
})
train(model, optimizer)
```

### SwiGLU

As it says in the paper, "We replace the ReLU non-linearity by the SwiGLU activation function...we use a dimension of $\frac{2}{3} 4d$ isntead of $4d$ as in PaLM." SwiGLU is defined as:
$$
\text{SwiGLU}(x) = \text{Swish}_\beta (xW + b) \otimes (xV + c)
$$
where $\otimes$ is a component-wise product. The Swish function is defined as:
$$
\text{Swish}_\beta(x) = x \sigma(\beta x)
$$
The general Swish formulation permits a learnable $\beta$. Here we use SiLU ($\beta=1$), expand the two projections to roughly $8d/3$, and project their product back to model width.

```python
class SwiGLU(nn.Module):
    def __init__(self, size, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(8 * size / 3)
        self.linear_gate = nn.Linear(size, hidden_dim, bias=False)
        self.linear = nn.Linear(size, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, size, bias=False)

    def forward(self, x):
        return self.output(F.silu(self.linear_gate(x)) * self.linear(x))
```

```python
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMaskedMultiheadAttention(config)

        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)

        # one block of attention
        x = x + self.rope_attention(self.rms(x))

        x = x + self.linear(self.rms(x))

        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)
```

Now, let's add multiple layers of RopeAttention by creating blocks.

```python
# add RMSNorm and residual conncection
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config['d_model'])
        self.ffn_norm = RMSNorm(config['d_model'])
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = SwiGLU(config['d_model'])

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.ffn_norm(x))
```

```python
block = LlamaBlock(MASTER_CONFIG)
block(torch.randn(MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model']));
```

```python
from collections import OrderedDict

MASTER_CONFIG.update({
    'n_layers': 4,
})
class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.llama_blocks = nn.Sequential(*[LlamaBlock(config) for _ in range(config['n_layers'])])
        self.final_norm = RMSNorm(config['d_model'])
        self.output = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    def forward(self, idx, targets=None):
        logits = self.output(self.final_norm(self.llama_blocks(self.embeddings(idx))))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, self.config['vocab_size']), targets.reshape(-1))

llama = Llama(MASTER_CONFIG)
optimizer = torch.optim.Adam(llama.parameters())
train(llama, optimizer, config=MASTER_CONFIG)
```

Continue training, monitoring the gap between training and validation loss.

```python
MASTER_CONFIG.update({
    'epochs': 10000,
})
train(llama, optimizer, scheduler=None, config=MASTER_CONFIG)
```

Use fresh validation measurements after correcting token-local normalization. The old losses included future-token leakage and are not comparable.

```python
train(llama, optimizer, config=MASTER_CONFIG)
```

```python
print(generate(llama, MASTER_CONFIG, 500)[0])
```

Finally, evaluate multiple held-out test batches in evaluation mode with gradients disabled.

```python
llama.eval()
with torch.inference_mode():
    test_losses = []
    for _ in range(100):
        xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
        _, loss = llama(xs, ys)
        test_losses.append(loss.item())
print('mean held-out test loss:', np.mean(test_losses))
```

# Miscellaneous

## Check for Gradient Flows

Let's inspect the gradients, we want to see how they're flowing. If there are too many gradients where the value is close to 0, that's a problem.

```python
# print the percentage that are near 0
def show_grads(model, tol=1e-2):
    """Percent of near-zero gradients; None means no gradient was computed."""
    return {name: None if p.grad is None else (p.grad.abs() <= tol).float().mean().item() * 100
            for name, p in model.named_parameters() if p.requires_grad}

show_grads(llama)
```

Inspect actual `.grad` tensors after backpropagation. `None` means no gradient has been computed; a nonzero gradient alone does not prove correctness.

Compare a cosine schedule using a fresh model and an optimizer attached to that same model. The original experiment accidentally optimized a different model, so its conclusion about cosine scheduling was invalid. Use matched seeds and training budgets for a controlled comparison.

```python
MASTER_CONFIG.update({
    "epochs": 1000
})
llama_with_cosine = Llama(MASTER_CONFIG)
llama_optimizer = torch.optim.Adam(
    llama_with_cosine.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, MASTER_CONFIG["epochs"], eta_min=1e-5)
train(llama_with_cosine, llama_optimizer, scheduler=scheduler)
```

```python
show_grads(llama_with_cosine, 1e-5)
```

Re-run the corrected experiment before drawing conclusions about scheduling or gradient flow.


