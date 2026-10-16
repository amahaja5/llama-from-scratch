"""Small character-level, Llama-inspired decoder used by the tutorial."""
import torch
from torch import nn
from torch.nn import functional as F


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


class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])])
        self.linear = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.dropout = nn.Dropout(config.get('dropout', .1))

    def forward(self, x):
        return self.dropout(self.linear(torch.cat([h(x) for h in self.heads], dim=-1)))


class SwiGLU(nn.Module):
    def __init__(self, size, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(8 * size / 3)
        self.linear_gate = nn.Linear(size, hidden_dim, bias=False)
        self.linear = nn.Linear(size, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, size, bias=False)

    def forward(self, x):
        return self.output(F.silu(self.linear_gate(x)) * self.linear(x))


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


@torch.inference_mode()
def sample(model, idx, max_new_tokens=30):
    was_training = model.training
    model.eval()
    try:
        idx = idx.to(next(model.parameters()).device)
        for _ in range(max_new_tokens):
            logits = model(idx[:, -model.config['context_window']:])
            next_idx = torch.multinomial(logits[:, -1].softmax(dim=-1), num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
    finally:
        model.train(was_training)


def show_grads(model, tol=1e-2):
    """Percent of near-zero gradients; None means no gradient was computed."""
    return {name: None if p.grad is None else (p.grad.abs() <= tol).float().mean().item() * 100
            for name, p in model.named_parameters() if p.requires_grad}


def validate_optimizer(model, optimizer):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group['params']}
    if expected != actual:
        raise ValueError('optimizer must own the trainable parameters of the model being trained')
