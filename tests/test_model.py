import ast
import json
from pathlib import Path
import pytest
import torch
from llama import (Llama, LlamaBlock, RMSNorm, RoPEMaskedAttentionHead,
                   get_rotary_matrix, sample, show_grads, validate_optimizer)
from train import run_training, evaluate


def config(**overrides):
    return dict(dict(vocab_size=7, context_window=8, d_model=16, n_heads=2,
                     n_layers=2, batch_size=2, dropout=.3), **overrides)


def test_norm_is_token_local_and_finite():
    norm = RMSNorm(16)
    x = torch.randn(2, 8, 16)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(norm(x), expected)
    changed = x.clone()
    changed[:, 3:] *= 100
    torch.testing.assert_close(norm(x)[:, :3], norm(changed)[:, :3])
    assert torch.isfinite(norm(torch.zeros_like(x))).all()
    assert norm.scale.shape == (16,)


def test_model_causality_and_eval_determinism():
    model = Llama(config()).eval()
    x = torch.randint(7, (2, 8))
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 7
    with torch.no_grad():
        original = model(x)
        torch.testing.assert_close(original, model(x), rtol=0, atol=0)
        torch.testing.assert_close(original[:, :4], model(changed)[:, :4])
        torch.testing.assert_close(original[:, :4], model(x[:, :4]))


def test_attention_weights_match_operation():
    head = RoPEMaskedAttentionHead(config()).eval()
    x = torch.randn(2, 8, 16)
    out, weights = head(x, True)
    assert torch.count_nonzero(weights.triu(1)) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 8))
    torch.testing.assert_close(out, weights @ head.w_v(x))
    assert out.shape == (2, 8, 8)
    assert 'rope_cos' in dict(head.named_buffers())


def test_rope_reference_and_dtype_transfer():
    head = RoPEMaskedAttentionHead(config()).double().eval()
    x = torch.randn(2, 8, 8, dtype=torch.float64)
    R = get_rotary_matrix(8, 8).double()
    expected = torch.einsum('tij,btj->bti', R, x)
    torch.testing.assert_close(head.rotate(x), expected)
    assert head.rope_cos.dtype == torch.float64
    assert torch.isfinite(head(torch.randn(2, 8, 16, dtype=torch.float64))).all()
    with pytest.raises(ValueError):
        RoPEMaskedAttentionHead(config(d_model=14))


def test_residual_identity_when_branches_zero():
    block = LlamaBlock(config()).eval()
    with torch.no_grad():
        for submodule in (block.attention, block.feedforward):
            for p in submodule.parameters():
                p.zero_()
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(block(x), x)


def test_optimizer_and_actual_gradients():
    model = Llama(config())
    other = Llama(config())
    with pytest.raises(ValueError, match='optimizer'):
        validate_optimizer(model, torch.optim.Adam(other.parameters()))
    assert all(value is None for value in show_grads(model).values())
    x = torch.randint(7, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for name, p in model.named_parameters():
        assert show_grads(model)[name] == pytest.approx((p.grad.abs() <= .01).float().mean().item() * 100)


def test_sampling_restores_mode_and_disables_gradients():
    model = Llama(config()).train()
    observed = []
    hook = model.register_forward_pre_hook(lambda m, args: observed.append((m.training, torch.is_grad_enabled())))
    result = sample(model, torch.zeros(2, 1, dtype=torch.long), 10)
    hook.remove()
    assert result.shape == (2, 11)
    assert model.training
    assert observed == [(False, False)] * 10


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    data = torch.arange(1000) % 7
    cfg = config(dropout=.1)
    vocab = list('abcdefg')
    whole = run_training(data, cfg, 4, tmp_path/'whole.pt', vocab)
    run_training(data, cfg, 2, tmp_path/'part.pt', vocab)
    resumed = run_training(data, cfg, 2, tmp_path/'resumed.pt', vocab, resume=tmp_path/'part.pt')
    for key, tensor in whole.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
    assert evaluate(whole, data, 2) == evaluate(resumed, data, 2)


def test_notebook_final_definitions_match_module():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root/'llama.ipynb').read_text())
    definitions = {}
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.ClassDef):
                    definitions[node.name] = ast.dump(node)
    for node in ast.parse((root/'llama.py').read_text()).body:
        if isinstance(node, ast.ClassDef):
            assert definitions[node.name] == ast.dump(node)


def test_cosine_experiment_updates_its_model(tmp_path):
    data = torch.arange(1000) % 7
    cfg = config()
    torch.manual_seed(1337)
    initial = Llama(cfg)
    trained = run_training(data, cfg, 2, tmp_path/'cosine.pt', list('abcdefg'), cosine=True)
    assert any(not torch.equal(p, q) for p, q in zip(initial.parameters(), trained.parameters()))
    saved = torch.load(tmp_path/'cosine.pt', weights_only=True)
    assert saved['scheduler']['last_epoch'] == 2
    with pytest.raises(ValueError, match='schedule budget'):
        run_training(data, cfg, 1, tmp_path/'next.pt', list('abcdefg'), resume=tmp_path/'cosine.pt')
