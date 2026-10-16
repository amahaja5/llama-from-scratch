"""Reproducible, small-scale training; run python train.py --help."""
import argparse
from pathlib import Path
import torch
from llama import Llama, sample, validate_optimizer


def get_batch(data, split, config, device='cpu', generator=None):
    n = len(data)
    bounds = {'train': (0, int(.8*n)), 'val': (int(.8*n), int(.9*n)), 'test': (int(.9*n), n)}
    start, end = bounds[split]
    chunk = data[start:end]
    context = config['context_window']
    if len(chunk) <= context:
        raise ValueError(f'{split} split must contain more than context_window tokens')
    indices = torch.randint(len(chunk)-context, (config['batch_size'],), generator=generator)
    x = torch.stack([chunk[i:i+context] for i in indices]).to(device)
    y = torch.stack([chunk[i+1:i+context+1] for i in indices]).to(device)
    return x, y


@torch.inference_mode()
def evaluate(model, data, batches=10):
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(2024)
    device = next(model.parameters()).device
    try:
        return {split: sum(model(*get_batch(data, split, model.config, device, generator))[1].item()
                           for _ in range(batches))/batches for split in ('train', 'val', 'test')}
    finally:
        model.train(was_training)


def save_checkpoint(path, model, optimizer, scheduler, step, vocab, seed, device):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict() if scheduler else None,
             'config': model.config, 'step': step, 'vocab': vocab, 'seed': seed,
             'rng_state': torch.get_rng_state(), 'device_type': torch.device(device).type}
    if state['device_type'] == 'cuda':
        state['device_rng'] = torch.cuda.get_rng_state_all()
    elif state['device_type'] == 'mps':
        state['device_rng'] = torch.mps.get_rng_state()
    torch.save(state, path)


def run_training(data, config, steps, checkpoint, vocab, seed=1337, device='cpu', resume=None, cosine=False):
    if steps < 1:
        raise ValueError('steps must be positive')
    torch.manual_seed(seed)
    saved = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if saved:
        if saved['vocab'] != vocab:
            raise ValueError('checkpoint vocabulary differs from dataset')
        if saved['device_type'] != torch.device(device).type:
            raise ValueError('resume on the same device type to restore RNG state')
        config = saved['config']
        seed = saved['seed']
    model = Llama(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(.9, .95))
    start = saved['step'] if saved else 0
    use_cosine = saved['scheduler'] is not None if saved else cosine
    total = saved['scheduler']['T_max'] if saved and use_cosine else steps
    if use_cosine and start + steps > total:
        raise ValueError('additional steps exceed the saved cosine schedule budget')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total, eta_min=1e-5) if use_cosine else None
    if saved:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if scheduler:
            scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['rng_state'])
        if saved['device_type'] == 'cuda':
            torch.cuda.set_rng_state_all(saved['device_rng'])
        elif saved['device_type'] == 'mps':
            torch.mps.set_rng_state(saved['device_rng'])
    validate_optimizer(model, optimizer)
    model.train()
    for step in range(start, start + steps):
        x, y = get_batch(data, 'train', config, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        if (step + 1) % 100 == 0:
            save_checkpoint(checkpoint, model, optimizer, scheduler, step + 1, vocab, seed, device)
    save_checkpoint(checkpoint, model, optimizer, scheduler, start + steps, vocab, seed, device)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('input.txt'))
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default='checkpoints/model.pt')
    parser.add_argument('--resume')
    parser.add_argument('--cosine', action='store_true')
    args = parser.parse_args()
    text = args.data.read_text()
    vocab = sorted(set(text))
    lookup = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([lookup[ch] for ch in text], dtype=torch.long)
    config = dict(vocab_size=len(vocab), context_window=16, d_model=128,
                  n_heads=8, n_layers=4, batch_size=32, dropout=.1)
    model = run_training(data, config, args.steps, args.checkpoint, vocab,
                         args.seed, args.device, args.resume, args.cosine)
    print('parameters:', sum(p.numel() for p in model.parameters()))
    print('cross-entropy:', evaluate(model, data))
    print('sample:', ''.join(vocab[i] for i in sample(model, torch.zeros(1, 1, dtype=torch.long), 100)[0].tolist()))
    print('checkpoint:', args.checkpoint)


if __name__ == '__main__':
    main()
