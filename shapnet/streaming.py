"""Bounded activation memory with exact gradients for a declared block operator."""
import torch

from .objectives import objective


def block_ranges(n, block_size):
    if n < 1 or block_size < 1:
        raise ValueError("Nonempty bag and positive block_size required")
    return [(start, min(start + block_size, n)) for start in range(0, n, block_size)]


def full_context(features, device):
    # Full bag remains in CPU RAM; only this vector and a block go to the accelerator.
    return features.mean(dim=0).to(device)


@torch.no_grad()
def score_bag(model, features, block_size, device):
    context = full_context(features, device)
    scores = [model(features[a:b].to(device), context).cpu()
              for a, b in block_ranges(len(features), block_size)]
    result = torch.cat(scores)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite student scores")
    return result


def backward_bag(model, features, targets, block_size, device, loss_kind="mse",
                 pairs=None, scale=1.0):
    """Two-pass VJP, exact for this deterministic block operator and chosen pairs.

    Pass 1 retains only O(N) scalar scores on CPU and computes dL/dscores there.
    Pass 2 recomputes each block, applies its vector-Jacobian product, then frees
    the graph. No cross-bag graphs survive. LayerNorm and zero dropout make replay
    identical; adding stateful/stochastic layers invalidates this contract.
    """
    if any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout))
           for m in model.modules()):
        # Transformer contains Dropout(p=0), which is deterministic.
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) or (
                    isinstance(m, torch.nn.Dropout) and m.p != 0):
                raise ValueError("Replay requires no BatchNorm or nonzero dropout")
    scores = score_bag(model, features, block_size, device).requires_grad_(True)
    loss = objective(scores, targets.cpu(), kind=loss_kind, pairs=pairs)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite student loss")
    score_grad, = torch.autograd.grad(loss, scores)
    context = full_context(features, device)
    for a, b in block_ranges(len(features), block_size):
        prediction = model(features[a:b].to(device), context)
        prediction.backward(score_grad[a:b].to(device) * scale)
    return float(loss.detach()), scores.detach()


def select_indices(scores, ratio=0.3, top_k=None, min_patches=10):
    n = len(scores)
    if n < 1 or not torch.isfinite(scores).all() or not 0 < ratio <= 1:
        raise ValueError("Invalid scores or ratio")
    if min_patches < 1 or (top_k is not None and top_k < 1):
        raise ValueError("Sampling counts must be positive")
    k = min(n, top_k if top_k is not None else max(int(n * ratio), min_patches))
    # Stable tie handling; restore original feature order after selection.
    return torch.argsort(scores, descending=True, stable=True)[:k].sort().values
