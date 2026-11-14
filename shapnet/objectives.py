"""Additive regression and globally sampled pairwise ranking, without N×N tensors."""
import torch
from torch.nn import functional as F


def normalize_targets(values, method="log"):
    values = values.detach().cpu().float()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite, nonnegative gradient-norm targets")
    if method == "none":
        return values
    if method == "quantile":
        # Average ties; a constant vector must not acquire an arbitrary ordering.
        from scipy.stats import rankdata
        if len(values) == 1:
            return torch.full_like(values, 0.5)
        return torch.tensor((rankdata(values.numpy()) - 1) / (len(values) - 1), dtype=torch.float32)
    if method == "log":
        values = torch.log1p(values * 1000)
    elif method != "minmax":
        raise ValueError("normalization must be log/minmax/quantile/none")
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 0 else torch.zeros_like(values)


def sample_pairs(n, count, seed):
    if count < 1 or n < 1:
        raise ValueError("Positive pair count and bag size required")
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(n, (count, 2), generator=rng)


def objective(scores, target, kind="mse", pairs=None, margin=1.0):
    if scores.shape != target.shape or scores.ndim != 1:
        raise ValueError("Scores and targets must be matching vectors")
    if kind == "mse":
        return F.mse_loss(scores, target)
    if kind == "huber":
        return F.huber_loss(scores, target)
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(scores, target)
    if kind in ("pairwise", "margin_pairwise_logistic"):
        if pairs is None:
            raise ValueError("Explicit global pairs required")
        left, right = pairs[:, 0], pairs[:, 1]
        sign = torch.sign(target[left] - target[right])
        valid = sign != 0
        if not valid.any():
            return scores.sum() * 0
        diff = scores[left] - scores[right]
        return F.softplus(margin - sign[valid] * diff[valid]).mean()
    raise ValueError("Supported losses: mse, huber, bce, pairwise, margin_pairwise_logistic")


def binary_targets(values, ratio):
    if not 0 < ratio <= 1:
        raise ValueError("binary_top_ratio must be in (0,1]")
    count = max(1, int(len(values) * ratio))
    indices = torch.argsort(values, descending=True, stable=True)[:count]
    labels = torch.zeros_like(values)
    labels[indices] = 1
    return labels
