"""Frozen-round normalized uncertainty and nonnegative facility-location coverage."""
import math
import numpy as np
import torch


def information_value(probabilities, alpha=1.):
    # [patients, 3 views, classes]; patient probabilities are slide means.
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    if p.ndim != 3 or p.shape[1] != 3 or p.shape[2] < 2 or alpha < 0:
        raise ValueError('Expected [patients,3,classes] and nonnegative alpha')
    if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(p.sum(-1), torch.ones_like(p[..., 0])):
        raise ValueError('Invalid probabilities')
    def entropy(v):
        return -(v*v.clamp_min(1e-15).log()).sum(-1)/math.log(p.shape[-1])
    h = entropy(p.mean(1))
    js = (h-entropy(p).mean(1)).clamp_min(0)
    return h.numpy(), js.numpy(), (h+alpha*js).numpy()


def similarity(representations, compositions, other_reps=None, other_comps=None,
               representation_weight=.5, sigma=1.):
    """RBF of squared L2-normalized MIL and Hellinger morphology distances."""
    if not 0 <= representation_weight <= 1 or sigma <= 0:
        raise ValueError('Invalid similarity kernel')
    a = np.asarray(representations, dtype=np.float64)
    b = a if other_reps is None else np.asarray(other_reps, dtype=np.float64)
    c = np.sqrt(np.asarray(compositions, dtype=np.float64).clip(0))
    d = c if other_comps is None else np.sqrt(np.asarray(other_comps, dtype=np.float64).clip(0))
    a = a/np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b/np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    dist = representation_weight*((a[:, None]-b[None])**2).sum(-1)/4
    dist += (1-representation_weight)*((c[:, None]-d[None])**2).sum(-1)/2
    return np.exp(-dist/(2*sigma*sigma))


def greedy_coverage(unlabeled_ids, kernel, q, base_coverage, count):
    """Exact supplied F(B); U and q stay fixed throughout the batch, including selected j."""
    ids = list(unlabeled_ids)
    s, q, current = map(lambda x: np.array(x, dtype=np.float64, copy=True), (kernel, q, base_coverage))
    if len(set(ids)) != len(ids) or s.shape != (len(ids), len(ids)) or q.shape != (len(ids),) or current.shape != q.shape:
        raise ValueError('Invalid coverage dimensions/identities')
    if not 0 <= count <= len(ids) or not all(np.isfinite(x).all() for x in (s,q,current)) or (q<0).any() or (s<0).any() or (current<0).any():
        raise ValueError('Invalid coverage weights or count')
    remaining, chosen, gains = list(range(len(ids))), [], []
    for _ in range(count):
        marginal = np.array([np.dot(q, np.maximum(current, s[:, i])-current) for i in remaining])
        # Stable ID order resolves ties, including all-zero q.
        index = int(np.argmax(marginal)); best = remaining.pop(index)
        chosen.append(ids[best]); gains.append(float(marginal[index]))
        current = np.maximum(current, s[:, best])
    return chosen, gains


def acquire(unlabeled, labeled, synthetic, count, strategy='coverage', alpha=1.,
            eta=.25, synthetic_coverage=True, representation_weight=.5, sigma=.5, seed=0):
    if not 0 <= eta < 1:
        raise ValueError('Synthetic discount eta must be in [0,1)')
    ids = sorted(unlabeled)
    if not 0 <= count <= len(ids):
        raise ValueError('Query count exceeds real unlabeled pool')
    if strategy not in ('random', 'entropy', 'disagreement', 'coverage'):
        raise ValueError('Unknown query strategy')
    probs = [unlabeled[p]['view_probs'] for p in ids]
    h, js, q = information_value(probs, alpha)
    record = dict(patient_ids=ids, entropy=h.tolist(), js=js.tolist(), q=q.tolist(),
                  strategy=strategy, frozen_batch=True, eta=eta)
    if strategy == 'random':
        chosen = np.random.default_rng(seed).permutation(ids)[:count].tolist(); gains = []
    elif strategy in ('entropy', 'disagreement'):
        scores = h if strategy == 'entropy' else q
        chosen = [ids[i] for i in np.argsort(-scores, kind='stable')[:count]]; gains = []
    else:
        reps = np.array([unlabeled[p]['representation'] for p in ids])
        comps = np.array([unlabeled[p]['composition'] for p in ids])
        def kernel(other):
            return similarity(reps, comps, [v['representation'] for v in other],
                              [v['composition'] for v in other], representation_weight, sigma)
        real_base = np.zeros(len(ids))
        synthetic_base = np.zeros(len(ids))
        if labeled:
            real_base = kernel(list(labeled.values())).max(1)
        if synthetic_coverage and synthetic:
            synthetic_base = eta*kernel(list(synthetic)).max(1)
        base = np.maximum(real_base, synthetic_base)
        record.update(base_coverage=base.tolist(), real_coverage=real_base.tolist(),
                      discounted_synthetic_coverage=synthetic_base.tolist(),
                      synthetic_dominates_count=int((synthetic_base>real_base).sum()),
                      mean_synthetic_coverage_increment=float((base-real_base).mean()))
        chosen, gains = greedy_coverage(ids, similarity(reps, comps, representation_weight=representation_weight, sigma=sigma), q, base, count)
    record.update(selected=chosen, marginal_gains=gains)
    return chosen, record
