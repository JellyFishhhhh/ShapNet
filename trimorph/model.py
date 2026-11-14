"""Shared gated ABMIL; subset heads receive no unconditional slide-label loss."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def attention_importance(model, x, morphology):
    """Replaceable interface. Only the current legal model may supply scores."""
    with torch.no_grad():
        z, sim, _ = morphology.encode(x)
        return model.pool(model.project(z, sim), 0)[1]


def stratified_indices(clusters, limit, exclude=None):
    """Deterministic round robin, evenly spaced within each morphology stratum."""
    excluded = set() if exclude is None else set(exclude.tolist())
    groups = []
    for c in torch.unique(clusters, sorted=True):
        ids = [i for i in torch.where(clusters == c)[0].tolist() if i not in excluded]
        if ids:
            # Alternating ends avoids taking only contiguous acquisition order.
            order = list(range(0, len(ids), 2)) + list(range(1, len(ids), 2))[::-1]
            groups.append([ids[i] for i in order])
    selected, offset = [], 0
    while len(selected) < limit and groups:
        emitted = False
        for group in groups:
            if offset < len(group) and len(selected) < limit:
                selected.append(group[offset]); emitted = True
        if not emitted:
            break
        offset += 1
    return torch.tensor(sorted(selected), dtype=torch.long, device=clusters.device)


class TriMorphMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, attention_dim=64, prototypes=8,
                 use_morphology=True, use_view_ids=True, use_fusion=True,
                 three_views=True, discriminative_ratio=.25, diverse_patches=32,
                 balanced_patches=256):
        super().__init__()
        if not 0 < discriminative_ratio <= 1 or min(input_dim, hidden_dim, attention_dim, prototypes, balanced_patches) < 1 or diverse_patches < 0:
            raise ValueError('Invalid network/view dimensions')
        self.base = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.morph = nn.Linear(prototypes, hidden_dim, bias=False) if use_morphology else None
        self.view_ids = nn.Embedding(3, hidden_dim) if use_view_ids else None
        self.a = nn.Linear(hidden_dim, attention_dim)
        self.b = nn.Linear(hidden_dim, attention_dim)
        self.attention = nn.Linear(attention_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim, 2)  # One classifier for every view/fusion.
        self.fusion_gate = nn.Linear(hidden_dim, 1) if use_fusion else None
        self.three_views = three_views
        self.discriminative_ratio, self.diverse_patches = discriminative_ratio, diverse_patches
        self.balanced_patches = balanced_patches

    def project(self, z, sim):
        return self.base(z) + (self.morph(sim) if self.morph is not None else 0)

    def pool(self, h, view):
        if self.view_ids is not None:
            h = h + self.view_ids.weight[view]
        scores = self.attention(torch.tanh(self.a(h)) * torch.sigmoid(self.b(h))).flatten()
        weights = scores.softmax(0)
        return (weights[:, None]*h).sum(0), weights

    def forward(self, x, morphology, importance=attention_importance):
        z, sim, clusters = morphology.encode(x)
        h = self.project(z, sim)
        full, weights = self.pool(h, 0)
        if self.three_views:
            scores = weights.detach() if importance is attention_importance else importance(self, x, morphology).detach()
            if scores.shape != (len(x),) or not torch.isfinite(scores).all():
                raise ValueError('Importance interface must return finite [N] scores')
            top = torch.argsort(scores, descending=True, stable=True)[:max(1, math.ceil(len(x)*self.discriminative_ratio))]
            extra = stratified_indices(clusters, self.diverse_patches, top)
            discr = torch.cat([top, extra]).sort().values
            balanced = stratified_indices(clusters, self.balanced_patches)
            reps = torch.stack([full, self.pool(h[discr], 1)[0], self.pool(h[balanced], 2)[0]])
        else:
            discr = balanced = torch.arange(len(x), device=x.device)
            reps = full[None].expand(3, -1)
        if self.fusion_gate is not None and self.three_views:
            gates = self.fusion_gate(reps).flatten().softmax(0)
            fused = (gates[:, None]*reps).sum(0)
        else:
            gates = x.new_tensor([1., 0., 0.]); fused = full
        logits = self.classifier(reps)
        return dict(view_logits=logits, view_probs=logits.softmax(-1), representation=fused,
                    fused_logits=self.classifier(fused), attention=weights, gates=gates,
                    composition=torch.bincount(clusters, minlength=len(morphology.centers)).float()/len(x),
                    indices=(torch.arange(len(x), device=x.device), discr, balanced))


def supervised_loss(output, label, fused_weight=1.):
    target = torch.as_tensor([label], dtype=torch.long, device=output['view_logits'].device)
    return (F.cross_entropy(output['view_logits'][0:1], target) +
            fused_weight*F.cross_entropy(output['fused_logits'][None], target)) / (1+fused_weight)
