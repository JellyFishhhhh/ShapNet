"""Explicit optimizer and epoch scheduler settings recorded in run config."""
import torch

def make_optimizer(parameters, name, lr, weight_decay):
    if name not in ('adam','adamw'):
        raise ValueError('optimizer must be adam or adamw')
    cls=torch.optim.Adam if name=='adam' else torch.optim.AdamW
    return cls(parameters,lr=lr,weight_decay=weight_decay)

def make_scheduler(optimizer, name, epochs, min_ratio=0.):
    if name=='constant': return None
    if name!='cosine': raise ValueError('scheduler must be cosine or constant')
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=optimizer.param_groups[0]['lr']*min_ratio)
