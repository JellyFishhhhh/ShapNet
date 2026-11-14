"""Resolved configuration is persisted and hashed before training."""
from copy import deepcopy
import json
from pathlib import Path

DEFAULT = dict(
    manifest='data/manifest.csv', output_dir='runs/trimorph', device='cpu', seed=0,
    split_seed=2026, val_fraction=.2, test_fraction=.2, splits_file=None,
    input_dim=1024, threads=1,
    budget=dict(initial=.15, step=.1, maximum=.5, ensure_initial_classes=True),
    morphology=dict(k=8, per_slide=512, max_fit=16384, protocol='queried_only'),
    model=dict(hidden_dim=128, attention_dim=64, use_morphology=True,
               use_view_ids=True, use_fusion=True, three_views=True,
               discriminative_ratio=.25, diverse_patches=32, balanced_patches=256),
    training=dict(warmup_epochs=1, epochs=20, lr=.0002, weight_decay=.00001,
                  fused_weight=1., synthetic_weight=.5, warm_start=False,
                  external_initialization=None, external_provenance=None),
    synthesis=dict(enabled=True, per_patient=1, max_patches=1024, retain_ratio=.25,
                   replacement_ratio=.5, mode='matched'),
    acquisition=dict(strategy='coverage', alpha=1., eta=.25, synthetic_coverage=True,
                     representation_weight=.5, sigma=.5),
)


def resolve(config):
    def merge(base, override):
        if set(override)-set(base):
            raise ValueError(f'Unknown configuration fields: {sorted(set(override)-set(base))}')
        result = deepcopy(base)
        for k, v in override.items():
            result[k] = merge(base[k], v) if isinstance(base[k], dict) else v
        return result
    c = merge(DEFAULT, config)
    if c['device'] != 'cpu' and not c['device'].startswith('cuda'):
        raise ValueError('Use cpu or cuda[:index]')
    t, a, s, m = (c[k] for k in ('training','acquisition','synthesis','morphology'))
    if min(t['epochs'],t['warmup_epochs'],c['input_dim'],c['threads']) < 1:
        raise ValueError('Positive epochs/dimensions/threads required')
    if t['lr'] <= 0 or min(t['weight_decay'],t['fused_weight'],t['synthetic_weight']) < 0:
        raise ValueError('Invalid training weights')
    if t['external_initialization'] and not t['external_provenance']:
        raise ValueError('External weights require a provenance JSON declaration')
    if not 0 <= a['eta'] < 1 or a['alpha'] < 0 or a['sigma'] <= 0 or not 0 <= a['representation_weight'] <= 1:
        raise ValueError('Invalid acquisition parameters')
    if a['strategy'] not in ('random','entropy','disagreement','coverage'):
        raise ValueError('Unknown acquisition strategy')
    if m['protocol'] not in ('queried_only','train_pool_unlabeled') or min(m['k'],m['per_slide'],m['max_fit'])<1:
        raise ValueError('Invalid morphology protocol')
    if s['mode'] not in ('matched','ordinary') or s['per_patient'] < 0 or s['max_patches']<1 or not 0<s['retain_ratio']<=1 or not 0<=s['replacement_ratio']<=1-s['retain_ratio']:
        raise ValueError('Invalid synthesis settings')
    return c


def read_config(path):
    return resolve(json.loads(Path(path).read_text()))
