"""Bounded feature health check. No patient inference, clustering, or task fitting."""
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np
import torch
from shapnet.io import write_json


def inspect(root, max_slides=2, max_patches=512):
    root=Path(root)
    if min(max_slides,max_patches)<1: raise ValueError('Positive inspection bounds required')
    with (root/'label.csv').open(newline='') as f:
        rows=[r for r in csv.reader(f) if r]
    if rows and rows[0][:2]==['slide_id','label']: rows=rows[1:]
    pt=sorted((root/'pt').glob('*.pt')); h5=sorted((root/'h5').glob('*.h5'))
    result=dict(label_rows=len(rows),pt_files=len(pt),h5_files=len(h5),patient_mapping='not inferred',
                note='Bounded health check only; no real task model or morphology fitted',samples=[])
    for path in h5[:max_slides]:
        with h5py.File(path,'r') as f:
            shape=list(f['features'].shape); x=f['features'][:max_patches]
        sample=dict(format='h5',shape=shape,sampled_patches=len(x),finite=bool(np.isfinite(x).all()))
        counterpart=root/'pt'/f'{path.stem}.pt'
        if counterpart.exists():
            tensor=torch.load(counterpart,map_location='cpu',weights_only=True,mmap=True)
            sample.update(pt_shape=list(tensor.shape),pt_sample_matches_h5=bool(np.array_equal(tensor[:max_patches].numpy(),x)))
            del tensor
        result['samples'].append(sample)
    if not result['samples']: raise ValueError('No H5 features to inspect')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--features',required=True); p.add_argument('--output',required=True)
    p.add_argument('--max-slides',type=int,default=2); p.add_argument('--max-patches',type=int,default=512)
    a=p.parse_args(); report=inspect(a.features,a.max_slides,a.max_patches); write_json(a.output,report); print(report)
