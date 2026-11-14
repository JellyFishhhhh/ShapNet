"""Small multi-slide synthetic cohort, never presented as clinical evidence."""
import argparse
import csv
import json
from pathlib import Path
import torch
from .config import resolve


def generate(root, patients=40, input_dim=8, seed=17):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.csv').exists():
        raise FileExistsError('Demo already exists')
    generator=torch.Generator().manual_seed(seed)
    with (root/'manifest.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['slide_id','patient_id','label','feature_path'])
        for i in range(patients):
            for s in range(1+i%2):
                sid=f's{i:03d}_{s}'
                x=torch.randn(12+i%7,input_dim,generator=generator)
                x[:max(1,len(x)//4),0] += (2*(i%2)-1)*2
                torch.save(x,root/f'{sid}.pt')
                writer.writerow([sid,f'p{i:03d}',i%2,f'{sid}.pt'])
    c=resolve(dict(manifest=str(root/'manifest.csv'),output_dir=str(root/'run'),input_dim=input_dim,
        model=dict(hidden_dim=12,attention_dim=6,diverse_patches=2,balanced_patches=6),
        morphology=dict(k=3,per_slide=12,max_fit=100),
        training=dict(epochs=1,warmup_epochs=1),
        synthesis=dict(max_patches=12),
        budget=dict(initial=.25,step=.125,maximum=.5)))
    path=root/'demo.json'; path.write_text(json.dumps(c,indent=2)+'\n')
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    args=parser.parse_args(); print(generate(args.output))
