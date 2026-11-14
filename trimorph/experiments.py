"""Generate paired 3-seed controls without launching a long experiment implicitly."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import shlex

import numpy as np

from shapnet.io import write_json
from .config import read_config
from .runner import run
from .evaluation import curve_integral

VARIANTS = {
    'random_no_synthesis': {'acquisition.strategy':'random','synthesis.enabled':False},
    'random_synthesis': {'acquisition.strategy':'random'},
    'coverage_no_synthesis': {'synthesis.enabled':False},
    'full': {},
    'entropy': {'acquisition.strategy':'entropy'},
    'three_view_information': {'acquisition.strategy':'disagreement'},
    'entropy_weighted_coverage': {'acquisition.alpha':0.},
    'real_only_coverage': {'acquisition.synthetic_coverage':False},
    'ordinary_recombination': {'synthesis.mode':'ordinary'},
    'no_morphology_input': {'model.use_morphology':False},
    'no_view_identifiers': {'model.use_view_ids':False},
    'no_gated_fusion': {'model.use_fusion':False},
    'single_view': {'model.three_views':False},
    'abmil': {'model.three_views':False,'model.use_morphology':False,'model.use_view_ids':False,'model.use_fusion':False},
}


def generate(config, destination, seeds=(0,1,2)):
    base=read_config(config)
    # Full is the reference; callers cannot accidentally disable its central modules.
    if (not base['synthesis']['enabled'] or base['synthesis']['mode']!='matched' or
        base['acquisition']['strategy']!='coverage' or not base['acquisition']['synthetic_coverage'] or
        not all(base['model'][k] for k in ('use_morphology','use_view_ids','use_fusion','three_views'))):
        raise ValueError('Suite base must be the full method; use individual configs for other protocols')
    root=Path(destination).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Suite destination must be new/empty')
    root.mkdir(parents=True,exist_ok=True)
    commands=[]; jobs=[]
    for name, changes in VARIANTS.items():
        for seed in seeds:
            c=deepcopy(base); c['seed']=seed
            for key,value in changes.items():
                group,field=key.split('.'); c[group][field]=value
            c['output_dir']=str(root/'runs'/name/f'seed_{seed}')
            path=root/'configs'/f'{name}_seed_{seed}.json'
            write_json(path,c)
            jobs.append(dict(variant=name,seed=seed,config=str(path),output=c['output_dir']))
            commands.append(f'python -m trimorph --config {shlex.quote(str(path))}')
    write_json(root/'jobs.json',jobs)
    (root/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n# Sequential launch. GPU use is configured explicitly in each JSON.\n'+'\n'.join(commands)+'\n')
    return jobs


def summarize(destination):
    """Paired methods/seeds use the common observed budget interval, without extrapolation."""
    from shapnet.io import load_checkpoint
    root=Path(destination)
    jobs=json.loads((root/'jobs.json').read_text())
    completed=[]; missing=[]
    for job in jobs:
        output=Path(job['output'])
        if not (output/'state.pt').exists() or load_checkpoint(output/'state.pt')['stage']!='complete':
            missing.append(dict(variant=job['variant'],seed=job['seed'])); continue
        with (output/'budget_performance.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        x=np.array([float(r['train_fraction']) for r in rows])
        completed.append(dict(**job,rows=rows,x=x))
    if not completed:
        result=dict(common_range=None,variants={},missing=missing)
        write_json(root/'summary.json',result); return result
    lo=max(float(j['x'][0]) for j in completed); hi=min(float(j['x'][-1]) for j in completed)
    if hi<=lo:
        raise ValueError('Completed curves have no common positive-width budget interval')
    grid=np.array(sorted({lo,hi}|{float(x) for j in completed for x in j['x'] if lo<=x<=hi}))
    variants={}
    keys=('slide_auc','slide_macro_f1','patient_auc','patient_macro_f1')
    for name in sorted({j['variant'] for j in completed}):
        runs=[j for j in completed if j['variant']==name]
        values={}
        for key in keys:
            curves=[]
            for job in runs:
                y=[float(r[key]) if r[key] else np.nan for r in job['rows']]
                if np.isfinite(y).all(): curves.append(np.interp(grid,job['x'],y))
            if not curves:
                values[key]=dict(n=0,mean_area=None,std_area=None); continue
            curves=np.array(curves)
            area=curve_integral(curves,grid,axis=1)/(hi-lo)
            values[key]=dict(n=len(curves),mean_area=float(area.mean()),
                std_area=float(area.std(ddof=1)) if len(area)>1 else None,
                mean_curve=curves.mean(0).tolist(),
                std_curve=curves.std(0,ddof=1).tolist() if len(curves)>1 else None)
        variants[name]=dict(seeds=[j['seed'] for j in runs],metrics=values)
    result=dict(common_range=[lo,hi],budget_grid=grid.tolist(),variants=variants,missing=missing,
                note='Normalized trapezoidal area on common interval; no extrapolation. Missing jobs are explicit.')
    write_json(root/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['generate','run','summarize'])
    parser.add_argument('--directory',required=True)
    parser.add_argument('--config')
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    args=parser.parse_args()
    if args.action=='generate':
        if not args.config: parser.error('--config required')
        print(f'Generated {len(generate(args.config,args.directory,args.seeds))} jobs; nothing launched.')
    elif args.action=='run':
        for job in json.loads((Path(args.directory)/'jobs.json').read_text()):
            run(job['config'],resume=(Path(job['output'])/'state.pt').exists())
    else:
        print(json.dumps(summarize(args.directory),indent=2))


if __name__=='__main__': main()
