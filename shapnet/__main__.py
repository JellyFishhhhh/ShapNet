import argparse
import json
from pathlib import Path
import torch
from .data import read_manifest, make_folds, select_rows, load_features, validate_fold, manifest_identity
from .io import seed_all, write_json, save_checkpoint, load_checkpoint, digest, environment
from .models import build_mil, build_student
from .objectives import normalize_targets, sample_pairs
from .streaming import score_bag, backward_bag, select_indices
from .metrics import ranking_metrics, macro_ranking, choose_threshold, binary_metrics


def run(config, stages):
    c = json.loads(Path(config).read_text())
    if c.get('student_loss', 'mse') not in ('mse', 'huber', 'pairwise', 'margin_pairwise_logistic'):
        raise ValueError('Unsupported CLI student loss')
    if c.get('mil_epochs', 100) < 1 or c.get('student_epochs', 50) < 1:
        raise ValueError('Epoch counts must be positive')
    if c['mil']['input_dim'] != c['student']['input_dim'] or c['block_size'] < 1:
        raise ValueError('Feature dimensions must agree; block_size must be positive')
    seed_all(c.get('seed', 2021))
    torch.set_num_threads(c.get('cpu_threads', 2))
    device = torch.device(c.get('device', 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    root = Path(c['output_dir']); root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != c:
        raise ValueError('Output directory already belongs to another configuration')
    write_json(config_path, c)
    write_json(root / 'environment.json', environment())
    rows = read_manifest(c['manifest'])
    split_path = root / 'splits.json'
    if split_path.exists():
        splits = json.loads(split_path.read_text())
        if splits['manifest_sha256'] != manifest_identity(rows):
            raise ValueError('Manifest changed since split creation')
    else:
        splits = make_folds(rows, c.get('folds', 3), c.get('val_ratio', .2), c.get('seed', 2021))
        write_json(split_path, splits)
    for k in c.get('run_folds', list(range(len(splits['folds'])))):
        fold = splits['folds'][k]; validate_fold(rows, fold)
        out = root / f'fold_{k}'; out.mkdir(exist_ok=True)
        train, val, test = [select_rows(rows, fold[s]) for s in ('train','val','test')]
        teacher = build_mil(c['mil']).to(device)
        tp = out / 'teacher.pt'; sp = out / 'student.pt'; targets_path = out / 'targets.pt'
        def features(row):
            return load_features(row, c['mil']['input_dim'])
        def probabilities(records, student=None):
            teacher.eval(); result=[]
            with torch.no_grad():
                for row in records:
                    x = features(row)
                    if student is not None:
                        scores=score_bag(student,x,c['block_size'],device)
                        x=x[select_indices(scores,c.get('sample_ratio',.3), min_patches=c.get('min_patches',10))]
                    result.append(float(teacher(x.unsqueeze(0).to(device)).softmax(-1)[0,1]))
            return result
        if 'teacher' in stages:
            if tp.exists(): raise FileExistsError('Teacher already exists; omit teacher stage or choose new output_dir')
            opt=torch.optim.AdamW(teacher.parameters(),lr=c.get('mil_lr',2e-4))
            best=-1.; history=[]
            for epoch in range(c.get('mil_epochs',100)):
                teacher.train(); losses=[]
                for j in torch.randperm(len(train)).tolist():
                    row=train[j]; opt.zero_grad(set_to_none=True)
                    loss=teacher.training_loss(features(row).unsqueeze(0).to(device),torch.tensor([row['label']],device=device))
                    if not torch.isfinite(loss): raise FloatingPointError('Teacher loss nonfinite')
                    loss.backward(); opt.step(); losses.append(float(loss.detach()))
                probs=probabilities(val)
                metrics=binary_metrics([r['label'] for r in val],probs,.5)
                history.append(dict(epoch=epoch,loss=sum(losses)/len(losses),validation=metrics))
                if metrics['auc']>best:
                    best=metrics['auc']; save_checkpoint(tp,{'model':teacher.state_dict(),'epoch':epoch})
                print(f'fold={k} teacher epoch={epoch} val_auc={metrics["auc"]:.4f}',flush=True)
            write_json(out/'teacher_history.json',history)
        teacher.load_state_dict(load_checkpoint(tp)['model']); teacher.eval()
        if 'targets' in stages:
            if targets_path.exists(): raise FileExistsError('Targets already exist')
            targets={}
            for row in train+val:
                x=features(row).unsqueeze(0).to(device).requires_grad_(True)
                grad,=torch.autograd.grad(teacher(x)[0,1],x)
                values=grad.squeeze(0).norm(dim=-1).detach().cpu()
                if not torch.isfinite(values).all(): raise FloatingPointError('Invalid importance target')
                targets[row['slide_id']]={'values':values,'feature_sha256':digest(row['feature_path'])}
                del grad,x
            save_checkpoint(targets_path,{'method':'positive_logit_input_gradient_l2','teacher_sha256':digest(tp),'targets':targets})
        student=build_student(c['student']).to(device)
        if 'student' in stages:
            if sp.exists(): raise FileExistsError('Student already exists')
            cache=load_checkpoint(targets_path)
            if cache['teacher_sha256']!=digest(tp): raise ValueError('Teacher/target identity mismatch')
            target={}
            for row in train+val:
                item=cache['targets'][row['slide_id']]
                if item['feature_sha256']!=digest(row['feature_path']): raise ValueError('Feature changed after target generation')
                target[row['slide_id']]=normalize_targets(item['values'],c.get('normalization','log'))
            opt=torch.optim.AdamW(student.parameters(),lr=c.get('student_lr',1e-3))
            best=-float('inf'); history=[]
            for epoch in range(c.get('student_epochs',50)):
                student.train()
                for j in torch.randperm(len(train)).tolist():
                    row=train[j]; x=features(row); opt.zero_grad(set_to_none=True)
                    pairs=sample_pairs(len(x),c.get('pair_count',4096),c.get('seed',2021)+epoch*len(train)+j)
                    backward_bag(student,x,target[row['slide_id']],c['block_size'],device,c.get('student_loss','mse'),pairs)
                    torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True); opt.step()
                student.eval(); ranks=[]
                for row in val:
                    scores=score_bag(student,features(row),c['block_size'],device)
                    ranks.append(ranking_metrics(scores,target[row['slide_id']],c.get('sample_ratio',.3)))
                metrics=macro_ranking(ranks); history.append(dict(epoch=epoch,validation=metrics))
                criterion=metrics['spearman'] if metrics['spearman'] is not None else -metrics['mae']
                if criterion>best:
                    best=criterion; save_checkpoint(sp,{'model':student.state_dict(),'epoch':epoch})
                print(f'fold={k} student epoch={epoch} val_spearman={metrics["spearman"]}',flush=True)
            write_json(out/'student_history.json',history)
        if 'evaluate' in stages:
            student.load_state_dict(load_checkpoint(sp)['model']); student.eval()
            report={}
            for mode, predictor in [('baseline',None),('sampled',student)]:
                threshold=choose_threshold([r['label'] for r in val],probabilities(val,predictor))
                report[mode]=binary_metrics([r['label'] for r in test],probabilities(test,predictor),threshold)
                report[mode]['threshold_source']='validation'
            write_json(out/'metrics.json',report)
            print(json.dumps({'fold':k,**report}),flush=True)


def main():
    p=argparse.ArgumentParser(description='ShapNet binary WSI experiment runner')
    p.add_argument('--config',required=True)
    p.add_argument('--stages',default='teacher,targets,student,evaluate')
    args=p.parse_args(); stages=args.stages.split(',')
    if not stages or any(s not in ['teacher','targets','student','evaluate'] for s in stages):
        p.error('Unknown stage')
    run(args.config,stages)

if __name__=='__main__': main()
