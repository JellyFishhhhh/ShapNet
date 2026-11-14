"""Atomic step-resumable active learning; test labels never enter construction."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from shapnet.io import digest, environment, load_checkpoint, object_digest, save_checkpoint, seed_all, write_json
from .config import read_config, resolve
from .data import FeatureStore, Oracle, budget_schedule, make_splits, read_cohort, validate_splits
from .evaluation import describe, describe_synthetic, evaluate, export_curves
from .integrity import feature_digests
from .model import TriMorphMIL, supervised_loss
from .morphology import Morphology
from .selection import acquire
from .synthesis import SyntheticBag, synthesize, validate_bank


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def build_model(c):
    return TriMorphMIL(input_dim=c['input_dim'], prototypes=c['morphology']['k'], **c['model'])


def new_bank(c, store, oracle, model, morphology, r):
    if not c['synthesis']['enabled']:
        return []
    cfg = {k:v for k,v in c['synthesis'].items() if k != 'enabled'}
    return synthesize(store, oracle.revealed, model, morphology, r, seed=c['seed']+1009*r, **cfg)


def _run(config, resume=False, max_updates=None, max_rounds=None):
    """Bounds pause after a committed update/round; resume replays no committed update."""
    c = read_config(config) if isinstance(config, (str,Path)) else resolve(config)
    if c['device'].startswith('cuda'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(c['threads'])
    seed_all(c['seed'])
    torch.use_deterministic_algorithms(True)
    if c['device'].startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; choose device=cpu explicitly')
        index = torch.device(c['device']).index
        torch.cuda.set_device(0 if index is None else index)
    if max_updates is not None and max_updates < 1:
        raise ValueError('max_updates must be positive')
    output = Path(c['output_dir']).resolve()
    slides, setup_labels = read_cohort(c['manifest'])
    splits = (validate_splits(json.loads(Path(c['splits_file']).read_text()), setup_labels)
              if c['splits_file'] else make_splits(setup_labels, c['split_seed'], c['val_fraction'], c['test_fraction']))
    store = FeatureStore(slides, c['input_dim'])
    schedule = budget_schedule(len(splits['train']), **{k:v for k,v in c['budget'].items() if k!='ensure_initial_classes'})
    # Integrity hashes are setup metadata, never model inputs or fitted statistics.
    identity = object_digest(dict(config=c, source={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        manifest=digest(c['manifest']),
        features=feature_digests(slides, c['feature_hash_cache']), splits=splits,
        external=digest(c['training']['external_initialization']) if c['training']['external_initialization'] else None,
        provenance=digest(c['training']['external_provenance']) if c['training']['external_provenance'] else None))
    checkpoint = output/'state.pt'
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError('No state.pt to resume')
        state = load_checkpoint(checkpoint)
        if state['identity'] != identity:
            raise ValueError('Config/data/initialization identity changed; use a new output directory')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Output is nonempty; use --resume or a new output directory')
        output.mkdir(parents=True, exist_ok=True)
        model = build_model(c)
        ext = c['training']['external_initialization']
        if ext:
            provenance = json.loads(Path(c['training']['external_provenance']).read_text())
            if provenance.get('uses_private_labels') is not False or not provenance.get('source'):
                raise ValueError('Declare source and uses_private_labels=false for external weights')
            model.load_state_dict(load_checkpoint(ext), strict=True)
        initial = cpu_tree(model.state_dict())
        state = dict(schema=1, identity=identity, initial=initial, last_best=None,
            splits=splits, schedule=schedule, revealed={}, audit=[], history=[], round=0,
            stage='prepare', updates=0, round_seconds=0., training_history=[])
        oracle = Oracle(setup_labels, splits)
        random_order = np.random.default_rng(c['seed']).permutation(splits['train']).tolist()
        oracle.query(random_order[:schedule[0]], 0, 'random_initialization')
        cursor = schedule[0]
        if c['budget']['ensure_initial_classes']:
            while len(set(oracle.revealed.values()))<2 and cursor<schedule[-1]:
                oracle.query([random_order[cursor]], 0, 'class_coverage_topup'); cursor+=1
        state.update(revealed=oracle.revealed, audit=oracle.audit, initial_query=dict(
            requested=schedule[0],actual=len(oracle.revealed),topups=len(oracle.revealed)-schedule[0],
            classes_observed=sorted(set(oracle.revealed.values())),
            policy='random sequential reveal; optional class-coverage topup bounded by maximum budget'))
        write_json(output/'config.json', c)
        write_json(output/'splits.json', splits)
        write_json(output/'initial_query.json', state['initial_query'])
        write_json(output/'environment.json', environment())
        write_json(output/'initialization.json', dict(
            source=ext or 'fixed seeded MIL initialization; pretrained patch features supplied externally',
            state_sha256=object_digest({k:v.tolist() for k,v in initial.items()}),
            note='Cold rounds reload exactly this state; morphology is refitted per round.'))
        save_checkpoint(checkpoint, state)
    oracle = Oracle(setup_labels, splits, state['revealed'], state['audit'])
    del setup_labels  # No acquisition/training code receives unrevealed or evaluation labels.
    write_json(output/'query_audit.json', state['audit'])
    update_start = state['updates']
    if state['stage'] == 'complete':
        export_curves(output, state['history'], len(splits['train']), len(splits['val']))
        return dict(status='complete', rounds=len(state['history']), updates=state['updates'])
    while state['stage'] != 'complete':
        if max_rounds is not None and len(state['history']) >= max_rounds:
            return dict(status='bounded', rounds=len(state['history']), updates=state['updates'])
        r = state['round']
        round_dir = output/f'round_{r:03d}'
        round_dir.mkdir(exist_ok=True)
        tick = time.perf_counter()
        if c['device'].startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(c['device'])
        if state['stage'] == 'prepare':
            morphology = Morphology.fit(store, oracle.revealed, splits['train'], seed=c['seed']+r, **c['morphology'])
            state.update(morphology=morphology.state_dict(),
                model=deepcopy(state['last_best'] if c['training']['warm_start'] and state['last_best'] is not None else state['initial']),
                optimizer=None, epoch=0, step=0, stage='warmup', bank_file=None, bank_sha256=None, best=None,
                best_score=-1., training_history=[], round_seconds=state['round_seconds']+time.perf_counter()-tick)
            save_checkpoint(checkpoint, state)
        morphology = Morphology.from_state(state['morphology'])
        model = build_model(c).to(c['device']); model.load_state_dict(state['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['training']['lr'], weight_decay=c['training']['weight_decay'])
        if state['optimizer'] is not None:
            optimizer.load_state_dict(state['optimizer'])
        if state['bank_file']:
            if digest(state['bank_file']) != state['bank_sha256']:
                raise ValueError('Synthetic training bank identity changed')
            bank = [SyntheticBag(**b) for b in load_checkpoint(state['bank_file'])]
        else:
            bank = []
        validate_bank(bank, oracle.revealed, store)
        while state['stage'] in ('warmup','train'):
            phase = state['stage']
            epochs = c['training']['warmup_epochs'] if phase=='warmup' else c['training']['epochs']
            if state['epoch'] >= epochs:
                if phase=='warmup':
                    tick = time.perf_counter()
                    bank = new_bank(c, store, oracle, model, morphology, r)
                    bank_file = round_dir/'training_bank.pt'
                    save_checkpoint(bank_file, [b.state_dict() for b in bank])
                    state.update(bank_file=str(bank_file), bank_sha256=digest(bank_file), stage='train', epoch=0, step=0)
                    state['round_seconds'] += time.perf_counter()-tick
                    save_checkpoint(checkpoint, state)
                    continue
                state['stage'] = 'finalize'; save_checkpoint(checkpoint, state); break
            real = store.for_patients(oracle.revealed)
            items = [('real',s.slide_id) for s in real]
            if phase=='train':
                items += [('synthetic',i) for i in range(len(bank))]
            phase_seed = c['seed'] + 100000*r + 1000*(phase=='train') + state['epoch']
            order = np.random.default_rng(phase_seed).permutation(len(items))
            model.train()
            for step in range(state['step'],len(order)):
                tick = time.perf_counter()
                kind, key = items[int(order[step])]
                if kind=='real':
                    x = store.load(key)
                    label = oracle.revealed[store.slides[key].patient_id]; weight=1.
                else:
                    bag = bank[key]; x=bag.features; label=bag.label
                    weight=c['training']['synthetic_weight']
                optimizer.zero_grad(set_to_none=True)
                out = model(x.to(c['device']), morphology)
                loss = weight*supervised_loss(out,label,c['training']['fused_weight'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward(); optimizer.step()
                state.update(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()), step=step+1)
                state['updates'] += 1
                state['round_seconds'] += time.perf_counter()-tick
                save_checkpoint(checkpoint, state)
                if state['step'] % 25 == 0 or state['step'] == len(order):
                    print(f"round={r} phase={phase} epoch={state['epoch']} step={state['step']}/{len(order)} loss={float(loss.detach()):.5f}", flush=True)
                if max_updates is not None and state['updates']-update_start >= max_updates:
                    return dict(status='bounded', rounds=len(state['history']), updates=state['updates'])
            if phase=='train':
                tick = time.perf_counter()
                val, _ = evaluate(model, morphology, store, splits['val'], oracle.evaluation_labels('val'))
                score = val['slide']['auc']
                state['training_history'].append(dict(epoch=state['epoch'], validation=val))
                if score > state['best_score']:
                    state['best'], state['best_score'] = cpu_tree(model.state_dict()), score
                state['round_seconds'] += time.perf_counter()-tick
            state.update(epoch=state['epoch']+1, step=0)
            save_checkpoint(checkpoint, state)
        # Everything below is deterministically reconstructible if interrupted before commit.
        tick = time.perf_counter()
        model.load_state_dict(state['best']); model.eval()
        train_bank = bank
        bank = new_bank(c, store, oracle, model, morphology, r)
        next_target = next((n for n in schedule if n>len(oracle.revealed)), None)
        if next_target is not None and c['acquisition']['strategy']=='random':
            # A random baseline does not need an inference pass over the unlabeled pool.
            ids = sorted(set(splits['train'])-set(oracle.revealed))
            selected = np.random.default_rng(c['seed']+r).permutation(ids)[:next_target-len(oracle.revealed)].tolist()
            acquisition = dict(strategy='random',patient_ids=ids,selected=selected,frozen_batch=True)
        elif next_target is not None:
            train_descriptors, _ = describe(model, morphology, store, splits['train'])
            unlabeled = {p:v for p,v in train_descriptors.items() if p not in oracle.revealed}
            labeled = {p:v for p,v in train_descriptors.items() if p in oracle.revealed}
            synthetic = describe_synthetic(model, morphology, bank) if c['acquisition']['synthetic_coverage'] else []
            selected, acquisition = acquire(unlabeled, labeled, synthetic, next_target-len(oracle.revealed),
                                            seed=c['seed']+r, **c['acquisition'])
        else:
            selected=[]; acquisition=dict(selected=[], reason='budget_exhausted')
        # Test evaluation is strictly downstream of all fit and acquisition decisions.
        test, predictions = evaluate(model, morphology, store, splits['test'], oracle.evaluation_labels('test'))
        state['round_seconds'] += time.perf_counter()-tick
        record = dict(round=r, queried=len(oracle.revealed), validation_patients=len(splits['val']),
            total_labeled_patients=len(oracle.revealed)+len(splits['val']),
            queried_patients=sorted(oracle.revealed), morphology_fit_patients=morphology.fit_patients,
            morphology_protocol=morphology.protocol, test=test, validation_history=state['training_history'],
            synthetic_bags=len(bank), training_synthetic_bags=len(train_bank),
            seconds=state['round_seconds'], peak_cuda_bytes=int(torch.cuda.max_memory_allocated(c['device'])) if c['device'].startswith('cuda') else 0,
            updates=state['updates'], next_query=selected)
        save_checkpoint(round_dir/'model.pt', dict(model=state['best'], morphology=state['morphology']))
        save_checkpoint(round_dir/'coverage_bank.pt', [b.state_dict() for b in bank])
        write_json(round_dir/'synthesis_provenance.json', [
            {k:v for k,v in b.state_dict().items() if k!='features'} for b in bank])
        write_json(round_dir/'acquisition.json', acquisition)
        write_json(round_dir/'test_predictions.json', predictions)
        write_json(round_dir/'metrics.json', record)
        state['history'].append(record)
        state['last_best'] = deepcopy(state['best'])
        if selected:
            oracle.query(selected, r+1)
            state.update(round=r+1, stage='prepare', round_seconds=0.)
        else:
            state['stage']='complete'
        state.update(revealed=dict(oracle.revealed), audit=list(oracle.audit))
        save_checkpoint(checkpoint, state)
        write_json(output/'query_audit.json', oracle.audit)
        export_curves(output, state['history'], len(splits['train']), len(splits['val']))
        print(f"round={r} queried={record['queried']} synth={len(bank)} slide_auc={test['slide']['auc']:.4f}", flush=True)
    return dict(status='complete', rounds=len(state['history']), updates=state['updates'])


def run(config, resume=False, max_updates=None, max_rounds=None):
    """Record active invocation wall time, including artifact I/O, excluding suspension."""
    import resource
    c = read_config(config) if isinstance(config, (str,Path)) else resolve(config)
    tick = time.perf_counter()
    result = _run(c, resume, max_updates, max_rounds)
    output = Path(c['output_dir'])
    path = output/'runtime.json'
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(dict(seconds=time.perf_counter()-tick, resume=resume, result=result,
                        process_peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
    write_json(path, history)
    return result
