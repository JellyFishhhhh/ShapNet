"""Slice metrics primary; patient probabilities/embeddings use unweighted slide means."""
import csv
import io
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from shapnet.io import write_json


@torch.no_grad()
def describe(model, morphology, store, patients):
    model.eval()
    device = next(model.parameters()).device
    by_patient, slides = {}, {}
    for slide in store.for_patients(patients):
        x = store.load(slide.slide_id)
        out = model(x.to(device), morphology)
        entry = dict(patient_id=slide.patient_id,
                     probability=float(out['fused_logits'].softmax(-1)[1]),
                     view_probs=out['view_probs'].cpu().tolist(),
                     representation=out['representation'].cpu().tolist(),
                     composition=out['composition'].cpu().tolist())
        slides[slide.slide_id] = entry
        by_patient.setdefault(slide.patient_id, []).append(entry)
    result = {}
    for p, entries in by_patient.items():
        result[p] = {k: np.mean([e[k] for e in entries], axis=0).tolist()
                     for k in ('probability', 'view_probs', 'representation', 'composition')}
    return result, slides


@torch.no_grad()
def describe_synthetic(model, morphology, bank):
    device = next(model.parameters()).device
    model.eval()
    result = []
    for bag in bank:
        out = model(bag.features.to(device), morphology)
        result.append(dict(bag_id=bag.bag_id, representation=out['representation'].cpu().tolist(),
                           composition=out['composition'].cpu().tolist()))
    return result


def metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite predictions')
    return dict(auc=float(roc_auc_score(y, p)) if len(set(y))==2 else None,
                macro_f1=float(f1_score(y, p>=.5, labels=[0,1], average='macro', zero_division=0)),
                accuracy=float(accuracy_score(y, p>=.5)), threshold=.5, n=len(y))


def evaluate(model, morphology, store, patients, labels):
    # Labels are accessed only after predictions have been constructed.
    patient, slide = describe(model, morphology, store, patients)
    return dict(patient=metrics([labels[p] for p in patient], [e['probability'] for e in patient.values()]),
                slide=metrics([labels[e['patient_id']] for e in slide.values()],
                              [e['probability'] for e in slide.values()])), dict(patients=patient, slides=slide)


def curve_integral(y, x, axis=-1):
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return integrate(y, x, axis=axis)


def export_curves(output, rounds, train_count, val_count):
    output = Path(output)
    columns = ['round', 'train_query_patients', 'validation_patients', 'total_labeled_patients',
               'train_fraction', 'slide_auc', 'slide_macro_f1', 'patient_auc', 'patient_macro_f1',
               'synthetic_bags', 'round_seconds', 'peak_cuda_bytes']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
    for r in rounds:
        writer.writerow(dict(round=r['round'], train_query_patients=r['queried'], validation_patients=val_count,
            total_labeled_patients=r['queried']+val_count, train_fraction=r['queried']/train_count,
            slide_auc=r['test']['slide']['auc'], slide_macro_f1=r['test']['slide']['macro_f1'],
            patient_auc=r['test']['patient']['auc'], patient_macro_f1=r['test']['patient']['macro_f1'],
            synthetic_bags=r['synthetic_bags'], round_seconds=r['seconds'], peak_cuda_bytes=r['peak_cuda_bytes']))
    tmp = output/'budget_performance.csv.tmp'; tmp.write_text(stream.getvalue())
    tmp.replace(output/'budget_performance.csv')
    x = np.array([r['queried']/train_count for r in rounds])
    area = {}
    for level in ('slide','patient'):
        for metric in ('auc','macro_f1'):
            y = [r['test'][level][metric] for r in rounds]
            key = f'{level}_{metric}'
            integral = float(curve_integral(y, x)) if all(v is not None for v in y) and len(x)>1 else None
            area[key] = dict(area=integral, normalized_area=integral/float(x[-1]-x[0]) if integral is not None else None)
    write_json(output/'curve_area.json', dict(x_axis='queried_fraction_of_train_patients',
        observed_range=[float(x[0]),float(x[-1])], interpolation='trapezoidal; no extrapolation', metrics=area))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1,2,figsize=(9,3.5))
    for ax, metric in zip(axes, ('auc','macro_f1')):
        ax.plot(x, [r['test']['slide'][metric] for r in rounds], 'o-', label='Slide')
        ax.plot(x, [r['test']['patient'][metric] for r in rounds], 's--', label='Patient')
        ax.set(xlabel='Queried / training patients', ylabel=metric, ylim=(0,1)); ax.legend()
    fig.tight_layout(); fig.savefig(output/'budget_performance.png', dpi=160); plt.close(fig)
