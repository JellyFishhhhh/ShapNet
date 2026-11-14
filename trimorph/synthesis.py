"""Controlled same-class cross-patient recombination with patch provenance."""
from dataclasses import dataclass
import math
import numpy as np
import torch

from .model import attention_importance, stratified_indices


@dataclass
class SyntheticBag:
    bag_id: str
    features: torch.Tensor
    label: int
    round_index: int
    source_patient: str
    donor_patient: str
    origins: list
    protected_positions: list
    mode: str
    requested_replacements: int
    actual_replacements: int

    def state_dict(self):
        return dict(vars(self))


def synthesize(store, revealed, model, morphology, round_index, per_patient=1,
               max_patches=1024, retain_ratio=.25, replacement_ratio=.5,
               mode='matched', seed=0, importance=attention_importance):
    if mode not in ('matched', 'ordinary') or per_patient < 0 or max_patches < 1:
        raise ValueError('Invalid synthesis mode/quota')
    if not 0 < retain_ratio <= 1 or not 0 <= replacement_ratio <= 1-retain_ratio:
        raise ValueError('Require protected fraction > 0 and replacement <= 1-retain_ratio')
    rng = np.random.default_rng(seed)
    slides = store.for_patients(revealed)
    device = next(model.parameters()).device
    bank = []
    model.eval()
    for patient in sorted(revealed):
        own = [s for s in slides if s.patient_id == patient]
        donors = [s for s in slides if s.patient_id != patient and revealed[s.patient_id] == revealed[patient]]
        if not donors:
            continue
        for replica in range(per_patient):
            source = own[replica % len(own)]
            donor = donors[int(rng.integers(len(donors)))]
            x, donor_x = store.load(source.slide_id), store.load(donor.slide_id)
            with torch.no_grad():
                scores = importance(model, x.to(device), morphology).cpu()
                clusters = morphology.encode(x)[2]
                donor_clusters = morphology.encode(donor_x)[2]
            n = min(len(x), max_patches)
            protected_n = min(n, max(1, math.ceil(n*retain_ratio)))
            protected = torch.argsort(scores, descending=True, stable=True)[:protected_n]
            other = stratified_indices(clusters, n-protected_n, protected)
            base_indices = torch.cat([protected, other])
            result = x[base_indices].clone()
            origins = [dict(patient_id=patient, slide_id=source.slide_id, patch_index=int(i)) for i in base_indices]
            requested = min(n-protected_n, int(n*replacement_ratio))
            positions = rng.permutation(np.arange(protected_n, n))[:requested]
            actual = 0
            for pos in positions:
                candidates = (torch.where(donor_clusters == clusters[base_indices[pos]])[0]
                              if mode == 'matched' else torch.arange(len(donor_x)))
                if not len(candidates):
                    continue  # Never silently substitute an unmatched morphology.
                donor_index = int(candidates[int(rng.integers(len(candidates)))])
                result[pos] = donor_x[donor_index]
                origins[pos] = dict(patient_id=donor.patient_id, slide_id=donor.slide_id, patch_index=donor_index)
                actual += 1
            if actual:
                bank.append(SyntheticBag(f'syn_r{round_index}_{patient}_{replica}', result,
                    revealed[patient], round_index, patient, donor.patient_id, origins,
                    list(range(protected_n)), mode, requested, actual))
    validate_bank(bank, revealed, store)
    return bank


def validate_bank(bank, revealed, store):
    for bag in bank:
        patients = {o['patient_id'] for o in bag.origins}
        if (bag.source_patient == bag.donor_patient or
                patients != {bag.source_patient, bag.donor_patient} or
                not patients <= set(revealed) or
                any(revealed[p] != bag.label for p in patients)):
            raise ValueError('Synthetic source label/patient violation')
        if len(bag.features) != len(bag.origins) or bag.actual_replacements <= 0:
            raise ValueError('Invalid synthetic provenance')
        for origin in bag.origins:
            if store.slides[origin['slide_id']].patient_id != origin['patient_id']:
                raise ValueError('Synthetic slide/patient mismatch')
        if any(bag.origins[i]['patient_id'] != bag.source_patient for i in bag.protected_positions):
            raise ValueError('Protected source patches were replaced')
