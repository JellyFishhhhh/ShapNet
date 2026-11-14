"""Round-specific dictionary fitted on authorized training features only."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from torch.nn import functional as F


class Morphology:
    def __init__(self, mean, scale, centers, fit_patients, protocol):
        self.mean, self.scale, self.centers = mean, scale, centers
        self.fit_patients, self.protocol = sorted(fit_patients), protocol

    @classmethod
    def fit(cls, store, queried, train, k=8, per_slide=512, max_fit=16384,
            seed=0, protocol='queried_only'):
        if protocol not in ('queried_only', 'train_pool_unlabeled'):
            raise ValueError('Unknown morphology protocol')
        patients = sorted(queried if protocol == 'queried_only' else train)
        if not set(patients) <= set(train) or not patients:
            raise ValueError('Morphology fit must use nonempty authorized training patients')
        if min(k, per_slide, max_fit) < 1:
            raise ValueError('Positive morphology sizes required')
        generator = torch.Generator().manual_seed(seed)
        samples = []
        for slide in store.for_patients(patients):
            x = store.load(slide.slide_id)
            samples.append(x[torch.randperm(len(x), generator=generator)[:per_slide]])
        x = torch.cat(samples)
        x = x[torch.randperm(len(x), generator=generator)[:max_fit]]
        mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)
        z = (x-mean) / scale
        # Parallel centroid accumulation can change rounding and downstream Adam paths.
        with threadpool_limits(limits=1):
            km = KMeans(n_clusters=min(k, len(z)), random_state=seed, n_init=5).fit(z.numpy())
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        # Keep the model input width fixed even with tiny queried material banks.
        centers = centers[torch.arange(k) % len(centers)]
        return cls(mean, scale, centers, patients, protocol)

    def encode(self, x):
        mean, scale, centers = (t.to(x.device) for t in (self.mean, self.scale, self.centers))
        z = (x-mean) / scale
        # Euclidean assignment defines real composition; cosine is the network input.
        clusters = torch.cdist(z, centers).argmin(1)
        similarity = F.normalize(z, dim=1) @ F.normalize(centers, dim=1).T
        return z, similarity, clusters

    def composition(self, x):
        ids = self.encode(x)[2]
        return torch.bincount(ids, minlength=len(self.centers)).float() / len(ids)

    def state_dict(self):
        return dict(mean=self.mean, scale=self.scale, centers=self.centers,
                    fit_patients=self.fit_patients, protocol=self.protocol)

    @classmethod
    def from_state(cls, state):
        return cls(**state)
