"""MIL compatibility adapters and explicitly bounded-context students."""
import torch
from torch import nn
from torch.nn import functional as F


MIL_NAMES = ("attmil", "rrtmil", "transmil", "dsmil", "meanmil", "maxmil")
STUDENT_NAMES = ("patch_mlp", "context_mlp", "transformer")


class MILClassifier(nn.Module):
    def __init__(self, name="attmil", input_dim=1024, dropout=0.25, params=None):
        super().__init__()
        if name not in MIL_NAMES:
            raise ValueError(f"Supported MIL models: {MIL_NAMES}")
        self.name = name
        params = dict(params or {})
        kw = dict(input_dim=input_dim, n_classes=2, dropout=dropout, act="relu")
        if name == "attmil":
            from .vendor.attmil import DAttention
            cls = DAttention
        elif name == "rrtmil":
            from .vendor.rrt import RRTMIL
            cls = RRTMIL
        elif name == "transmil":
            from .vendor.transmil import TransMIL
            cls = TransMIL
        elif name == "dsmil":
            from .vendor.dsmil import MILNet
            cls = MILNet
        else:
            from .vendor.mean_max import MeanMIL, MaxMIL
            cls = MeanMIL if name == "meanmil" else MaxMIL
        self.network = cls(**kw, **params)

    def forward(self, x):
        # DSMIL's training forward requires labels; its eval branch returns a tuple.
        if self.name == "dsmil" and self.training:
            raise RuntimeError("Use training_loss for DSMIL training")
        result = self.network(x)
        return result[0] if isinstance(result, tuple) else result

    def training_loss(self, x, label):
        if self.name == "dsmil":
            logits, instance_loss, _ = self.network(x, label, nn.CrossEntropyLoss())
            return 0.5 * F.cross_entropy(logits, label) + 0.5 * instance_loss
        return F.cross_entropy(self(x), label)

    def load_legacy(self, path):
        from .io import load_checkpoint
        checkpoint = load_checkpoint(path)
        self.network.load_state_dict(checkpoint.get("model", checkpoint), strict=True)


class ImportanceStudent(nn.Module):
    """No BatchNorm/dropout: deterministic replay of the declared context operator.

    patch_mlp: independent patch function.
    context_mlp: patch plus exact full-bag mean.
    transformer: fixed contiguous block self-attention plus exact full-bag mean.
    Block membership is part of the experiment, not an invisible implementation detail.
    """
    def __init__(self, input_dim=1024, kind="context_mlp", hidden_dim=128,
                 n_heads=4, n_layers=2, use_global_context=True):
        super().__init__()
        if kind not in STUDENT_NAMES:
            raise ValueError(f"Supported students: {STUDENT_NAMES}")
        if hidden_dim < 2 or (kind == "transformer" and hidden_dim % n_heads):
            raise ValueError("Invalid hidden dimension or attention head count")
        self.kind = kind
        self.use_global_context = kind == "context_mlp" or (
            kind == "transformer" and use_global_context)
        dim = input_dim * (2 if self.use_global_context else 1)
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim),
                                     nn.GELU(), nn.LayerNorm(hidden_dim))
        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(hidden_dim, n_heads,
                       dim_feedforward=hidden_dim * 2, dropout=0.0, activation="gelu",
                       batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        else:
            self.encoder = nn.Identity()
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                                  nn.Linear(hidden_dim // 2, 1))

    def forward(self, patches, context=None):
        if patches.ndim != 2:
            raise ValueError("Student expects a single block [N,D]")
        if self.use_global_context:
            if context is None or context.ndim != 1:
                raise ValueError("Full-bag mean context [D] is required")
            patches = torch.cat((patches, context.expand(len(patches), -1)), dim=-1)
        x = self.project(patches)
        x = self.encoder(x.unsqueeze(0)).squeeze(0)
        return self.head(x).squeeze(-1)


def build_mil(config):
    return MILClassifier(**config)


def build_student(config):
    return ImportanceStudent(**config)
