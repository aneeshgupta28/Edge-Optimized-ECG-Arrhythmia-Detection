"""
Compact 1D CNN for Beat Classification
======================================
Small convolutional network over the raw beat window, with the RR-interval
features fed in alongside the conv embedding. RR timing is what separates
most supraventricular beats from normal ones (an S beat can look almost
identical to N but arrives early), so dropping it would handicap the CNN.

Input
-----
  beat : (B, 1, 100)  200-sample window resampled 360 Hz -> 180 Hz, per-beat z-scored
  rr   : (B, 4)       pre_rr, post_rr, pre_rr_ratio, post_rr_ratio (standardised)

Also provides a MAC/parameter counter so cost can be reported without
depending on any particular hardware.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

RR_INPUTS = ["pre_rr", "post_rr", "pre_rr_ratio", "post_rr_ratio"]
DOWNSAMPLE = 2


class BeatCNN(nn.Module):
    def __init__(self, n_classes: int = 4, n_rr: int = len(RR_INPUTS), width: int = 16):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv1d(1, w, 7, padding=3), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(w, w, 5, padding=2), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(w, 2 * w, 5, padding=2), nn.BatchNorm1d(2 * w), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(2 * w, 2 * w, 3, padding=1), nn.BatchNorm1d(2 * w), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * w + n_rr, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, beat: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        z = self.features(beat).flatten(1)
        return self.head(torch.cat([z, rr], dim=1))


def prepare_beats(windows: np.ndarray) -> np.ndarray:
    """Downsample and z-score each beat independently (removes per-patient gain)."""
    x = windows[:, ::DOWNSAMPLE].astype(np.float32)
    x = x - np.median(x, axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-6)
    return x[:, None, :]


def count_macs_params(model: nn.Module, beat_len: int, n_rr: int = len(RR_INPUTS)) -> tuple[int, int]:
    """Multiply-accumulates for one beat (conv + linear layers) and parameter count.

    BatchNorm is folded into the preceding conv at deployment, so it is not
    counted. Pooling and ReLU are comparisons, not MACs.
    """
    macs = 0

    def conv_hook(m, inp, out):
        nonlocal macs
        macs += out.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]

    def linear_hook(m, inp, out):
        nonlocal macs
        macs += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, beat_len), torch.zeros(1, n_rr))
    for h in hooks:
        h.remove()

    params = sum(p.numel() for p in model.parameters())
    return macs, params
