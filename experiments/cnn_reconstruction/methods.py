"""Airbench methods adapted to dense RGB reconstruction."""
from math import ceil

import torch
from torch import nn
from torch.nn import functional as F


# A random flip baseline isolates alternating the orientation across epochs.
METHODS = ("baseline", "frozen_whitening", "dirac", "scalebias", "lookahead",
           "alternating_flip", "all_methods")


def enabled(condition, method):
    """Return whether a single-method or combined condition uses this feature."""
    return condition in (method, "all_methods")


def optimizer_for(model, scaled):
    """Build Adam with a matched base rate and optional 64x BatchNorm bias rate."""
    biases = [layer.bias for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)]
    bias_ids = {id(bias) for bias in biases}
    others = [p for p in model.parameters() if id(p) not in bias_ids]
    groups = [{"params": others}, {"params": biases, "lr": 0.064 if scaled else 0.001}]
    return torch.optim.Adam(groups, lr=0.001)


class PaperLookahead:
    """Mix floating model parameters and buffers every five steps using Airbench's schedule."""

    def __init__(self, model, epochs, size, batch_size):
        self.total = epochs * ceil(size / batch_size)
        self.slow = {k: v.clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model, step):
        """Write interpolated weights back after a fast step, including the final partial interval."""
        if step % 5 and step != self.total:
            return
        decay = 0.95 ** 5 * (step / self.total) ** 3
        for key, value in model.state_dict().items():
            if key in self.slow:
                slow = self.slow[key]
                if step % 5 == 0:
                    slow.lerp_(value, 1 - decay)
                value.copy_(slow)


def orientations(phase, epoch, alternating, generator):
    """Return one horizontal flip decision per training image, indexed before shuffling."""
    if epoch == 1:
        return phase
    if alternating:
        return ~phase if epoch % 2 == 0 else phase
    return torch.rand(len(phase), generator=generator) < 0.5


def six_views(model, inputs):
    """Predict from six corrupted views, align RGB outputs and renormalize valid border weights."""
    combined = torch.zeros_like(inputs)
    weights = torch.zeros_like(inputs[:1, :1])
    padded = F.pad(inputs, (1, 1, 1, 1), mode="reflect")
    size = inputs.shape[-1]
    for shift, weight in ((0, 0.25), (-1, 0.125), (1, 0.125)):
        offset = shift + 1
        view = padded[:, :, offset:offset + size, offset:offset + size]
        pair = model(view) + model(view.flip(-1)).flip(-1)
        src = slice(max(0, -shift), min(size, size - shift))
        dst = slice(max(0, shift), min(size, size + shift))
        combined[:, :, dst, dst] += weight * pair[:, :, src, src]
        weights[:, :, dst, dst] += 2 * weight
    return combined / weights
