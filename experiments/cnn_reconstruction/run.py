"""Compare whitening or Airbench training methods on masked CIFAR-10 RGB reconstruction."""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from methods import METHODS, PaperLookahead, enabled, optimizer_for, orientations, six_views

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common.data import CIFAR10_MEAN, CIFAR10_STD, load_cifar10

VARIANTS = ("random", "trainable_whitening", "frozen_whitening")


def reconstruction_net(dirac=True):
    """Return a CNN mapping normalized, corrupted N x 3 x 32 x 32 images to RGB in [0,1]."""
    layers = [nn.Conv2d(3, 24, 2), nn.GELU()]
    nn.init.zeros_(layers[0].bias)
    channels = 24
    for width in (64, 256, 256):
        for index in range(2):
            conv = nn.Conv2d(channels, width, 3, padding=1, bias=False)
            if dirac:
                nn.init.dirac_(conv.weight[:channels])
            layers.append(conv)
            if index == 0:
                layers.append(nn.MaxPool2d(2))
            norm = nn.BatchNorm2d(width, eps=1e-12, momentum=0.4)
            norm.weight.requires_grad_(False)
            layers.extend([norm, nn.GELU()])
            channels = width
    for size, width in ((8, 128), (16, 64), (32, 32)):
        layers.extend([nn.Upsample(size=(size, size), mode="nearest"),
                       nn.Conv2d(channels, width, 3, padding=1), nn.GELU()])
        channels = width
    layers.extend([nn.Conv2d(channels, 3, 1), nn.Sigmoid()])
    return nn.Sequential(*layers)


@torch.no_grad()
def whiten(layer, images, mean, std):
    """Set 24 RGB filters from overlapping 2 x 2 patches of up to 5000 clean training images."""
    moment = mean.new_zeros((12, 12))
    count = 0
    for batch in images[:5000].split(64):
        normalized = (batch.to(mean.device) - mean) / std
        patches = F.unfold(normalized, 2).transpose(1, 2).reshape(-1, 12)
        moment += patches.T @ patches
        count += len(patches)
    values, vectors = torch.linalg.eigh(moment / count)
    filters = (vectors.T / (values.clamp_min(0) + 5e-4).sqrt()[:, None]).flip(0)
    layer.weight.copy_(torch.cat([filters, -filters]).reshape_as(layer.weight))


def masked_mse(model, target, mean, std, generator):
    """Hide 32 of 64 blocks per image and return mean squared RGB error on hidden pixels."""
    scores = torch.rand(len(target), 64, device=target.device, generator=generator)
    hidden = scores.argsort(dim=1)[:, :32]
    blocks = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, hidden, True)
    mask = blocks.reshape(-1, 1, 8, 8).repeat_interleave(4, 2).repeat_interleave(4, 3)
    inputs = ((target - mean) / std).masked_fill(mask, 0)
    error = (model(inputs) - target).square()
    return (error * mask).sum() / (len(target) * 3 * 512)


def timestamp(device):
    """Return wall time in seconds after outstanding GPU work completes."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def run_variant(variant, train, valid, args, device):
    """Train one matched condition and return per-epoch validation error and training seconds."""
    torch.manual_seed(args.seed)
    order = torch.Generator().manual_seed(args.seed + 1)
    masks = torch.Generator(device=device).manual_seed(args.seed + 2)
    mean = torch.as_tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(CIFAR10_STD, device=device).view_as(mean)
    paper = args.suite == "paper"
    model = reconstruction_net(dirac=not paper or enabled(variant, "dirac")).to(device)
    optimizer = (optimizer_for(model, enabled(variant, "scalebias")) if paper
                 else torch.optim.Adam(model.parameters(), lr=0.001))
    start = timestamp(device)
    if enabled(variant, "frozen_whitening") or variant == "trainable_whitening":
        whiten(model[0], train, mean, std)
        model[0].weight.requires_grad_(variant == "trainable_whitening")
    slow = None
    if enabled(variant, "lookahead"):
        slow = PaperLookahead(model, args.epochs, len(train), args.batch_size)
    step = 0
    flip_rng = torch.Generator().manual_seed(args.seed + 4)
    if paper:
        phase = torch.rand(len(train), generator=flip_rng) < 0.5
    elapsed = timestamp(device) - start
    history = []
    for epoch in range(1, args.epochs + 1):
        start = timestamp(device)
        model.train()
        if paper:
            flips = orientations(phase, epoch, enabled(variant, "alternating_flip"), flip_rng)
            if epoch == 4 and enabled(variant, "frozen_whitening"):
                model[0].bias.requires_grad_(False)
        for indices in torch.randperm(len(train), generator=order).split(args.batch_size):
            target = train[indices].to(device)
            if paper:
                flip = flips[indices].to(device).view(-1, 1, 1, 1)
                target = torch.where(flip, target.flip(-1), target)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model, target, mean, std, masks)
            loss.backward()
            optimizer.step()
            step += 1
            if slow is not None:
                slow.update(model, step)
        elapsed += timestamp(device) - start
        model.eval()
        fixed = torch.Generator(device=device).manual_seed(args.seed + 3)
        total = 0.0
        multi_total = 0.0
        multi_masks = torch.Generator(device=device).manual_seed(args.seed + 3)
        with torch.no_grad():
            for target in valid.split(args.batch_size):
                error = masked_mse(model, target.to(device), mean, std, fixed)
                total += error.item() * len(target)
                if paper:
                    multi = masked_mse(lambda x: six_views(model, x), target.to(device),
                                       mean, std, multi_masks)
                    multi_total += multi.item() * len(target)
        row = dict(epoch=epoch, training_seconds=elapsed, validation_mse=total / len(valid))
        if paper:
            row["six_view_validation_mse"] = multi_total / len(valid)
        history.append(row)
        print(variant, row, flush=True)
    return history


def main():
    """Load local CIFAR-10, compare the selected variants, and write metrics and a timing plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("whitening", "paper"), default="whitening")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/Cifar-10")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this comparison")
    parser.add_argument("--limit", type=int, help="Cap each split for a quick smoke test")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("epochs, batch size, and limit must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = load_cifar10(args.data_dir)[0]
    split = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
    valid_ids, train_ids = split[:5000], split[5000:]
    images = torch.from_numpy(images)
    train = images[train_ids[:args.limit]]
    valid = images[valid_ids[:args.limit]]
    del images
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(suite=args.suite, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                  train_images=len(train), validation_images=len(valid), torch_version=torch.__version__,
                  device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    (args.output / "settings.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for variant in METHODS if args.suite == "paper" else VARIANTS:
        history = run_variant(variant, train, valid, args, device)
        (args.output / f"{variant}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plt.plot([row["training_seconds"] for row in history],
                 [row["validation_mse"] for row in history], marker=".", label=variant)
    plt.xlabel("Training time including whitening (seconds)")
    plt.ylabel("Validation masked RGB mean squared error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
