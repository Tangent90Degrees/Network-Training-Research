"""Experiment: Dirac/identity init v2 — soft interpolation with random init.

Fix: instead of full identity (which collapses all heads to the same
transformation), interpolate between identity and the standard truncated-normal
init at alpha=0.3. This preserves some near-identity structure while allowing
heads to break symmetry more easily.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import torch
import torch.nn as nn
from data    import load_cifar10, make_loaders
from vit     import ViT, count_parameters
from trainer import run_experiment

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Cifar-10')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

DEVICE       = 'cuda:3'
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 0.05
ALPHA        = 0.3   # blend weight toward identity; 0.0 = pure random, 1.0 = pure Dirac


def init_soft_dirac_attention(vit_model, alpha: float = 0.3):
    """Blend attention QKV and output projections toward identity.

    W_new = alpha * I + (1 - alpha) * W_default
    """
    for blk in vit_model.blocks:
        attn = blk.attn
        D = attn.proj.weight.shape[0]
        with torch.no_grad():
            # QKV: blend each D×D sub-block toward identity
            qkv_w = attn.qkv.weight     # [3D, D]
            for k in range(3):
                sub = qkv_w[k*D:(k+1)*D]
                sub.mul_(1 - alpha).add_(alpha * torch.eye(D, device=sub.device))
            # Output projection
            op = attn.proj.weight
            op.mul_(1 - alpha).add_(alpha * torch.eye(D, device=op.device))
    print(f"[soft-dirac] attention projections blended: alpha={alpha} toward identity")


if __name__ == '__main__':
    print(f"=== ViT + Soft Dirac Init (alpha={ALPHA}) ===")
    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    train_loader, val_loader, test_loader = make_loaders(
        x_train, y_train, x_test, y_test,
        batch_size=BATCH_SIZE, alternating_flip=False)

    model = ViT()
    print(f"Parameters: {count_parameters(model):,}")

    init_soft_dirac_attention(model, alpha=ALPHA)

    run_experiment(model, train_loader, val_loader, test_loader,
                   epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                   output_dir=OUTPUT_DIR, device=DEVICE)
