"""Soft-Dirac v3: improved recipe + soft identity init (alpha=0.3)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import torch
from data    import load_cifar10, make_loaders
from vit     import ViT, count_parameters
from trainer import run_experiment

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Cifar-10')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

DEVICE       = 'cuda:3'
EPOCHS       = 300
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 0.05
ALPHA        = 0.3


def init_soft_dirac_attention(vit_model, alpha=0.3):
    for blk in vit_model.blocks:
        attn = blk.attn
        D = attn.proj.weight.shape[0]
        with torch.no_grad():
            qkv_w = attn.qkv.weight
            for k in range(3):
                sub = qkv_w[k*D:(k+1)*D]
                sub.mul_(1 - alpha).add_(alpha * torch.eye(D, device=sub.device))
            op = attn.proj.weight
            op.mul_(1 - alpha).add_(alpha * torch.eye(D, device=op.device))
    print(f"[soft-dirac] alpha={alpha}")


if __name__ == '__main__':
    print(f"=== Soft-Dirac v3: improved recipe + soft identity (alpha={ALPHA}) ===")
    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    train_loader, val_loader, test_loader = make_loaders(
        x_train, y_train, x_test, y_test,
        batch_size=BATCH_SIZE, use_randaugment=True)

    model = ViT(drop_path_rate=0.1)
    print(f"Parameters: {count_parameters(model):,}")
    init_soft_dirac_attention(model, alpha=ALPHA)

    run_experiment(model, train_loader, val_loader, test_loader,
                   epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                   output_dir=OUTPUT_DIR, device=DEVICE,
                   scheduler_type='cosine',
                   use_mixup_cutmix=True, label_smoothing=0.1)
