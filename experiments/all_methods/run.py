"""Experiment: all four paper tricks combined on ViT."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from data    import load_cifar10, make_loaders
from vit     import ViT, count_parameters
from tricks  import (init_whitening_patch_embed, init_dirac_attention,
                     ViTWithScaleBias, Lookahead)
from trainer import run_experiment

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Cifar-10')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

DEVICE       = 'cuda:1'
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 0.05

if __name__ == '__main__':
    print("=== ViT + All Paper Tricks ===")
    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    train_loader, val_loader, test_loader = make_loaders(
        x_train, y_train, x_test, y_test,
        batch_size=BATCH_SIZE, alternating_flip=True)

    base_vit = ViT()
    # 1. Frozen patch-whitening init
    init_whitening_patch_embed(base_vit.patch_embed, x_train, eps=5e-4, freeze=True)
    # 2. Dirac/identity init on attention projections
    init_dirac_attention(base_vit)
    # 3. ScaleBias head
    model = ViTWithScaleBias(base_vit, embed_dim=192, num_classes=10,
                             init_scale=1.0/9.0)
    print(f"Parameters: {count_parameters(model):,}")

    # 4. Lookahead wraps the inner AdamW (done inside run_experiment)
    run_experiment(model, train_loader, val_loader, test_loader,
                   epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                   output_dir=OUTPUT_DIR, device=DEVICE,
                   use_lookahead=True)
