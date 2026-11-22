"""Experiment v2: ScaleBias head + Lookahead — with fixes.

Fixes applied vs original scalebias_lookahead experiment:
  1. ScaleBias init_scale = 1.0 (not 1/9) — prevents gradient suppression
     at the head. The CNN's 1/9 was motivated by 3x3 max-pool aggregating
     9 positions; ViT has no such pooling so 1/9 is arbitrary and harmful.
  2. scheduler_type='cosine' — smooth LR envelope so Lookahead's slow-weight
     pull does not fight a rapidly changing OneCycleLR arc.
"""

import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from data    import load_cifar10, make_loaders
from vit     import ViT, count_parameters
from tricks  import ViTWithScaleBias, Lookahead
from trainer import run_experiment

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Cifar-10')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

DEVICE       = 'cuda:2'
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 0.05

if __name__ == '__main__':
    print("=== ViT + ScaleBias (scale=1.0) + Lookahead (cosine LR) ===")
    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    train_loader, val_loader, test_loader = make_loaders(
        x_train, y_train, x_test, y_test,
        batch_size=BATCH_SIZE, alternating_flip=False)

    base_vit = ViT()
    # Fix 1: init_scale=1.0 instead of 1/9
    model = ViTWithScaleBias(base_vit, embed_dim=192, num_classes=10,
                             init_scale=1.0)
    print(f"Parameters: {count_parameters(model):,}")

    # Fix 2: scheduler_type='cosine' decouples Lookahead from OneCycleLR
    run_experiment(model, train_loader, val_loader, test_loader,
                   epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                   output_dir=OUTPUT_DIR, device=DEVICE,
                   use_lookahead=True, scheduler_type='cosine')
