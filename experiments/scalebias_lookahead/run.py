"""Experiment: ScaleBias head + Lookahead optimizer on ViT."""

import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from data    import load_cifar10, make_loaders
from vit     import ViT, count_parameters
from tricks  import ViTWithScaleBias, Lookahead
from trainer import run_experiment

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Cifar-10')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

DEVICE       = 'cuda:3'
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 0.05

if __name__ == '__main__':
    print("=== ViT + ScaleBias head + Lookahead ===")
    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    train_loader, val_loader, test_loader = make_loaders(
        x_train, y_train, x_test, y_test,
        batch_size=BATCH_SIZE, alternating_flip=False)

    base_vit = ViT()
    model = ViTWithScaleBias(base_vit, embed_dim=192, num_classes=10,
                             init_scale=1.0/9.0)
    print(f"Parameters: {count_parameters(model):,}")

    run_experiment(model, train_loader, val_loader, test_loader,
                   epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                   output_dir=OUTPUT_DIR, device=DEVICE,
                   use_lookahead=True)
