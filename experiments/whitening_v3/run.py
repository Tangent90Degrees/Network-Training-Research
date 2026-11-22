"""Whitening v3: improved recipe + frozen patch-whitening init."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

from data    import load_cifar10, make_loaders
from vit     import ViT, count_parameters
from tricks  import init_whitening_patch_embed
from trainer import run_experiment

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'Cifar-10')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

DEVICE       = 'cuda:0'
EPOCHS       = 300
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 0.05

if __name__ == '__main__':
    print("=== Whitening v3: improved recipe + frozen patch-whitening ===")
    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    train_loader, val_loader, test_loader = make_loaders(
        x_train, y_train, x_test, y_test,
        batch_size=BATCH_SIZE, use_randaugment=True)

    model = ViT(drop_path_rate=0.1)
    print(f"Parameters: {count_parameters(model):,}")
    init_whitening_patch_embed(model.patch_embed, x_train, eps=5e-4, freeze=True)

    run_experiment(model, train_loader, val_loader, test_loader,
                   epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                   output_dir=OUTPUT_DIR, device=DEVICE,
                   scheduler_type='cosine',
                   use_mixup_cutmix=True, label_smoothing=0.1)
